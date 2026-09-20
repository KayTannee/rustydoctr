//! Bounded decode/prepare -> detection -> CPU crops -> recognition -> ordered writer.
use crate::{
    Metadata,
    crops::{self, Mapping},
    detection::{self, Word},
    infer, preprocess, recognize,
};
use anyhow::{Result, anyhow, ensure};
use crossbeam_channel::{Receiver, RecvTimeoutError, SendTimeoutError, Sender, bounded};
use image::RgbImage;
use ort::{
    execution_providers::{CUDAExecutionProvider, cuda::CuDNNConvAlgorithmSearch},
    session::{Session, builder::GraphOptimizationLevel},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub size: usize,
    pub reco_batch: usize,
    pub det_batch: usize,
    pub workers: usize,
    pub inflight: usize,
    pub arena_mib: usize,
    pub det_arena_mib: usize,
    pub vram_limit_mib: usize,
    pub seconds: f64,
    pub pages: usize,
}
#[derive(Clone, Deserialize)]
pub struct InputPage {
    pub id: String,
    pub image: PathBuf,
}
#[derive(Deserialize)]
pub struct Workload {
    pub pages: Vec<InputPage>,
}
#[derive(Default)]
struct Metrics {
    read: AtomicU64,
    admission_wait: AtomicU64,
    det: AtomicU64,
    det_wait: AtomicU64,
    post: AtomicU64,
    post_send_wait: AtomicU64,
    reco: AtomicU64,
    reco_wait: AtomicU64,
    active: AtomicUsize,
    high: AtomicUsize,
}
fn add(counter: &AtomicU64, start: Instant) {
    counter.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
}
pub(crate) fn send<T>(tx: &Sender<T>, mut value: T, abort: &AtomicBool) -> Result<()> {
    loop {
        ensure!(!abort.load(Ordering::Relaxed), "Pipeline cancelled");
        match tx.send_timeout(value, Duration::from_millis(50)) {
            Ok(()) => return Ok(()),
            Err(SendTimeoutError::Timeout(v)) => value = v,
            Err(SendTimeoutError::Disconnected(_)) => {
                return Err(anyhow!("Downstream disconnected"));
            }
        }
    }
}
pub(crate) fn recv<T>(rx: &Receiver<T>, abort: &AtomicBool) -> Result<Option<T>> {
    loop {
        ensure!(!abort.load(Ordering::Relaxed), "Pipeline cancelled");
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(x) => return Ok(Some(x)),
            Err(RecvTimeoutError::Disconnected) => return Ok(None),
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
}
fn checked<T: Send>(abort: &AtomicBool, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|_| Err(anyhow!("Pipeline worker panicked")));
    if let Err(e) = &r {
        eprintln!("Pipeline stage error: {e:#}");
        abort.store(true, Ordering::Relaxed);
    }
    r
}
struct Prepared {
    id: String,
    sequence: usize,
    page: usize,
    admitted: Instant,
    image: RgbImage,
    tensor: Vec<f32>,
}
struct Detected {
    prepared: Prepared,
    probability: Vec<f32>,
}
struct Accumulator {
    id: String,
    sequence: usize,
    page: usize,
    admitted: Instant,
    words: Vec<Word>,
    maps: Vec<Mapping>,
    parts: Vec<Option<(String, f32)>>,
    remaining: usize,
}
type Reference = (Arc<Mutex<Accumulator>>, usize);
struct Chunk {
    tensor: Vec<f32>,
    references: Vec<Reference>,
}
struct Complete {
    id: String,
    sequence: usize,
    page: usize,
    admitted: Instant,
    words: Vec<Word>,
}
#[derive(Serialize)]
pub struct Record {
    pub id: String,
    pub sequence: usize,
    pub page: usize,
    pub words: Vec<Word>,
}
#[derive(Serialize)]
pub struct Stats {
    pub pages: usize,
    pub words: usize,
    pub wall_seconds: f64,
    pub pages_per_second: f64,
    pub latency_seconds: BTreeMap<String, f64>,
    pub first_pages: BTreeMap<usize, Record>,
    pub timeline: Vec<(f64, usize)>,
    pub max_inflight: usize,
    pub stage_seconds: BTreeMap<String, f64>,
}

pub fn sessions(models: &std::path::Path, config: &Config) -> Result<(Session, Session)> {
    fn make(path: PathBuf, limit: usize) -> Result<Session> {
        Ok(Session::builder()?
            .with_intra_threads(1)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CUDAExecutionProvider::default()
                .with_tf32(false)
                .with_memory_limit(limit)
                .with_conv_algorithm_search(CuDNNConvAlgorithmSearch::Heuristic)
                .build()
                .error_on_failure()])?
            .commit_from_file(path)?)
    }
    ensure!(
        config.det_arena_mib > 0 && config.det_arena_mib < config.arena_mib,
        "Invalid arena split"
    );
    let det_bytes = config.det_arena_mib * 1024 * 1024;
    let reco_bytes = (config.arena_mib - config.det_arena_mib) * 1024 * 1024;
    Ok((
        make(models.join("db_resnet34.onnx"), det_bytes)?,
        make(models.join("parseq.onnx"), reco_bytes)?,
    ))
}

pub fn run(
    config: &Config,
    inputs: &[InputPage],
    meta: &Metadata,
    det: Session,
    reco: Session,
    output: Option<PathBuf>,
    cancel: Option<&AtomicBool>,
) -> Result<(Stats, Session, Session)> {
    ensure!(!inputs.is_empty(), "Empty workload");
    let began = Instant::now();
    let mut sequence = 0;
    let source = || {
        if (config.pages > 0 && sequence >= config.pages)
            || (config.pages == 0
                && sequence > 0
                && sequence % inputs.len() == 0
                && began.elapsed().as_secs_f64() >= config.seconds)
        {
            return Ok(None);
        }
        let page = sequence % inputs.len();
        sequence += 1;
        Ok(Some(Frame {
            page,
            id: inputs[page].id.clone(),
            image: PageImage::File(inputs[page].image.clone()),
        }))
    };
    let progress = if output.is_some() { inputs.len() } else { 0 };
    let file = output.map(File::create).transpose()?.map(BufWriter::new);
    run_stream(
        config,
        meta,
        det,
        reco,
        RunIo {
            source,
            sink: FileSink(file),
            capture_first: inputs.len(),
            progress,
        },
        cancel,
    )
}

pub enum PageImage {
    File(PathBuf),
    Rgb(RgbImage),
}
pub struct Frame {
    pub page: usize,
    pub id: String,
    pub image: PageImage,
}
pub trait Sink: Send {
    fn write(&mut self, record: &Record) -> Result<()>;
    fn finish(&mut self) -> Result<()> {
        Ok(())
    }
}
struct FileSink(Option<BufWriter<File>>);
impl Sink for FileSink {
    fn write(&mut self, record: &Record) -> Result<()> {
        if let Some(f) = &mut self.0 {
            serde_json::to_writer(&mut *f, record)?;
            f.write_all(b"\n")?;
            f.flush()?;
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<()> {
        if let Some(f) = &mut self.0 {
            f.flush()?;
            f.get_ref().sync_all()?;
        }
        Ok(())
    }
}
pub struct RunIo<F, S> {
    pub source: F,
    pub sink: S,
    pub capture_first: usize,
    pub progress: usize,
}
pub fn run_stream<F, S>(
    config: &Config,
    meta: &Metadata,
    mut det: Session,
    mut reco: Session,
    io: RunIo<F, S>,
    cancel: Option<&AtomicBool>,
) -> Result<(Stats, Session, Session)>
where
    F: FnMut() -> Result<Option<Frame>> + Send,
    S: Sink,
{
    let RunIo {
        mut source,
        mut sink,
        capture_first,
        progress,
    } = io;
    ensure!(
        config.inflight > 0 && config.workers > 0 && config.det_batch > 0 && config.reco_batch > 0,
        "Empty workload/invalid pipeline capacity"
    );
    let local_abort = AtomicBool::new(false);
    let abort = cancel.unwrap_or(&local_abort);
    let metrics = Metrics::default();
    let began = Instant::now();
    let (prep_tx, prep_rx) = bounded::<Prepared>(2);
    let (det_tx, det_rx) = bounded::<Detected>(2);
    let (crop_tx, crop_rx) = bounded::<Chunk>(2);
    let (done_tx, done_rx) = bounded::<Complete>(config.inflight);
    let (credit_tx, credit_rx) = bounded::<()>(config.inflight);
    for _ in 0..config.inflight {
        credit_tx.send(())?;
    }
    thread::scope(|scope| -> Result<(Stats, Session, Session)> {
        let reader = scope.spawn(|| {
            checked(abort, || {
                let mut sequence = 0;
                loop {
                    let wait = Instant::now();
                    if recv(&credit_rx, abort)?.is_none() {
                        break;
                    }
                    add(&metrics.admission_wait, wait);
                    let Some(frame) = source()? else {
                        break;
                    };
                    let active = metrics.active.fetch_add(1, Ordering::Relaxed) + 1;
                    metrics.high.fetch_max(active, Ordering::Relaxed);
                    let admitted = Instant::now();
                    let page = frame.page;
                    let start = Instant::now();
                    let image = match frame.image {
                        PageImage::File(path) => image::open(path)?.to_rgb8(),
                        PageImage::Rgb(image) => image,
                    };
                    let tensor = preprocess::prepare(
                        &image,
                        config.size,
                        config.size,
                        true,
                        &meta.db_resnet34.mean,
                        &meta.db_resnet34.std,
                    );
                    add(&metrics.read, start);
                    send(
                        &prep_tx,
                        Prepared {
                            id: frame.id,
                            sequence,
                            page,
                            admitted,
                            image,
                            tensor,
                        },
                        abort,
                    )?;
                    sequence += 1;
                }
                drop(prep_tx);
                drop(credit_rx);
                Ok(())
            })
        });
        let detector = scope.spawn(|| {
            checked(abort, || {
                loop {
                    let wait = Instant::now();
                    let Some(first) = recv(&prep_rx, abort)? else {
                        break;
                    };
                    add(&metrics.det_wait, wait);
                    let mut batch = vec![first];
                    while batch.len() < config.det_batch {
                        match prep_rx.recv_timeout(Duration::from_millis(2)) {
                            Ok(p) => batch.push(p),
                            Err(_) => break,
                        }
                    }
                    let start = Instant::now();
                    let mut data = Vec::with_capacity(batch.len() * 3 * config.size * config.size);
                    for p in &mut batch {
                        data.append(&mut p.tensor);
                    }
                    let (shape, logits) =
                        infer(&mut det, data, [batch.len(), 3, config.size, config.size])?;
                    ensure!(
                        shape
                            == [
                                batch.len() as i64,
                                1,
                                config.size as i64,
                                config.size as i64
                            ],
                        "Unexpected detector shape"
                    );
                    add(&metrics.det, start);
                    for (prepared, raw) in batch
                        .into_iter()
                        .zip(logits.chunks(config.size * config.size))
                    {
                        let probability = raw.iter().map(|x| 1. / (1. + (-x).exp())).collect();
                        send(
                            &det_tx,
                            Detected {
                                prepared,
                                probability,
                            },
                            abort,
                        )?;
                    }
                }
                drop(det_tx);
                Ok(det)
            })
        });
        let mut post_workers = Vec::new();
        for _ in 0..config.workers {
            let rx = det_rx.clone();
            let tx = crop_tx.clone();
            let done = done_tx.clone();

            let metrics = &metrics;
            post_workers.push(scope.spawn(move || {
                checked(abort, || {
                    while let Some(item) = recv(&rx, abort)? {
                        let start = Instant::now();
                        let p = item.prepared;
                        let words = detection::boxes(
                            &item.probability,
                            config.size,
                            config.size,
                            p.image.height() as usize,
                            p.image.width() as usize,
                        );
                        let (crops, maps) = crops::extract(&p.image, &words)?;
                        drop(p.image);
                        let count = crops.len();
                        let page = Arc::new(Mutex::new(Accumulator {
                            id: p.id.clone(),
                            sequence: p.sequence,
                            page: p.page,
                            admitted: p.admitted,
                            words,
                            maps,
                            parts: vec![None; count],
                            remaining: count,
                        }));
                        add(&metrics.post, start);
                        if count == 0 {
                            send(
                                &done,
                                Complete {
                                    id: p.id.clone(),
                                    sequence: p.sequence,
                                    page: p.page,
                                    admitted: p.admitted,
                                    words: vec![],
                                },
                                abort,
                            )?;
                            continue;
                        }
                        for (batch, images) in crops.chunks(config.reco_batch).enumerate() {
                            let start = Instant::now();
                            let tensor = images
                                .iter()
                                .flat_map(|image| {
                                    preprocess::prepare(
                                        image,
                                        32,
                                        128,
                                        false,
                                        &meta.parseq.mean,
                                        &meta.parseq.std,
                                    )
                                })
                                .collect();
                            let references = (0..images.len())
                                .map(|i| (page.clone(), batch * config.reco_batch + i))
                                .collect();
                            add(&metrics.post, start);
                            let wait = Instant::now();
                            send(&tx, Chunk { tensor, references }, abort)?;
                            add(&metrics.post_send_wait, wait);
                        }
                    }
                    Ok(())
                })
            }));
        }
        drop(det_rx);
        drop(crop_tx);
        let recognition = scope.spawn(|| {
            checked(abort, || {
                let mut data = Vec::with_capacity(config.reco_batch * 3 * 32 * 128);
                let mut references = Vec::with_capacity(config.reco_batch);
                fn flush(
                    reco: &mut Session,
                    data: &mut Vec<f32>,
                    references: &mut Vec<Reference>,
                    meta: &Metadata,
                    done: &Sender<Complete>,
                    abort: &AtomicBool,
                    metrics: &Metrics,
                ) -> Result<()> {
                    if references.is_empty() {
                        return Ok(());
                    }
                    let start = Instant::now();
                    let n = references.len();
                    let (shape, logits) = infer(reco, std::mem::take(data), [n, 3, 32, 128])?;
                    let decoded = recognize(&logits, &shape, &meta.parseq.vocab)?;
                    ensure!(decoded.len() == n, "Recognition count mismatch");
                    for ((page, index), pred) in references.drain(..).zip(decoded) {
                        let mut p = page.lock().map_err(|_| anyhow!("Page state poisoned"))?;
                        ensure!(p.parts[index].is_none(), "Duplicate crop result");
                        p.parts[index] = Some(pred);
                        p.remaining -= 1;
                        if p.remaining == 0 {
                            let parts = p
                                .parts
                                .iter_mut()
                                .map(|x| x.take().unwrap())
                                .collect::<Vec<_>>();
                            let mut words = std::mem::take(&mut p.words);
                            crops::remap(&mut words, &parts, &p.maps);
                            send(
                                done,
                                Complete {
                                    id: p.id.clone(),
                                    sequence: p.sequence,
                                    page: p.page,
                                    admitted: p.admitted,
                                    words,
                                },
                                abort,
                            )?;
                        }
                    }
                    add(&metrics.reco, start);
                    Ok(())
                }
                loop {
                    ensure!(!abort.load(Ordering::Relaxed), "Pipeline cancelled");
                    let wait = Instant::now();
                    let item = crop_rx.recv_timeout(Duration::from_millis(5));
                    add(&metrics.reco_wait, wait);
                    match item {
                        Ok(chunk) => {
                            for (input, reference) in
                                chunk.tensor.chunks(3 * 32 * 128).zip(chunk.references)
                            {
                                data.extend_from_slice(input);
                                references.push(reference);
                                if references.len() == config.reco_batch {
                                    flush(
                                        &mut reco,
                                        &mut data,
                                        &mut references,
                                        meta,
                                        &done_tx,
                                        abort,
                                        &metrics,
                                    )?;
                                }
                            }
                        }
                        Err(RecvTimeoutError::Timeout) => flush(
                            &mut reco,
                            &mut data,
                            &mut references,
                            meta,
                            &done_tx,
                            abort,
                            &metrics,
                        )?,
                        Err(RecvTimeoutError::Disconnected) => {
                            flush(
                                &mut reco,
                                &mut data,
                                &mut references,
                                meta,
                                &done_tx,
                                abort,
                                &metrics,
                            )?;
                            break;
                        }
                    }
                }
                drop(done_tx);
                Ok(reco)
            })
        });
        let writer = scope.spawn(|| {
            checked(abort, || {
                let mut pending = BTreeMap::new();
                let mut count = 0;
                let mut words = 0;
                let mut latencies = Vec::new();
                let mut first_pages = BTreeMap::new();
                let mut timeline = Vec::new();
                while let Some(page) = recv(&done_rx, abort)? {
                    ensure!(
                        pending.insert(page.sequence, page).is_none(),
                        "Duplicate page result"
                    );
                    while let Some(page) = pending.remove(&count) {
                        words += page.words.len();
                        let record = Record {
                            id: page.id,
                            sequence: page.sequence,
                            page: page.page,
                            words: page.words,
                        };
                        sink.write(&record)?;
                        let latency = page.admitted.elapsed().as_secs_f64();
                        if latencies.len() < 4096 {
                            latencies.push(latency);
                        } else {
                            latencies[count % 4096] = latency;
                        }
                        if page.page < capture_first {
                            first_pages.entry(page.page).or_insert(record);
                        }
                        count += 1;
                        metrics.active.fetch_sub(1, Ordering::Relaxed);
                        let _ = credit_tx.try_send(());
                        if timeline.len() == 8192 {
                            timeline.drain(..4096);
                        }
                        timeline.push((began.elapsed().as_secs_f64(), count));
                        if progress > 0 && count % progress == 0 {
                            println!("{count} pages; {:.1}s", began.elapsed().as_secs_f64());
                        }
                    }
                }
                ensure!(pending.is_empty(), "Missing page before pending results");
                ensure!(
                    metrics.active.load(Ordering::Relaxed) == 0,
                    "Missing admitted pages"
                );
                sink.finish()?;
                let wall = began.elapsed().as_secs_f64();
                latencies.sort_by(f64::total_cmp);
                let mut latency_seconds = BTreeMap::new();
                for p in [50, 95, 99] {
                    latency_seconds.insert(
                        format!("p{p}"),
                        if latencies.is_empty() {
                            0.
                        } else {
                            latencies[((latencies.len() - 1) * p) / 100]
                        },
                    );
                }
                let stage_seconds = [
                    ("decode_prepare", &metrics.read),
                    ("admission_wait", &metrics.admission_wait),
                    ("detection", &metrics.det),
                    ("detector_input_wait", &metrics.det_wait),
                    ("postprocess_crop_prepare", &metrics.post),
                    ("crop_queue_wait", &metrics.post_send_wait),
                    ("recognition", &metrics.reco),
                    ("recognizer_input_wait", &metrics.reco_wait),
                ]
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v.load(Ordering::Relaxed) as f64 / 1e9))
                .collect();
                Ok(Stats {
                    pages: count,
                    words,
                    wall_seconds: wall,
                    pages_per_second: count as f64 / wall,
                    latency_seconds,
                    first_pages,
                    timeline,
                    max_inflight: metrics.high.load(Ordering::Relaxed),
                    stage_seconds,
                })
            })
        });
        let r = reader.join().map_err(|_| anyhow!("Reader panic"))?;
        let d = detector.join().map_err(|_| anyhow!("Detector panic"))?;
        let mut post_error = None;
        for w in post_workers {
            if let Err(e) = w.join().map_err(|_| anyhow!("Postprocessor panic"))? {
                post_error = Some(e);
            }
        }
        let rec = recognition
            .join()
            .map_err(|_| anyhow!("Recognizer panic"))?;
        let written = writer.join().map_err(|_| anyhow!("Writer panic"))?;
        // Prefer the originating failure over the cancellation it caused elsewhere.
        for (stage, error) in [
            ("reader", r.as_ref().err()),
            ("detector", d.as_ref().err()),
            ("postprocessor", post_error.as_ref()),
            ("recognizer", rec.as_ref().err()),
            ("writer", written.as_ref().err()),
        ] {
            if let Some(error) = error {
                let message = error.to_string();
                if ![
                    "Pipeline cancelled",
                    "Missing admitted pages",
                    "Downstream disconnected",
                ]
                .contains(&message.as_str())
                {
                    return Err(anyhow!("{stage}: {error:#}"));
                }
            }
        }
        r?;
        if let Some(e) = post_error {
            return Err(e);
        }
        Ok((written?, d?, rec?))
    })
}
