use anyhow::{Context, Result, ensure};
use clap::Parser;
use detection::Word;
use image::RgbImage;
use ort::session::Session;
use rustydoctr::{Metadata, detection, infer, preprocess, recognize, session};
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser, Debug)]
#[command(about = "Upright DB ResNet34 + PARSeq CUDA word OCR vertical slice")]
struct Args {
    #[arg(long,required=true,num_args=1..)]
    image: Vec<PathBuf>,
    #[arg(long, default_value = "models")]
    models: PathBuf,
    #[arg(long, default_value_t = 1024)]
    size: usize,
    #[arg(long, default_value_t = 128)]
    reco_batch: usize,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long, default_value_t = 0.)]
    seconds: f64,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    dump: bool,
    #[arg(long)]
    profile: bool,
    #[arg(long)]
    cpu: bool,
}
#[derive(Default, Serialize)]
struct Stages {
    preprocess: f64,
    detect: f64,
    postprocess: f64,
    crops: f64,
    recognition: f64,
}
#[derive(Serialize)]
struct Page {
    image: String,
    words: Vec<Word>,
    #[serde(rename = "warmup_stages")]
    stages: Stages,
}

fn save_floats(path: PathBuf, data: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write(path, bytes)?;
    Ok(())
}
fn run_page(
    image: &RgbImage,
    path: &Path,
    det: &mut Session,
    reco: &mut Session,
    meta: &Metadata,
    a: &Args,
    dump: Option<&Path>,
) -> Result<Page> {
    let mut stages = Stages::default();
    let start = Instant::now();
    let data = preprocess::prepare(
        image,
        a.size,
        a.size,
        true,
        &meta.db_resnet34.mean,
        &meta.db_resnet34.std,
    );
    stages.preprocess = start.elapsed().as_secs_f64();
    if let Some(d) = dump {
        save_floats(d.join("detector_input.f32"), &data)?;
    }
    let start = Instant::now();
    let (shape, logits) = infer(det, data, [1, 3, a.size, a.size])?;
    ensure!(
        shape == [1, 1, a.size as i64, a.size as i64],
        "Unexpected detector shape: {shape:?}"
    );
    let probabilities: Vec<f32> = logits.iter().map(|x| 1. / (1. + (-x).exp())).collect();
    stages.detect = start.elapsed().as_secs_f64();
    if let Some(d) = dump {
        save_floats(d.join("detector_probability.f32"), &probabilities)?;
    }
    let start = Instant::now();
    let mut words = detection::boxes(
        &probabilities,
        a.size,
        a.size,
        image.height() as usize,
        image.width() as usize,
    );
    stages.postprocess = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut crops = Vec::new();
    for word in &words {
        let [[x0, y0], [x1, y1]] = word.polygon;
        let (w, h) = (image.width() as f32, image.height() as f32);
        let (x0, y0) = (
            (x0 * w).round_ties_even() as u32,
            (y0 * h).round_ties_even() as u32,
        );
        let (x1, y1) = (
            ((x1 * w).round_ties_even() as u32 + 1).min(image.width()),
            ((y1 * h).round_ties_even() as u32 + 1).min(image.height()),
        );
        ensure!(x1 > x0 && y1 > y0, "Empty word crop");
        ensure!(
            (x1 - x0) as f32 / (y1 - y0) as f32 <= 8.,
            "Wide crop requires docTR split/remap support (not yet in this upright slice)"
        );
        crops.push(image::imageops::crop_imm(image, x0, y0, x1 - x0, y1 - y0).to_image());
    }
    stages.crops = start.elapsed().as_secs_f64();
    let start = Instant::now();
    for (batch, crops) in crops.chunks(a.reco_batch).enumerate() {
        let data: Vec<f32> = crops
            .iter()
            .flat_map(|c| {
                preprocess::prepare(c, 32, 128, false, &meta.parseq.mean, &meta.parseq.std)
            })
            .collect();
        if batch == 0
            && let Some(d) = dump
        {
            save_floats(d.join("recognizer_input.f32"), &data)?;
        }
        let (shape, logits) = infer(reco, data, [crops.len(), 3, 32, 128])?;
        for (i, (text, confidence)) in recognize(&logits, &shape, &meta.parseq.vocab)?
            .into_iter()
            .enumerate()
        {
            words[batch * a.reco_batch + i].text = text;
            words[batch * a.reco_batch + i].confidence = confidence;
        }
    }
    stages.recognition = start.elapsed().as_secs_f64();
    Ok(Page {
        image: path.to_string_lossy().into_owned(),
        words,
        stages,
    })
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        a.size > 0 && a.size % 32 == 0 && a.reco_batch > 0 && a.repeats > 0 && a.seconds >= 0.,
        "Invalid size, batch, repetitions or duration"
    );
    ensure!(!a.output.exists(), "Choose a fresh output directory");
    fs::create_dir_all(&a.output)?;
    let meta: Metadata = serde_json::from_slice(&fs::read(a.models.join("metadata.json"))?)?;
    let start = Instant::now();
    let mut det = session(
        &a.models.join("db_resnet34.onnx"),
        a.cpu,
        a.profile.then(|| a.output.join("detector_profile")),
    )
    .context("Loading detector / CUDA provider")?;
    let mut reco = session(
        &a.models.join("parseq.onnx"),
        a.cpu,
        a.profile.then(|| a.output.join("recognizer_profile")),
    )
    .context("Loading recognizer / CUDA provider")?;
    let load_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let images: Vec<_> = a
        .image
        .iter()
        .map(|p| image::open(p).map(|x| x.to_rgb8()))
        .collect::<std::result::Result<_, _>>()?;
    let decode_seconds = start.elapsed().as_secs_f64();
    println!(
        "Models ready: {}; {} pages; warming up",
        if a.cpu {
            "CPU (explicit)"
        } else {
            "CUDA required"
        },
        images.len()
    );
    let start = Instant::now();
    let mut pages = Vec::new();
    for (i, (image, path)) in images.iter().zip(&a.image).enumerate() {
        pages.push(run_page(
            image,
            path,
            &mut det,
            &mut reco,
            &meta,
            &a,
            if a.dump && i == 0 {
                Some(&a.output)
            } else {
                None
            },
        )?);
    }
    let warmup_seconds = start.elapsed().as_secs_f64();
    let mut trials = Vec::new();
    let mut stage_totals = Stages::default();
    while trials.len() < a.repeats || trials.iter().sum::<f64>() < a.seconds {
        let start = Instant::now();
        for (image, path) in images.iter().zip(&a.image) {
            let p = run_page(image, path, &mut det, &mut reco, &meta, &a, None)?;
            stage_totals.preprocess += p.stages.preprocess;
            stage_totals.detect += p.stages.detect;
            stage_totals.postprocess += p.stages.postprocess;
            stage_totals.crops += p.stages.crops;
            stage_totals.recognition += p.stages.recognition;
        }
        let seconds = start.elapsed().as_secs_f64();
        trials.push(seconds);
        println!(
            "Pass {}: {:.3}s ({:.2} pages/s)",
            trials.len(),
            seconds,
            images.len() as f64 / seconds
        );
    }
    let result = serde_json::json!({"backend":if a.cpu{"ORT CPU"}else{"ORT CUDA (TF32 disabled)"},"detector":"db_resnet34","recognizer":"parseq",
        "size":a.size,"reco_batch":a.reco_batch,"model_load_seconds":load_seconds,"image_decode_seconds":decode_seconds,"runtime":ort::info(),
        "model_metadata":serde_json::from_slice::<serde_json::Value>(&fs::read(a.models.join("metadata.json"))?)?,"timed_stage_totals_seconds":stage_totals,
        "warmup_seconds":warmup_seconds,"trials_seconds":trials,"pages_per_second":images.len() as f64*trials.len() as f64/trials.iter().sum::<f64>(),
        "scope":"Upright word OCR, in-memory pages, no rotation, no wide-crop splitting or document layout. Load/decode/warmup/JSON write excluded from timings. PARSeq full-length ONNX decoding.","pages":pages});
    fs::write(
        a.output.join("result.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    if a.profile {
        println!(
            "Profiles: {}, {}",
            det.end_profiling()?,
            reco.end_profiling()?
        );
    }
    println!("Results: {}", a.output.display());
    Ok(())
}
