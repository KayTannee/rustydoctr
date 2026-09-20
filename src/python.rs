//! Python never runs on inference workers. Blocking operations release the GIL.
use crate::{
    Metadata,
    pipeline::{self, Config, Frame, PageImage, Record, RunIo, Sink},
};
use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, Sender, bounded};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
};

fn pyerr(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}
struct Output {
    tx: Sender<String>,
    abort: Arc<AtomicBool>,
}
impl Sink for Output {
    fn write(&mut self, record: &Record) -> Result<()> {
        pipeline::send(&self.tx, serde_json::to_string(record)?, &self.abort)
    }
}

#[pyclass]
struct RawStream {
    input: Mutex<Option<Sender<Frame>>>,
    output: Receiver<String>,
    abort: Arc<AtomicBool>,
    outcome: Arc<Mutex<Option<Result<String, String>>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
}
impl RawStream {
    fn stop(&self) {
        self.abort.store(true, Ordering::Relaxed);
    }
    fn join(&self) -> PyResult<()> {
        if let Some(handle) = self.worker.lock().map_err(pyerr)?.take() {
            handle
                .join()
                .map_err(|_| pyerr("Pipeline worker panicked"))?;
        }
        Ok(())
    }
}
#[pymethods]
impl RawStream {
    #[new]
    fn new(py: Python<'_>, models: PathBuf, config_json: &str) -> PyResult<Self> {
        let config: Config = serde_json::from_str(config_json).map_err(pyerr)?;
        if config.deskew && !config.page_orientation {
            return Err(PyValueError::new_err("deskew requires page_orientation"));
        }
        if config.inflight == 0
            || config.workers == 0
            || config.det_batch == 0
            || config.reco_batch == 0
            || config.size < 32
            || !config.size.is_multiple_of(32)
        {
            return Err(PyValueError::new_err(
                "Positive capacities and detector size divisible by 32 required",
            ));
        }
        py.detach(move || {
            let meta: Metadata = serde_json::from_slice(
                &std::fs::read(models.join("metadata.json")).map_err(pyerr)?,
            )
            .map_err(pyerr)?;
            let (input, rx) = bounded(1);
            let (tx, output) = bounded(config.inflight);
            let abort = Arc::new(AtomicBool::new(false));
            let outcome = Arc::new(Mutex::new(None));
            let cancel = abort.clone();
            let result = outcome.clone();
            // Load before returning, so initialization failures reach the constructor.
            let baseline = if config.vram_limit_mib > 0 {
                Some(
                    crate::gpu::Nvml::new()
                        .and_then(|n| n.memory())
                        .map_err(pyerr)?
                        .used,
                )
            } else {
                None
            };
            let sessions = pipeline::sessions(&models, &config).map_err(pyerr)?;
            if let Some(base) = baseline {
                let used = crate::gpu::Nvml::new()
                    .and_then(|n| n.memory())
                    .map_err(pyerr)?
                    .used;
                if used.saturating_sub(base) > config.vram_limit_mib as u64 * 1024 * 1024 {
                    return Err(pyerr(
                        "VRAM exceeded configured incremental budget during loading",
                    ));
                }
            }
            let worker = thread::spawn(move || {
                // Keep the result channel alive until the terminal status is published.
                let keepalive = tx.clone();
                let guard_done = Arc::new(AtomicBool::new(false));
                let guard_error = Arc::new(Mutex::new(None));
                let guard = baseline.map(|base| {
                    let done = guard_done.clone();
                    let error = guard_error.clone();
                    let abort = cancel.clone();
                    let limit = config.vram_limit_mib as u64 * 1024 * 1024;
                    thread::spawn(move || {
                        let check = || -> Result<()> {
                            let nvml = crate::gpu::Nvml::new()?;
                            while !done.load(Ordering::Relaxed) {
                                anyhow::ensure!(
                                    nvml.memory()?.used.saturating_sub(base) <= limit,
                                    "VRAM exceeded configured incremental budget"
                                );
                                thread::sleep(std::time::Duration::from_millis(100));
                            }
                            Ok(())
                        };
                        if let Err(e) = check() {
                            *error.lock().unwrap() = Some(format!("{e:#}"));
                            abort.store(true, Ordering::Relaxed);
                        }
                    })
                });
                let mut run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let source = || pipeline::recv(&rx, &cancel);
                    let sink = Output {
                        tx,
                        abort: cancel.clone(),
                    };
                    pipeline::run_stream(
                        &config,
                        &meta,
                        sessions,
                        RunIo {
                            source,
                            sink,
                            capture_first: 0,
                            progress: 0,
                        },
                        Some(&cancel),
                    )
                }))
                .unwrap_or_else(|_| Err(anyhow!("Pipeline worker panicked")));
                guard_done.store(true, Ordering::Relaxed);
                if let Some(guard) = guard {
                    let _ = guard.join();
                }
                if let Some(error) = guard_error.lock().unwrap().take() {
                    run = Err(anyhow!(error));
                }
                if run.is_err() {
                    cancel.store(true, Ordering::Relaxed);
                }
                *result.lock().unwrap() = Some(match &run {
                    Ok((stats, _)) => serde_json::to_string(stats).map_err(|e| e.to_string()),
                    Err(e) => Err(format!("{e:#}")),
                });
                drop(keepalive);
                // Match CLI timing: retain both sessions until explicit close, outside drain timing.
                while run.is_ok() && !cancel.load(Ordering::Relaxed) {
                    thread::sleep(std::time::Duration::from_millis(20));
                }
                drop(run);
            });
            Ok(Self {
                input: Mutex::new(Some(input)),
                output,
                abort,
                outcome,
                worker: Mutex::new(Some(worker)),
            })
        })
    }

    fn submit_rgb(
        &self,
        py: Python<'_>,
        id: String,
        width: u32,
        height: u32,
        data: &Bound<'_, PyBytes>,
    ) -> PyResult<()> {
        let bytes = data.as_bytes();
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(3));
        if width == 0 || height == 0 || expected != Some(bytes.len()) {
            return Err(PyValueError::new_err(
                "Expected tightly packed RGB bytes: width * height * 3",
            ));
        }
        py.detach(|| {
            // Lock before copying: concurrent submitters cannot allocate an unbounded backlog.
            let guard = self.input.lock().map_err(pyerr)?;
            let tx = guard
                .as_ref()
                .ok_or_else(|| pyerr("Input already finished"))?;
            let image = image::RgbImage::from_raw(width, height, bytes.to_vec()).unwrap();
            pipeline::send(
                tx,
                Frame {
                    page: 0,
                    id,
                    image: PageImage::Rgb(image),
                },
                &self.abort,
            )
            .map_err(pyerr)
        })
    }
    fn finish_input(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.input.lock().map_err(pyerr)?.take();
            Ok(())
        })
    }
    fn recv_json(&self, py: Python<'_>) -> PyResult<Option<String>> {
        py.detach(|| {
            // Drain queued results, then expose the original worker failure.
            match self.output.recv() {
                Ok(value) => Ok(Some(value)),
                Err(_) => match self.outcome.lock().map_err(pyerr)?.as_ref() {
                    Some(Err(e)) => Err(pyerr(e)),
                    Some(Ok(_)) => Ok(None),
                    None => Err(pyerr("Worker ended without status")),
                },
            }
        })
    }
    fn stats_json(&self) -> PyResult<String> {
        match self.outcome.lock().map_err(pyerr)?.as_ref() {
            Some(Ok(stats)) => Ok(stats.clone()),
            Some(Err(e)) => Err(pyerr(e)),
            None => Err(pyerr("Drain results before requesting final statistics")),
        }
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.stop();
        py.detach(|| self.join())
    }
}
impl Drop for RawStream {
    fn drop(&mut self) {
        self.stop();
        if let Ok(worker) = self.worker.get_mut()
            && let Some(handle) = worker.take()
        {
            let _ = handle.join();
        }
    }
}
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RawStream>()
}
