use anyhow::{Context, Result, ensure};
use clap::Parser;
use rustydoctr::{
    Metadata,
    gpu::{Nvml, process_stats},
    pipeline::{self, Config, Workload},
};
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
#[derive(Parser, Debug)]
#[command(about = "Bounded CPU/GPU word-OCR pipeline with optional measured batch calibration")]
struct Args {
    /// Experimental bounded dense-text rescans before recognition.
    #[arg(long)]
    dense_refine: bool,
    #[arg(long, default_value = "testdata/throughput.json")]
    workload: PathBuf,
    #[arg(long, default_value = "models")]
    models: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 1536)]
    size: usize,
    #[arg(long, default_value_t = 256)]
    reco_batch: usize,
    #[arg(long, default_value_t = 1)]
    det_batch: usize,
    #[arg(long, default_value_t = 300.)]
    seconds: f64,
    #[arg(long, default_value_t = 0)]
    pages: usize,
    #[arg(long, default_value_t = 0)]
    arena_mib: usize,
    #[arg(long, default_value_t = 0)]
    det_arena_mib: usize,
    /// Stop if sampled incremental device memory exceeds this budget (0 disables).
    #[arg(long, default_value_t = 0)]
    vram_limit_mib: usize,
    /// Calibrate for a 4 GiB GPU, keeping both FP32 models resident.
    #[arg(long)]
    vram_4gb: bool,
    #[arg(long, default_value_t = 512)]
    host_mib: usize,
    #[arg(long, default_value_t = 0)]
    inflight: usize,
    #[arg(long, default_value_t = 0)]
    workers: usize,
    #[arg(long)]
    auto: bool,
    /// Prefer the least-memory trial within this fraction of peak throughput.
    #[arg(long, default_value_t = 0.03)]
    throughput_tolerance: f64,
}
fn choose_candidate(candidates: &[(f64, u64)], tolerance: f64) -> Option<usize> {
    let fastest = candidates.iter().map(|c| c.0).fold(0., f64::max);
    candidates
        .iter()
        .enumerate()
        .filter(|(_, (pps, _))| *pps >= fastest * (1. - tolerance))
        .min_by_key(|(_, (_, memory))| *memory)
        .map(|(index, _)| index)
}
fn now() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}
fn signature(summary: &Value) -> Value {
    let mut result = serde_json::Map::new();
    for (k, p) in summary["first_pages"].as_object().unwrap() {
        result.insert(
            k.clone(),
            Value::Array(
                p["words"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|w| w["text"].clone())
                    .collect(),
            ),
        );
    }
    Value::Object(result)
}
fn child(a: &Args, config: &Config, output: &Path, pages: usize, seconds: f64) -> Result<bool> {
    let mut command = Command::new(std::env::current_exe()?);
    if config.dense_refine {
        command.arg("--dense-refine");
    }
    let status = command
        .args([
            "--workload",
            a.workload.to_str().unwrap(),
            "--models",
            a.models.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--size",
            &config.size.to_string(),
            "--reco-batch",
            &config.reco_batch.to_string(),
            "--det-batch",
            &config.det_batch.to_string(),
            "--workers",
            &config.workers.to_string(),
            "--inflight",
            &config.inflight.to_string(),
            "--arena-mib",
            &config.arena_mib.to_string(),
            "--det-arena-mib",
            &config.det_arena_mib.to_string(),
            "--vram-limit-mib",
            &config.vram_limit_mib.to_string(),
            "--host-mib",
            &a.host_mib.to_string(),
            "--pages",
            &pages.to_string(),
            "--seconds",
            &seconds.to_string(),
        ])
        .status()?;
    Ok(status.success())
}
fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(
        a.size > 0
            && a.size % 32 == 0
            && a.reco_batch > 0
            && a.det_batch > 0
            && a.seconds >= 0.
            && a.seconds.is_finite()
            && (0.0..1.0).contains(&a.throughput_tolerance)
            && a.host_mib > 0,
        "Invalid configuration"
    );
    ensure!(!a.output.exists(), "Choose a fresh output directory");
    fs::create_dir_all(&a.output)?;
    let workload: Workload = serde_json::from_slice(&fs::read(&a.workload)?)?;
    ensure!(!workload.pages.is_empty(), "Empty workload");
    let available = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(2);
    let memory = Nvml::new().ok().and_then(|n| n.memory().ok());
    ensure!(
        !(a.vram_4gb && a.vram_limit_mib > 0),
        "Use either --vram-4gb or an explicit --vram-limit-mib"
    );
    let vram_limit_mib = if a.vram_4gb {
        let memory = memory.context("The 4 GB profile requires NVML memory telemetry")?;
        3584.min((memory.free as f64 / 1048576. * 0.9) as usize)
    } else {
        a.vram_limit_mib
    };
    ensure!(
        vram_limit_mib == 0 || (memory.is_some() && vram_limit_mib >= 1024),
        "Insufficient free VRAM or missing NVML for requested memory target"
    );
    let arena_mib = if a.arena_mib > 0 {
        a.arena_mib
    } else if a.vram_4gb {
        vram_limit_mib.saturating_sub(512).min(3072)
    } else {
        memory
            .map(|m| ((m.free as f64 * 0.6) as usize / 1048576).min(8192))
            .unwrap_or(2048)
    };
    ensure!(
        arena_mib >= 512,
        "Not enough free GPU memory for the minimum arena budget"
    );
    ensure!(
        !a.vram_4gb || arena_mib <= vram_limit_mib.saturating_sub(512),
        "4 GB profile reserves 512 MiB beyond CUDA arenas"
    );
    let det_arena_mib = if a.det_arena_mib > 0 {
        a.det_arena_mib
    } else if a.vram_4gb {
        2048.min(arena_mib.saturating_sub(512))
    } else {
        arena_mib / 4
    };
    let max_pixels = workload
        .pages
        .iter()
        .map(|p| image::image_dimensions(&p.image).map(|(w, h)| w as usize * h as usize))
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap();
    let per_page = max_pixels * (if a.dense_refine { 18 } else { 9 })
        + a.size * a.size * 16
        + if a.dense_refine {
            2 * 1024 * 1024 * 16
        } else {
            0
        };
    let automatic_inflight =
        (a.host_mib * 1048576 / per_page).clamp(1, if a.vram_4gb { 2 } else { 8 });
    let config = Config {
        dense_refine: a.dense_refine,
        size: a.size,
        reco_batch: a.reco_batch,
        det_batch: a.det_batch,
        workers: if a.workers > 0 {
            a.workers
        } else {
            (available / 4).clamp(1, 2)
        },
        inflight: if a.inflight > 0 {
            a.inflight
        } else {
            automatic_inflight
        },
        arena_mib,
        det_arena_mib,
        vram_limit_mib,
        seconds: a.seconds,
        pages: a.pages,
    };
    println!("Configuration: {}", serde_json::to_string(&config)?);
    println!(
        "GPU arenas total {} MiB; host page estimate {} MiB/page; admission slots {}",
        arena_mib,
        per_page / 1048576,
        config.inflight
    );
    if a.auto || a.vram_4gb {
        let mut trials = Vec::new();
        let mut reference = None;
        let mut eligible = Vec::new();
        let mut eligible_configs = Vec::new();
        let candidates = if a.vram_4gb {
            vec![(1, 32), (1, 64), (1, 128)]
        } else {
            vec![(1, 128), (1, 256), (1, 512), (2, 512), (1, 1024)]
        };
        for (det_batch, reco_batch) in candidates {
            if det_batch > config.inflight {
                continue;
            }
            let mut c = config.clone();
            c.det_batch = det_batch;
            c.reco_batch = reco_batch;
            let folder = a
                .output
                .join(format!("calibrate_d{det_batch}_r{reco_batch}"));
            println!("Calibrating detector {det_batch}, recognition {reco_batch}");
            if !child(&a, &c, &folder, workload.pages.len(), 0.)? {
                let failure = fs::read(folder.join("failure.json"))
                    .ok()
                    .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
                trials.push(json!({"det_batch":det_batch,"reco_batch":reco_batch,"failed":true,"failure":failure}));
                continue;
            }
            let r: Value = serde_json::from_slice(&fs::read(folder.join("summary.json"))?)?;
            let sig = signature(&r);
            let reference = reference.get_or_insert_with(|| sig.clone());
            let parity = reference == &sig;
            let within_budget = r["resources"]["vram_increment_peak_bytes"]
                .as_f64()
                .map(|m| {
                    m <= (if vram_limit_mib > 0 {
                        vram_limit_mib
                    } else {
                        arena_mib
                    }) as f64
                        * 1048576.
                })
                .unwrap_or(false);
            let pps = r["pages_per_second"].as_f64().unwrap();
            let memory = r["resources"]["vram_increment_peak_bytes"].as_u64();
            trials.push(json!({"config":c,"pages_per_second":pps,"same_calibration_text":parity,"within_observed_budget":within_budget,"vram_increment_peak_bytes":memory}));
            if parity && within_budget {
                eligible.push((pps, memory.unwrap()));
                eligible_configs.push(c);
            }
        }
        let index = choose_candidate(&eligible, a.throughput_tolerance).context(
            "No calibration candidate preserved text and stayed within the observed memory budget",
        )?;
        let best = &eligible_configs[index];
        fs::write(
            a.output.join("autotune.json"),
            serde_json::to_vec_pretty(
                &json!({"selected":best,"trials":trials,"throughput_tolerance":a.throughput_tolerance,"profile":if a.vram_4gb {"4gb"} else {"default"},"policy":"Least observed memory within throughput tolerance of fastest eligible trial","scope":"Startup calibration on supplied pages, not continuous adaptation. Requires identical calibration text to smallest completed candidate. Device-wide memory sampling is approximate; arena caps do not include all allocations."}),
            )?,
        )?;
        ensure!(
            child(&a, best, &a.output.join("run"), a.pages, a.seconds)?,
            "Selected run failed"
        );
        return Ok(());
    }
    let meta: Metadata = serde_json::from_slice(&fs::read(a.models.join("metadata.json"))?)?;
    let running = Arc::new(AtomicBool::new(true));
    let cancel = Arc::new(AtomicBool::new(false));
    let exceeded = Arc::new(AtomicBool::new(false));
    let telemetry_failed = Arc::new(AtomicBool::new(false));
    let sampler_cancel = cancel.clone();
    let sampler_exceeded = exceeded.clone();
    let sampler_failed = telemetry_failed.clone();
    let limit_bytes = config.vram_limit_mib as u64 * 1048576;
    let flag = running.clone();
    let output = a.output.clone();
    let sampler = thread::spawn(move || -> Result<Vec<Value>> {
        let nv = Nvml::new().ok();
        let mut rows = Vec::new();
        let mut file = fs::File::create(output.join("telemetry.jsonl"))?;
        let mut last_cpu = process_stats();
        let mut last_time = Instant::now();
        while flag.load(Ordering::Relaxed) {
            let mem = nv.as_ref().and_then(|n| n.memory().ok());
            if limit_bytes > 0 && mem.is_none() {
                sampler_failed.store(true, Ordering::Relaxed);
                sampler_cancel.store(true, Ordering::Relaxed);
            }
            if limit_bytes > 0
                && mem.zip(memory).is_some_and(|(current, initial)| {
                    current.used.saturating_sub(initial.used) > limit_bytes
                })
            {
                sampler_exceeded.store(true, Ordering::Relaxed);
                sampler_cancel.store(true, Ordering::Relaxed);
            }
            let util = nv.as_ref().and_then(|n| n.utilization().ok());
            let process = process_stats();
            let cpu = process.zip(last_cpu).map(|(current, previous)| {
                (current.1 - previous.1) / last_time.elapsed().as_secs_f64() * 100.
            });
            last_time = Instant::now();
            last_cpu = process;
            let row = json!({"time_ns":now(),"gpu_util_pct":util.map(|u|u.gpu),"device_vram_bytes":mem.map(|m|m.used),"rss_bytes":process.map(|p|p.0),"process_cpu_pct":cpu});
            serde_json::to_writer(&mut file, &row)?;
            file.write_all(b"\n")?;
            rows.push(row);
            thread::sleep(Duration::from_millis(100));
        }
        Ok(rows)
    });
    let work = (|| -> Result<Value> {
        let loading = Instant::now();
        let (det, reco) = pipeline::sessions(&a.models, &config)?;
        let model_load_seconds = loading.elapsed().as_secs_f64();
        let mut warm = config.clone();
        warm.pages = workload.pages.len();
        warm.seconds = 0.;
        let (warmup, det, reco) = pipeline::run(
            &warm,
            &workload.pages,
            &meta,
            det,
            reco,
            None,
            Some(&cancel),
        )?;
        println!(
            "Warmup complete ({:.1}s); starting measurement",
            warmup.wall_seconds
        );
        let start_ns = now();
        let (stats, _det, _reco) = pipeline::run(
            &config,
            &workload.pages,
            &meta,
            det,
            reco,
            Some(a.output.join("pages.jsonl")),
            Some(&cancel),
        )?;
        let end_ns = now();
        let mut r = serde_json::to_value(stats)?;
        r["implementation"] = json!("rust_pipeline");
        r["config"] = serde_json::to_value(&config)?;
        r["start_ns"] = json!(start_ns);
        r["end_ns"] = json!(end_ns);
        r["model_load_seconds"] = json!(model_load_seconds);
        r["warmup_seconds"] = json!(warmup.wall_seconds);
        r["model"] = json!("db_resnet34 + parseq");
        r["scope"] = json!(
            "Warm-cache decode, bounded interleaved upright OCR with cross-page crop batches, ordered JSONL flush/fsync. Load/warmup excluded. No orientation or document layout."
        );
        Ok(r)
    })();
    running.store(false, Ordering::Relaxed);
    let rows = sampler
        .join()
        .map_err(|_| anyhow::anyhow!("Sampler panic"))??;
    let work = if telemetry_failed.load(Ordering::Relaxed) {
        Err(anyhow::anyhow!(
            "VRAM telemetry unavailable; stopped rather than run without the requested memory guard"
        ))
    } else if exceeded.load(Ordering::Relaxed) {
        Err(anyhow::anyhow!(
            "Sampled incremental VRAM exceeded {} MiB; stopped the pipeline. Both models must fit together; no CPU/offload fallback was used.",
            config.vram_limit_mib
        ))
    } else {
        work
    };
    let mut r = match work {
        Ok(r) => r,
        Err(e) => {
            fs::write(
                a.output.join("failure.json"),
                serde_json::to_vec_pretty(
                    &json!({"error":format!("{e:#}"),"config":config,"vram_budget_exceeded":exceeded.load(Ordering::Relaxed),"initial_device_vram_bytes":memory.map(|m|m.used),"peak_device_vram_bytes":rows.iter().filter_map(|x|x["device_vram_bytes"].as_u64()).max()}),
                )?,
            )?;
            return Err(e);
        }
    };
    let start = r["start_ns"].as_u64().unwrap();
    let end = r["end_ns"].as_u64().unwrap();
    let timed: Vec<_> = rows
        .iter()
        .filter(|x| {
            x["time_ns"]
                .as_u64()
                .is_some_and(|t| t >= start && t <= end)
        })
        .collect();
    let mut resources = serde_json::Map::new();
    resources.insert("samples".into(), json!(timed.len()));
    for key in [
        "gpu_util_pct",
        "device_vram_bytes",
        "rss_bytes",
        "process_cpu_pct",
    ] {
        let vals: Vec<_> = timed.iter().filter_map(|x| x[key].as_f64()).collect();
        if !vals.is_empty() {
            resources.insert(
                format!("{key}_mean"),
                json!(vals.iter().sum::<f64>() / vals.len() as f64),
            );
            resources.insert(
                format!("{key}_max"),
                json!(vals.iter().copied().fold(0., f64::max)),
            );
        }
    }
    let peak = rows
        .iter()
        .filter_map(|x| x["device_vram_bytes"].as_u64())
        .max();
    resources.insert(
        "vram_increment_peak_bytes".into(),
        json!(peak.zip(memory).map(|(p, m)| p.saturating_sub(m.used))),
    );
    r["resources"] = Value::Object(resources);
    r["initial_device_vram_bytes"] = json!(memory.map(|m| m.used));
    r["both_models_resident"] = json!(true);
    r["runtime"] = json!(ort::info());
    r["workload"] = serde_json::to_value(
        workload
            .pages
            .iter()
            .map(|p| (&p.id, &p.image))
            .collect::<Vec<_>>(),
    )?;
    fs::write(
        a.output.join("summary.json"),
        serde_json::to_vec_pretty(&r)?,
    )?;
    println!(
        "DONE {:.3} pages/s; {} pages; max {} admitted",
        r["pages_per_second"].as_f64().unwrap(),
        r["pages"],
        r["max_inflight"]
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calibration_prefers_memory_saving_near_peak() {
        let trials = [(0.899, 3487), (0.996, 4000), (1.040, 4700), (1.061, 7875)];
        assert_eq!(choose_candidate(&trials, 0.03), Some(2));
        assert_eq!(choose_candidate(&trials, 0.), Some(3));
        assert_eq!(choose_candidate(&[], 0.03), None);
    }
}
