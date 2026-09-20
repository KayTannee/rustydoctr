//! CPU-only diagnostic for the page correction stage.
use anyhow::Result;
use clap::Parser;
use rustydoctr::{orientation::PageOrientation, pipeline::Workload};
use std::{fs, path::PathBuf};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    workload: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "models")]
    models: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let workload: Workload = serde_json::from_slice(&fs::read(a.workload)?)?;
    let mut model = PageOrientation::load(&a.models)?;
    let mut rows = vec![];
    for p in workload.pages {
        let start = std::time::Instant::now();
        let (_, geometry) = model.correct(image::open(p.image)?.to_rgb8(), true)?;
        println!(
            "{} quarter {} skew {:.3} applied {:.3} {}",
            p.id,
            geometry.applied_quarter_deg,
            geometry.skew.estimated_deg,
            geometry.skew.applied_deg,
            geometry.skew.reason
        );
        rows.push(serde_json::json!({"id":p.id,"geometry":geometry,"seconds":start.elapsed().as_secs_f64()}));
    }
    fs::write(a.output, serde_json::to_vec_pretty(&rows)?)?;
    Ok(())
}
