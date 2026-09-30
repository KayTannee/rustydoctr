//! Cache native detector maps, then sweep tile postprocessing without GPU inference.
use anyhow::{Result, ensure};
use clap::Parser;
use ort::{
    execution_providers::{CUDAExecutionProvider, cuda::CuDNNConvAlgorithmSearch},
    session::Session,
};
use rustydoctr::{
    Metadata,
    detection::{self, Params, Word},
    orientation,
    pipeline::Workload,
    preprocess,
    refinement::{self, Tile},
};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    /// Include pre-ownership tile boxes for CPU-only seam diagnostics.
    #[arg(long)]
    dump_tiles: bool,
    #[arg(long)]
    thin_recovery: bool,
    #[arg(long)]
    workload: Option<PathBuf>,
    #[arg(long)]
    cache: PathBuf,
    #[arg(long)]
    grid: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value = "models")]
    models: PathBuf,
}
#[derive(Serialize, Deserialize)]
struct Cached {
    id: String,
    image: String,
    width: u32,
    height: u32,
    geometry: serde_json::Value,
    base: Vec<Word>,
    #[serde(default)]
    base_map: Option<String>,
    tiles: Vec<(Tile, String)>,
}
fn main() -> Result<()> {
    let a = Args::parse();
    if let Some(grid) = a.grid {
        let rows: Vec<Cached> = serde_json::from_slice(&fs::read(a.cache.join("cache.json"))?)?;
        let params: Vec<Params> = serde_json::from_slice(&fs::read(grid)?)?;
        ensure!(params.iter().all(|p| p.valid()), "Invalid sweep parameters");
        let mut results = vec![];
        for row in rows {
            let maps: Vec<_> = row
                .tiles
                .iter()
                .map(|(tile, file)| -> Result<_> {
                    let raw = fs::read(a.cache.join(file))?;
                    ensure!(
                        raw.len() == 1024 * 1024 * 4,
                        "Invalid probability map length"
                    );
                    Ok((
                        tile,
                        raw.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|b| f32::from_le_bytes(*b))
                            .collect::<Vec<_>>(),
                    ))
                })
                .collect::<Result<_>>()?;
            for (index, &param) in params.iter().enumerate() {
                let mut words = row.base.clone();
                let mut recovered = Vec::new();
                let mut unmerged_tiles = Vec::new();
                let mut tile_words = Vec::new();
                for (tile, prob) in &maps {
                    let (extra, candidates) = detection::boxes_and_thin(
                        prob,
                        1024,
                        1024,
                        tile.height as usize,
                        tile.width as usize,
                        param,
                        a.thin_recovery,
                    );
                    if a.dump_tiles {
                        unmerged_tiles.push(serde_json::json!({"tile":tile,"words":extra}));
                    }
                    tile_words.push(((*tile).clone(), extra));
                    let mut owned = Vec::new();
                    refinement::merge(&mut owned, candidates, tile, row.width, row.height);
                    for word in &mut owned {
                        word.thin_recovery.as_mut().unwrap().source = "dense_tile".into();
                    }
                    recovered.extend(owned);
                }
                refinement::merge_tiles(&mut words, tile_words, row.width, row.height);
                if a.thin_recovery
                    && let Some(file) = &row.base_map
                {
                    let raw = fs::read(a.cache.join(file))?;
                    ensure!(raw.len() == 1536 * 1536 * 4, "Invalid full-page map length");
                    let prob: Vec<_> = raw
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect();
                    let (_, mut candidates) = detection::boxes_and_thin(
                        &prob,
                        1536,
                        1536,
                        row.height as usize,
                        row.width as usize,
                        Params::default(),
                        true,
                    );
                    for word in &mut candidates {
                        word.thin_recovery.as_mut().unwrap().source = "full_page".into();
                    }
                    recovered.extend(candidates);
                }
                rustydoctr::thin_recovery::append(&mut words, recovered, row.width, row.height);
                results.push(
                    serde_json::json!({"page":row.id,"setting":index,"params":param,"words":words,"unmerged_tiles":unmerged_tiles}),
                );
            }
            println!("CPU postprocess {}", row.id);
        }
        fs::write(
            a.output.expect("--output required for postprocessing"),
            serde_json::to_vec(&results)?,
        )?;
        return Ok(());
    }
    ensure!(!a.cache.exists(), "Choose a fresh cache directory");
    let workload: Workload = serde_json::from_slice(&fs::read(
        a.workload.expect("--workload required for caching"),
    )?)?;
    fs::create_dir_all(&a.cache)?;
    let meta: Metadata = serde_json::from_slice(&fs::read(a.models.join("metadata.json"))?)?;
    let mut det = Session::builder()?
        .with_intra_threads(1)?
        .with_execution_providers([CUDAExecutionProvider::default()
            .with_tf32(false)
            .with_memory_limit(4 * 1024 * 1024 * 1024)
            .with_conv_algorithm_search(CuDNNConvAlgorithmSearch::Heuristic)
            .build()
            .error_on_failure()])?
        .commit_from_file(a.models.join("db_resnet34.onnx"))?;
    let mut orientation = orientation::PageOrientation::load(&a.models)?;
    let mut rows = vec![];
    for (i, page) in workload.pages.iter().enumerate() {
        let (image, geometry) = orientation.correct(image::open(&page.image)?.to_rgb8(), true)?;
        let image_name = format!("page_{i}.png");
        image.save(a.cache.join(&image_name))?;
        let tensor = preprocess::prepare(
            &image,
            1536,
            1536,
            true,
            &meta.db_resnet34.mean,
            &meta.db_resnet34.std,
        );
        let (_, logits) = rustydoctr::infer(&mut det, tensor, [1, 3, 1536, 1536])?;
        ensure!(
            logits.len() == 1536 * 1536,
            "Unexpected full-page detector output"
        );
        let prob: Vec<_> = logits.into_iter().map(|x| 1. / (1. + (-x).exp())).collect();
        let base_map = format!("page_{i}_base.f32");
        fs::write(
            a.cache.join(&base_map),
            prob.iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let base = detection::boxes(
            &prob,
            1536,
            1536,
            image.height() as usize,
            image.width() as usize,
        );
        let mut tiles = vec![];
        for (j, tile) in refinement::select(&image, 1536).into_iter().enumerate() {
            let crop = image::imageops::crop_imm(&image, tile.x, tile.y, tile.width, tile.height)
                .to_image();
            let tensor = preprocess::prepare(
                &crop,
                1024,
                1024,
                true,
                &meta.db_resnet34.mean,
                &meta.db_resnet34.std,
            );
            let (_, logits) = rustydoctr::infer(&mut det, tensor, [1, 3, 1024, 1024])?;
            ensure!(
                logits.len() == 1024 * 1024,
                "Unexpected tile detector output"
            );
            let raw: Vec<u8> = logits
                .into_iter()
                .flat_map(|x| (1_f32 / (1. + (-x).exp())).to_le_bytes())
                .collect();
            let name = format!("page_{i}_tile_{j}.f32");
            fs::write(a.cache.join(&name), raw)?;
            tiles.push((tile, name));
        }
        println!("Cached {}: {} tiles", page.id, tiles.len());
        rows.push(Cached {
            id: page.id.clone(),
            image: image_name,
            width: image.width(),
            height: image.height(),
            geometry: serde_json::to_value(geometry)?,
            base,
            base_map: Some(base_map),
            tiles,
        });
    }
    fs::write(
        a.cache.join("cache.json"),
        serde_json::to_vec_pretty(&rows)?,
    )?;
    Ok(())
}
