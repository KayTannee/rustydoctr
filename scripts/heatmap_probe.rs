//! Experimental whole-page probability fusion. Not part of the production streaming API.
use anyhow::{Result, ensure};
use clap::Parser;
use image::{GrayImage, Luma, Rgb, RgbImage};
use ort::{
    execution_providers::{CUDAExecutionProvider, cuda::CuDNNConvAlgorithmSearch},
    session::Session,
};
use rustydoctr::{Metadata, crops, detection, infer, preprocess, recognize};
use serde::Deserialize;
use std::{
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    jobs: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "models")]
    models: PathBuf,
}
#[derive(Deserialize)]
struct Page {
    id: String,
    image: PathBuf,
    tiles: Vec<[i32; 4]>,
}
fn session(path: &Path, mib: usize) -> Result<Session> {
    Ok(Session::builder()?
        .with_intra_threads(1)?
        .with_execution_providers([CUDAExecutionProvider::default()
            .with_tf32(false)
            .with_memory_limit(mib * 1024 * 1024)
            .with_conv_algorithm_search(CuDNNConvAlgorithmSearch::Heuristic)
            .build()
            .error_on_failure()])?
        .commit_from_file(path)?)
}
fn sample(p: &[f32], side: usize, x: f64, y: f64) -> f32 {
    let x = x.clamp(0., (side - 1) as f64);
    let y = y.clamp(0., (side - 1) as f64);
    let a = x.floor() as usize;
    let b = y.floor() as usize;
    let c = (a + 1).min(side - 1);
    let d = (b + 1).min(side - 1);
    let fx = (x - a as f64) as f32;
    let fy = (y - b as f64) as f32;
    (p[b * side + a] * (1. - fx) + p[b * side + c] * fx) * (1. - fy)
        + (p[d * side + a] * (1. - fx) + p[d * side + c] * fx) * fy
}
struct Fusion {
    w: usize,
    h: usize,
    scale: f64,
    sums: [Vec<f32>; 2],
    weights: [Vec<f32>; 2],
}
impl Fusion {
    fn new(w: u32, h: u32, side: usize) -> Self {
        let scale = 1024. / side as f64;
        let w = (w as f64 * scale).ceil() as usize;
        let h = (h as f64 * scale).ceil() as usize;
        Self {
            w,
            h,
            scale,
            sums: [vec![0.; w * h], vec![0.; w * h]],
            weights: [vec![0.; w * h], vec![0.; w * h]],
        }
    }
    fn add(&mut self, p: &[f32], tile: [i32; 4], iw: u32, ih: u32) {
        let [x, y, x1, y1] = tile;
        let lo_x = (x as f64 * self.scale).floor().max(0.) as usize;
        let lo_y = (y as f64 * self.scale).floor().max(0.) as usize;
        let hi_x = ((x1 as f64 * self.scale).ceil() as usize).min(self.w);
        let hi_y = ((y1 as f64 * self.scale).ceil() as usize).min(self.h);
        for gy in lo_y..hi_y {
            for gx in lo_x..hi_x {
                let sx = (gx as f64 + 0.5) / self.scale;
                let sy = (gy as f64 + 0.5) / self.scale;
                if sx < x as f64 || sx >= x1 as f64 || sy < y as f64 || sy >= y1 as f64 {
                    continue;
                }
                let tx = (sx - x as f64) * self.scale;
                let ty = (sy - y as f64) * self.scale;
                let probability = sample(p, 1024, tx - 0.5, ty - 0.5);
                let mut edge = 128f64;
                if x > 0 {
                    edge = edge.min(tx);
                }
                if y > 0 {
                    edge = edge.min(ty);
                }
                if x1 < (iw as i32) {
                    edge = edge.min(1024. - tx);
                }
                if y1 < (ih as i32) {
                    edge = edge.min(1024. - ty);
                }
                let weight = (edge / 128.).clamp(0.001, 1.) as f32;
                let i = gy * self.w + gx;
                for (mode, weight) in [1., weight].into_iter().enumerate() {
                    self.sums[mode][i] += probability * weight;
                    self.weights[mode][i] += weight;
                }
            }
        }
    }
    fn finish(&self, mode: usize) -> Result<Vec<f32>> {
        ensure!(
            self.weights[mode].iter().all(|&w| w > 0.),
            "Uncovered heatmap pixel"
        );
        Ok(self.sums[mode]
            .iter()
            .zip(&self.weights[mode])
            .map(|(s, w)| s / w)
            .collect())
    }
}
fn recognize_words(
    image: &RgbImage,
    words: &mut [detection::Word],
    reco: &mut Session,
    meta: &Metadata,
) -> Result<usize> {
    let (images, maps) = crops::extract(image, words)?;
    let mut parts = vec![];
    for batch in images.chunks(256) {
        let data = batch
            .iter()
            .flat_map(|im| {
                preprocess::prepare(im, 32, 128, false, &meta.parseq.mean, &meta.parseq.std)
            })
            .collect();
        let (shape, logits) = infer(reco, data, [batch.len(), 3, 32, 128])?;
        parts.extend(recognize(&logits, &shape, &meta.parseq.vocab)?);
    }
    crops::remap(words, &parts, &maps);
    Ok(images.len())
}
fn main() -> Result<()> {
    let a = Args::parse();
    let jobs: Vec<Page> = serde_json::from_slice(&fs::read(&a.jobs)?)?;
    ensure!(!jobs.is_empty(), "Empty jobs");
    fs::create_dir_all(&a.output)?;
    let start = Instant::now();
    let meta: Metadata = serde_json::from_slice(&fs::read(a.models.join("metadata.json"))?)?;
    let mut det = session(&a.models.join("db_resnet34.onnx"), 4096)?;
    let mut reco = session(&a.models.join("parseq.onnx"), 2048)?;
    let names = ["uniform", "weighted"];
    let mut files = vec![];
    for name in names {
        let folder = a.output.join(name);
        fs::create_dir_all(&folder)?;
        files.push(BufWriter::new(fs::File::create(
            folder.join("pages.jsonl"),
        )?));
    }
    let mut passes = 0;
    let mut crops_count = [0usize; 2];
    for (sequence, page) in jobs.iter().enumerate() {
        let image = image::open(&page.image)?.to_rgb8();
        let (w, h) = image.dimensions();
        ensure!(!page.tiles.is_empty(), "Empty tile list");
        let side = (page.tiles[0][2] - page.tiles[0][0]) as usize;
        let mut fusion = Fusion::new(w, h, side);
        for (index, &tile) in page.tiles.iter().enumerate() {
            let [x, y, x1, y1] = tile;
            ensure!(
                x1 - x == side as i32 && y1 - y == side as i32,
                "Inconsistent tile size"
            );
            let mut crop = RgbImage::from_pixel(side as u32, side as u32, Rgb([255, 255, 255]));
            for yy in y.max(0)..y1.min(h as i32) {
                for xx in x.max(0)..x1.min(w as i32) {
                    crop.put_pixel(
                        (xx - x) as u32,
                        (yy - y) as u32,
                        *image.get_pixel(xx as u32, yy as u32),
                    );
                }
            }
            let data = preprocess::prepare(
                &crop,
                1024,
                1024,
                true,
                &meta.db_resnet34.mean,
                &meta.db_resnet34.std,
            );
            let (shape, logits) = infer(&mut det, data, [1, 3, 1024, 1024])?;
            ensure!(shape == [1, 1, 1024, 1024], "Unexpected map shape");
            let p: Vec<_> = logits.into_iter().map(|x| 1. / (1. + (-x).exp())).collect();
            if sequence == 0 {
                fs::write(
                    a.output.join(format!("control_tile_{index}.json")),
                    serde_json::to_vec(&detection::boxes(&p, 1024, 1024, side, side))?,
                )?;
            }
            fusion.add(&p, tile, w, h);
            passes += 1;
        }
        for (mode, file) in files.iter_mut().enumerate() {
            let p = fusion.finish(mode)?;
            // The fused map is rectangular and has NO letterbox padding. Equal image
            // dimensions explicitly bypass DB's square-input unpadding adjustment.
            let mut words = detection::boxes(&p, fusion.h, fusion.w, 1, 1);
            for word in &mut words {
                for point in &mut word.polygon {
                    point[0] = (point[0] as f64 * fusion.w as f64 / (fusion.scale * w as f64))
                        .clamp(0., 1.) as f32;
                    point[1] = (point[1] as f64 * fusion.h as f64 / (fusion.scale * h as f64))
                        .clamp(0., 1.) as f32;
                }
            }
            words.retain(|word| {
                word.polygon[1][0] > word.polygon[0][0] && word.polygon[1][1] > word.polygon[0][1]
            });
            crops_count[mode] += recognize_words(&image, &mut words, &mut reco, &meta)?;
            serde_json::to_writer(
                &mut *file,
                &serde_json::json!({"id":page.id,"sequence":sequence,"words":words,"tiles":page.tiles,"fusion":names[mode]}),
            )?;
            writeln!(file)?;
            file.flush()?;
            if sequence < 2 {
                let mut heat = GrayImage::new(fusion.w as u32, fusion.h as u32);
                for (i, pixel) in heat.pixels_mut().enumerate() {
                    *pixel = Luma([(p[i] * 255.).round() as u8]);
                }
                heat.save(
                    a.output
                        .join(names[mode])
                        .join(format!("heatmap_{sequence}.png")),
                )?;
            }
        }
        if (sequence + 1) % 25 == 0 {
            println!("fused {} / {}", sequence + 1, jobs.len());
        }
    }
    fs::write(
        a.output.join("summary.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"pages":jobs.len(),"detector_passes":passes,"recognizer_crops":crops_count,"wall_seconds":start.elapsed().as_secs_f64(),"note":"Both fusion policies share detector inference; cold serial accuracy probe, not production throughput."}),
        )?,
    )?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constant_fields_survive_overlap() {
        let mut f = Fusion::new(1500, 1000, 1000);
        let p = vec![0.7; 1024 * 1024];
        f.add(&p, [0, 0, 1000, 1000], 1500, 1000);
        f.add(&p, [500, 0, 1500, 1000], 1500, 1000);
        for mode in 0..2 {
            assert!(
                f.finish(mode)
                    .unwrap()
                    .iter()
                    .all(|&v| (v - 0.7).abs() < 1e-6)
            );
        }
    }
    #[test]
    fn internal_edge_gets_less_weight() {
        let mut f = Fusion::new(1500, 1000, 1000);
        f.add(&vec![1.; 1024 * 1024], [0, 0, 1000, 1000], 1500, 1000);
        f.add(&vec![0.; 1024 * 1024], [500, 0, 1500, 1000], 1500, 1000);
        let i = 500 * f.w + 515;
        let avg = f.finish(0).unwrap();
        let weighted = f.finish(1).unwrap();
        assert!((avg[i] - 0.5).abs() < 1e-6);
        assert!(weighted[i] > 0.9);
    }
    #[test]
    fn shifted_grid_has_coverage() {
        let mut f = Fusion::new(1300, 2000, 1000);
        let p = vec![0.4; 1024 * 1024];
        for y in [-438, 437, 1312] {
            for x in [-438, 437, 1312] {
                f.add(&p, [x, y, x + 1000, y + 1000], 1300, 2000);
            }
        }
        assert!(f.finish(1).unwrap().iter().all(|v| (*v - 0.4).abs() < 1e-6));
    }
    #[test]
    fn bilinear_sampling() {
        assert!((sample(&[0., 1., 1., 0.], 2, 0.5, 0.5) - 0.5).abs() < 1e-6);
    }
}
#[cfg(test)]
mod geometry_tests {
    use super::*;
    #[test]
    fn word_crossing_tile_boundary_is_one_box_in_rectangular_map() {
        let mut f = Fusion::new(1536, 1024, 1024);
        for origin in [0usize, 512] {
            let mut p = vec![0.; 1024 * 1024];
            for y in 400..430 {
                for x in 800..1100 {
                    if x >= origin && x < origin + 1024 {
                        p[y * 1024 + x - origin] = 0.9;
                    }
                }
            }
            f.add(
                &p,
                [origin as i32, 0, origin as i32 + 1024, 1024],
                1536,
                1024,
            );
        }
        for mode in 0..2 {
            let p = f.finish(mode).unwrap();
            let words = detection::boxes(&p, f.h, f.w, 1, 1);
            assert_eq!(words.len(), 1);
            let b = words[0].polygon;
            assert!(b[0][0] < 800. / 1536. && b[1][0] > 1100. / 1536.);
            let cx = (b[0][0] + b[1][0]) * 0.5;
            assert!((cx - 950. / 1536.).abs() < 0.002);
        }
    }
}
