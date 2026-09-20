pub mod crops;
pub mod detection;
pub mod gpu;
pub mod orientation;
pub mod pipeline;
pub mod preprocess;
#[cfg(feature = "python")]
mod python;
pub mod refinement;
use anyhow::{Result, ensure};
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use serde::Deserialize;
use std::path::{Path, PathBuf};
#[derive(Clone, Deserialize)]
pub struct ModelConfig {
    pub mean: [f32; 3],
    pub std: [f32; 3],
    #[serde(default)]
    pub vocab: String,
}
#[derive(Clone, Deserialize)]
pub struct Metadata {
    pub db_resnet34: ModelConfig,
    pub parseq: ModelConfig,
}
pub fn session(path: &Path, cpu: bool, profile: Option<PathBuf>) -> Result<Session> {
    let mut b = Session::builder()?
        .with_intra_threads(1)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?;
    if !cpu {
        b = b.with_execution_providers([CUDAExecutionProvider::default()
            .with_tf32(false)
            .build()
            .error_on_failure()])?;
    }
    if let Some(p) = profile {
        b = b.with_profiling(p)?;
    }
    Ok(b.commit_from_file(path)?)
}
pub fn infer(
    session: &mut Session,
    data: Vec<f32>,
    shape: [usize; 4],
) -> Result<(Vec<i64>, Vec<f32>)> {
    let input = Tensor::from_array((shape, data))?;
    let outputs = session.run(ort::inputs![input])?;
    let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    Ok((shape.to_vec(), data.to_vec()))
}
pub fn recognize(logits: &[f32], shape: &[i64], vocab: &str) -> Result<Vec<(String, f32)>> {
    ensure!(shape.len() == 3, "Unexpected recognizer shape: {shape:?}");
    let chars: Vec<char> = vocab.chars().collect();
    let (n, t, c) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    ensure!(c == chars.len() + 1, "Vocabulary size mismatch");
    let mut results = Vec::new();
    for sample in 0..n {
        let mut word = String::new();
        let mut sum = 0.;
        let mut count = 0;
        for step in 0..t {
            let row = &logits[(sample * t + step) * c..(sample * t + step + 1) * c];
            let (index, &max) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap();
            if index == chars.len() {
                break;
            }
            word.push(chars[index]);
            sum += 1. / row.iter().map(|x| (x - max).exp()).sum::<f32>();
            count += 1;
        }
        results.push((word, if count > 0 { sum / count as f32 } else { 0. }));
    }
    Ok(results)
}
