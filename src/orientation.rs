//! Opt-in CPU page direction and conservative projection-profile deskew.
use crate::{detection::Word, infer, preprocess};
use anyhow::{Result, ensure};
use image::{Rgb, RgbImage, imageops};
use ort::session::Session;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Deserialize)]
struct Model {
    mean: [f32; 3],
    std: [f32; 3],
    classes: Vec<i32>,
    input_shape: [usize; 3],
}
pub struct PageOrientation {
    session: Session,
    model: Model,
}
#[derive(Clone, Debug, Serialize)]
pub struct Geometry {
    pub original_size: [u32; 2],
    pub corrected_size: [u32; 2],
    pub clockwise_class_deg: i32,
    pub class_confidence: f32,
    pub applied_quarter_deg: i32,
    pub skew: Skew,
    /// Corrected pixel-edge coordinates -> original pixel-edge coordinates.
    pub corrected_to_original: [[f64; 3]; 2],
}
#[derive(Clone, Debug, Serialize)]
pub struct Skew {
    pub estimated_deg: f64,
    pub applied_deg: f64,
    pub agreeing_bands: usize,
    pub evidence_bands: usize,
    pub evidence_points: usize,
    pub score_gain: f64,
    pub reason: String,
}

impl PageOrientation {
    pub fn load(models: &Path) -> Result<Self> {
        let model: Model =
            serde_json::from_slice(&std::fs::read(models.join("page_orientation.json"))?)?;
        ensure!(
            model.input_shape == [3, 512, 512] && model.classes == [0, -90, 180, 90],
            "Unexpected page orientation metadata"
        );
        let session = crate::session(&models.join("page_orientation.onnx"), true, None)?;
        Ok(Self { session, model })
    }
    pub fn correct(&mut self, image: RgbImage, deskew: bool) -> Result<(RgbImage, Geometry)> {
        let original_size = [image.width(), image.height()];
        let data = preprocess::prepare(&image, 512, 512, true, &self.model.mean, &self.model.std);
        let (shape, logits) = infer(&mut self.session, data, [1, 3, 512, 512])?;
        ensure!(shape == [1, 4], "Unexpected orientation logits");
        let (index, &peak) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        let confidence = 1. / logits.iter().map(|v| (v - peak).exp()).sum::<f32>();
        let predicted = self.model.classes[index];
        // Low-confidence page direction abstains; never run local skew on an uncertain page.
        let quarter = if confidence >= 0.9 {
            predicted.rem_euclid(360)
        } else {
            0
        };
        let canonical = match quarter {
            90 => imageops::rotate270(&image),
            180 => imageops::rotate180(&image),
            270 => imageops::rotate90(&image),
            _ => image,
        };
        let skew = if deskew && confidence >= 0.9 {
            estimate_skew(&canonical)
        } else {
            Skew {
                estimated_deg: 0.,
                applied_deg: 0.,
                agreeing_bands: 0,
                evidence_bands: 0,
                evidence_points: 0,
                score_gain: 1.,
                reason: if deskew {
                    "uncertain_page_direction"
                } else {
                    "disabled"
                }
                .into(),
            }
        };
        let (fixed, fine_inverse) = rectify(canonical, skew.applied_deg);
        let quarter_inverse = quarter_inverse(original_size, quarter);
        let geometry = Geometry {
            original_size,
            corrected_size: [fixed.width(), fixed.height()],
            clockwise_class_deg: predicted,
            class_confidence: confidence,
            applied_quarter_deg: quarter,
            skew,
            corrected_to_original: compose(quarter_inverse, fine_inverse),
        };
        Ok((fixed, geometry))
    }
}

/// Project horizontal ink boundaries, not the interior of dark title panels.
/// Four independent horizontal bands prevent one decorative line deciding skew.
pub fn estimate_skew(image: &RgbImage) -> Skew {
    let scale = (1000. / image.width().max(image.height()) as f64).min(1.);
    let small = imageops::resize(
        image,
        (image.width() as f64 * scale).round().max(1.) as u32,
        (image.height() as f64 * scale).round().max(1.) as u32,
        imageops::FilterType::Triangle,
    );
    let (w, h) = (small.width() as usize, small.height() as usize);
    let gray: Vec<bool> = small
        .pixels()
        .map(|p| 299 * p[0] as u32 + 587 * p[1] as u32 + 114 * (p[2] as u32) < 160000)
        .collect();
    let mut points: Vec<(f64, f64, usize)> = vec![];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            if gray[y * w + x] != gray[(y + 1) * w + x] {
                points.push((x as f64 - w as f64 / 2., y as f64, (y * 4 / h).min(3)));
            }
        }
    }
    let mut result = Skew {
        estimated_deg: 0.,
        applied_deg: 0.,
        agreeing_bands: 0,
        evidence_bands: 0,
        evidence_points: points.len(),
        score_gain: 1.,
        reason: "insufficient_evidence".into(),
    };
    if points.len() < 500 || h < 16 {
        return result;
    }
    let scores = |angle: f64| -> [f64; 4] {
        let mut rows = vec![[0f64; 4]; h + 128];
        let slope = angle.to_radians().tan();
        for &(x, y, band) in &points {
            let value = y - slope * x + 64.;
            let row = value.floor() as usize;
            let frac = value - row as f64;
            if row + 1 < rows.len() {
                rows[row][band] += 1. - frac;
                rows[row + 1][band] += frac;
            }
        }
        let mut score = [0.; 4];
        for row in rows {
            for b in 0..4 {
                score[b] += row[b] * row[b];
            }
        }
        score
    };
    let mut trials: Vec<(f64, [f64; 4])> = (-30..=30)
        .map(|i| {
            let a = i as f64 / 10.;
            (a, scores(a))
        })
        .collect();
    let best = trials
        .iter()
        .max_by(|a, b| a.1.iter().sum::<f64>().total_cmp(&b.1.iter().sum::<f64>()))
        .unwrap()
        .0;
    trials.extend((-4..=4).map(|i| {
        let a = best + i as f64 * 0.02;
        (a, scores(a))
    }));
    let &(angle, score) = trials
        .iter()
        .max_by(|a, b| a.1.iter().sum::<f64>().total_cmp(&b.1.iter().sum::<f64>()))
        .unwrap();
    result.estimated_deg = angle;
    let zero = scores(0.);
    result.score_gain = score.iter().sum::<f64>() / zero.iter().sum::<f64>().max(1.);
    for b in 0..4 {
        let best_band = trials
            .iter()
            .max_by(|a, c| a.1[b].total_cmp(&c.1[b]))
            .unwrap();
        let count = points.iter().filter(|p| p.2 == b).count();
        if count >= 200 {
            result.evidence_bands += 1;
        }
        if count >= 200 && (best_band.0 - angle).abs() <= 0.2 && score[b] >= zero[b] * 1.03 {
            result.agreeing_bands += 1;
        }
    }
    result.reason = if angle.abs() < 0.2 {
        "near_upright"
    } else if angle.abs() >= 2.9 {
        "search_boundary"
    } else if result.agreeing_bands < 2
        || result.agreeing_bands * 2 <= result.evidence_bands
        || result.score_gain < 1.05
    {
        "weak_consensus"
    } else {
        result.applied_deg = angle;
        "applied"
    }
    .into();
    result
}

fn identity() -> [[f64; 3]; 2] {
    [[1., 0., 0.], [0., 1., 0.]]
}
fn quarter_inverse(size: [u32; 2], q: i32) -> [[f64; 3]; 2] {
    let [w, h] = size.map(f64::from);
    match q {
        90 => [[0., -1., w], [1., 0., 0.]],
        180 => [[-1., 0., w], [0., -1., h]],
        270 => [[0., 1., 0.], [-1., 0., h]],
        _ => identity(),
    }
}
fn compose(a: [[f64; 3]; 2], b: [[f64; 3]; 2]) -> [[f64; 3]; 2] {
    let mut c = [[0.; 3]; 2];
    for r in 0..2 {
        for col in 0..3 {
            c[r][col] = a[r][0] * b[0][col] + a[r][1] * b[1][col];
        }
        c[r][2] += a[r][2];
    }
    c
}
pub fn transform(m: [[f64; 3]; 2], p: [f64; 2]) -> [f64; 2] {
    m.map(|r| r[0] * p[0] + r[1] * p[1] + r[2])
}

/// Quarter turns are lossless pixel permutations. Only fine skew is resampled.
pub fn rectify(image: RgbImage, angle: f64) -> (RgbImage, [[f64; 3]; 2]) {
    if angle.abs() < 1e-9 {
        return (image, identity());
    }
    let (w, h) = (image.width() as f64, image.height() as f64);
    let (sin, cos) = angle.to_radians().sin_cos();
    let (nw, nh) = (
        (cos.abs() * w + sin.abs() * h).ceil() as u32,
        (sin.abs() * w + cos.abs() * h).ceil() as u32,
    );
    let tx = nw as f64 / 2. - cos * w / 2. - sin * h / 2.;
    let ty = nh as f64 / 2. + sin * w / 2. - cos * h / 2.;
    let inverse = [
        [cos, -sin, -cos * tx + sin * ty],
        [sin, cos, -sin * tx - cos * ty],
    ];
    let out = RgbImage::from_fn(nw, nh, |x, y| {
        let [sx, sy] = transform(inverse, [x as f64 + 0.5, y as f64 + 0.5]);
        let (sx, sy) = (sx - 0.5, sy - 0.5);
        let (x0, y0) = (sx.floor() as i64, sy.floor() as i64);
        let (dx, dy) = (sx - x0 as f64, sy - y0 as f64);
        let mut rgb = [0.; 3];
        for (xx, wx) in [(x0, 1. - dx), (x0 + 1, dx)] {
            for (yy, wy) in [(y0, 1. - dy), (y0 + 1, dy)] {
                let pixel = if xx >= 0
                    && yy >= 0
                    && xx < image.width() as i64
                    && yy < image.height() as i64
                {
                    image.get_pixel(xx as u32, yy as u32).0
                } else {
                    [255; 3]
                };
                for c in 0..3 {
                    rgb[c] += pixel[c] as f64 * wx * wy;
                }
            }
        }
        Rgb(rgb.map(|v| v.round().clamp(0., 255.) as u8))
    });
    (out, inverse)
}

impl Geometry {
    pub fn map_words(&self, words: &mut [Word]) {
        let [w, h] = self.corrected_size.map(f64::from);
        let [ow, oh] = self.original_size.map(f64::from);
        for word in words {
            let [[x0, y0], [x1, y1]] = word.polygon;
            let quad = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(|p| {
                let [x, y] = transform(
                    self.corrected_to_original,
                    [p[0] as f64 * w, p[1] as f64 * h],
                );
                [(x / ow) as f32, (y / oh) as f32]
            });
            word.polygon = [
                [
                    quad.iter()
                        .map(|p| p[0])
                        .fold(f32::INFINITY, f32::min)
                        .clamp(0., 1.),
                    quad.iter()
                        .map(|p| p[1])
                        .fold(f32::INFINITY, f32::min)
                        .clamp(0., 1.),
                ],
                [
                    quad.iter()
                        .map(|p| p[0])
                        .fold(f32::NEG_INFINITY, f32::max)
                        .clamp(0., 1.),
                    quad.iter()
                        .map(|p| p[1])
                        .fold(f32::NEG_INFINITY, f32::max)
                        .clamp(0., 1.),
                ],
            ];
            word.quadrilateral = Some(quad);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quarter_turn_inverse_matches_pixels() {
        let original = RgbImage::from_fn(7, 11, |x, y| Rgb([x as u8, y as u8, 0]));
        for q in [0, 90, 180, 270] {
            let corrected = match q {
                90 => imageops::rotate270(&original),
                180 => imageops::rotate180(&original),
                270 => imageops::rotate90(&original),
                _ => original.clone(),
            };
            for (x, y, pixel) in corrected.enumerate_pixels() {
                let [sx, sy] = transform(
                    quarter_inverse([7, 11], q),
                    [x as f64 + 0.5, y as f64 + 0.5],
                );
                assert_eq!(
                    pixel,
                    original.get_pixel(sx.floor() as u32, sy.floor() as u32)
                );
            }
        }
    }
    #[test]
    fn fractional_mapping_preserves_center_and_distances() {
        for angle in [-1.13, 0.37, 1.5] {
            let (fixed, m) = rectify(RgbImage::from_pixel(160, 240, Rgb([255; 3])), angle);
            let center = transform(m, [fixed.width() as f64 / 2., fixed.height() as f64 / 2.]);
            assert!((center[0] - 80.).abs() < 1e-10 && (center[1] - 120.).abs() < 1e-10);
            let p = transform(m, [10., 20.]);
            let q = transform(m, [40., 60.]);
            assert!(((q[0] - p[0]).hypot(q[1] - p[1]) - 50.).abs() < 1e-10);
        }
    }
    #[test]
    fn blank_and_single_rule_abstain() {
        let mut page = RgbImage::from_pixel(600, 800, Rgb([255; 3]));
        assert_eq!(estimate_skew(&page).applied_deg, 0.);
        for x in 30..570 {
            for y in 80..84 {
                page.put_pixel(x, y, Rgb([0; 3]));
            }
        }
        let (tilted, _) = rectify(page, -1.13);
        assert_eq!(estimate_skew(&tilted).applied_deg, 0.);
    }
    #[test]
    fn non_grid_skew_requires_distributed_evidence() {
        let mut page = RgbImage::from_pixel(600, 800, Rgb([255; 3]));
        for row in 0..22 {
            for col in 0..16 {
                for y in 30 + row * 32..39 + row * 32 {
                    for x in 20 + col * 34..44 + col * 34 {
                        page.put_pixel(x, y, Rgb([0; 3]));
                    }
                }
            }
        }
        let (tilted, _) = rectify(page, -0.37);
        let skew = estimate_skew(&tilted);
        assert!((skew.estimated_deg - 0.37).abs() < 0.12, "{skew:?}");
        assert!(
            skew.agreeing_bands >= 2 && skew.applied_deg != 0.,
            "{skew:?}"
        );
    }
    #[test]
    fn opposing_directions_abstain() {
        let mut page = RgbImage::from_pixel(600, 800, Rgb([255; 3]));
        for row in 0..20 {
            let slope = if row < 10 { 1.5_f64 } else { -1.5_f64 }.to_radians().tan();
            for x in 30..570 {
                let y = (30. + row as f64 * 38. + (x as f64 - 300.) * slope).round() as u32;
                for yy in y..y + 3 {
                    page.put_pixel(x, yy, Rgb([0; 3]));
                }
            }
        }
        let skew = estimate_skew(&page);
        assert_eq!(skew.applied_deg, 0., "{skew:?}");
    }
}
