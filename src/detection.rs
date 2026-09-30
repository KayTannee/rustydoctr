//! Straight-page DB postprocessing, adapted from docTR (Apache-2.0).
//! Uses component bounds instead of contour points: the straight DB path reduces
//! each external contour to its axis-aligned bounding rectangle before expansion.
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Params {
    pub bin_thresh: f32,
    pub box_thresh: f32,
    pub unclip_ratio: f64,
}
impl Default for Params {
    fn default() -> Self {
        Self {
            bin_thresh: 0.3,
            box_thresh: 0.1,
            unclip_ratio: 1.5,
        }
    }
}
impl Params {
    pub fn valid(self) -> bool {
        self.bin_thresh > 0.
            && self.bin_thresh < 1.
            && (0.0..=1.0).contains(&self.box_thresh)
            && self.unclip_ratio.is_finite()
            && (0.0..=5.0).contains(&self.unclip_ratio)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Word {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thin_recovery: Option<crate::thin_recovery::Evidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crop_decision: Option<crate::line_orientation::Decision>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quadrilateral: Option<[[f32; 2]; 4]>,
    pub polygon: [[f32; 2]; 2],
    pub objectness: f32,
    pub text: String,
    pub confidence: f32,
}

pub fn boxes(prob: &[f32], h: usize, w: usize, ih: usize, iw: usize) -> Vec<Word> {
    boxes_with_params(prob, h, w, ih, iw, Params::default())
}
pub fn boxes_with_params(
    prob: &[f32],
    h: usize,
    w: usize,
    ih: usize,
    iw: usize,
    params: Params,
) -> Vec<Word> {
    boxes_and_thin(prob, h, w, ih, iw, params, false).0
}

#[allow(clippy::too_many_arguments)]
pub fn boxes_and_thin(
    prob: &[f32],
    h: usize,
    w: usize,
    ih: usize,
    iw: usize,
    params: Params,
    recover: bool,
) -> (Vec<Word>, Vec<Word>) {
    let n = h * w;
    let mask: Vec<bool> = prob.iter().map(|&p| p >= params.bin_thresh).collect();
    let mut eroded = vec![true; n];
    let mut opened = vec![false; n];
    for y in 0..h {
        for x in 0..w {
            eroded[y * w + x] = (y.saturating_sub(1)..=(y + 1).min(h - 1))
                .all(|yy| (x.saturating_sub(1)..=(x + 1).min(w - 1)).all(|xx| mask[yy * w + xx]));
        }
    }
    for y in 0..h {
        for x in 0..w {
            opened[y * w + x] = (y.saturating_sub(1)..=(y + 1).min(h - 1))
                .any(|yy| (x.saturating_sub(1)..=(x + 1).min(w - 1)).any(|xx| eroded[yy * w + xx]));
        }
    }
    // Four-connected exterior background distinguishes RETR_EXTERNAL from holes.
    let mut exterior = vec![false; n];
    let mut queue = VecDeque::new();
    for y in 0..h {
        for x in 0..w {
            if (y == 0 || x == 0 || y == h - 1 || x == w - 1) && !opened[y * w + x] {
                exterior[y * w + x] = true;
                queue.push_back(y * w + x);
            }
        }
    }
    while let Some(i) = queue.pop_front() {
        let (y, x) = (i / w, i % w);
        for j in [
            if x > 0 { Some(i - 1) } else { None },
            if x + 1 < w { Some(i + 1) } else { None },
            if y > 0 { Some(i - w) } else { None },
            if y + 1 < h { Some(i + w) } else { None },
        ]
        .into_iter()
        .flatten()
        {
            if !opened[j] && !exterior[j] {
                exterior[j] = true;
                queue.push_back(j);
            }
        }
    }
    let mut seen = vec![false; n];
    let mut result = Vec::new();
    let mut anchors = Vec::new();
    for start in 0..n {
        if !opened[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        queue.push_back(start);
        let (mut x0, mut x1, mut y0, mut y1) = (start % w, start % w, start / w, start / w);
        let mut external = false;
        while let Some(i) = queue.pop_front() {
            let (y, x) = (i / w, i % w);
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
            external |= x == 0 || y == 0 || x + 1 == w || y + 1 == h;
            for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let j = yy * w + xx;
                    external |= exterior[j];
                    if opened[j] && !seen[j] {
                        seen[j] = true;
                        queue.push_back(j);
                    }
                }
            }
        }
        if recover && x1 - x0 + 1 >= 3 && y1 - y0 + 1 >= 4 && x1 - x0 >= y1 - y0 {
            anchors.push([
                x0 as f64,
                y0 as f64,
                (x1 - x0 + 1) as f64,
                (y1 - y0 + 1) as f64,
            ]);
        }
        if !external || x1 - x0 < 2 || y1 - y0 < 2 {
            continue;
        }
        let (bw, bh) = ((x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32);
        let (sx1, sy1) = ((x1 + 1).min(w - 1), (y1 + 1).min(h - 1));
        let mut score = 0.;
        for y in y0..=sy1 {
            for x in x0..=sx1 {
                score += prob[y * w + x];
            }
        }
        score /= ((sx1 - x0 + 1) * (sy1 - y0 + 1)) as f32;
        if score < params.box_thresh {
            continue;
        }
        // Clipper rounds expanded vertex coordinates, not the distance itself.
        // At half-pixel offsets those differ on the minimum edges.
        let distance = bw as f64 * bh as f64 * params.unclip_ratio / (2. * (bw as f64 + bh as f64));
        let mut b = [
            ((x0 as f64 - distance).round() as f32 / w as f32).clamp(0., 1.),
            ((y0 as f64 - distance).round() as f32 / h as f32).clamp(0., 1.),
            (((x0 as f64 + bw as f64 + distance).round() + 1.) as f32 / w as f32).clamp(0., 1.),
            (((y0 as f64 + bh as f64 + distance).round() + 1.) as f32 / h as f32).clamp(0., 1.),
        ];
        if ih > iw {
            for i in [0, 2] {
                b[i] = ((b[i] - 0.5) * ih as f32 / iw as f32 + 0.5).clamp(0., 1.);
            }
        } else if iw > ih {
            for i in [1, 3] {
                b[i] = ((b[i] - 0.5) * iw as f32 / ih as f32 + 0.5).clamp(0., 1.);
            }
        }
        if b[2] > b[0] && b[3] > b[1] {
            result.push(Word {
                thin_recovery: None,
                crop_decision: None,
                quadrilateral: None,
                polygon: [[b[0], b[1]], [b[2], b[3]]],
                objectness: score,
                text: String::new(),
                confidence: 0.,
            });
        }
    }
    // OpenCV RETR_EXTERNAL returns contours in reverse raster discovery order.
    result.reverse();
    let recovered = if recover {
        crate::thin_recovery::propose(
            prob,
            &mask,
            &opened,
            &anchors,
            crate::thin_recovery::Geometry {
                height: h,
                width: w,
                image_height: ih,
                image_width: iw,
            },
        )
    } else {
        vec![]
    };
    (result, recovered)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tile_threshold_can_separate_a_weak_bridge() {
        let mut p = vec![0.; 40 * 20];
        for y in 5..13 {
            for x in (5..13).chain(18..26) {
                p[y * 40 + x] = 0.9;
            }
        }
        for y in 8..11 {
            for x in 13..18 {
                p[y * 40 + x] = 0.35;
            }
        }
        assert_eq!(boxes(&p, 20, 40, 20, 40).len(), 1);
        let params = Params {
            bin_thresh: 0.5,
            ..Params::default()
        };
        assert_eq!(boxes_with_params(&p, 20, 40, 20, 40, params).len(), 2);
        assert!(
            boxes_with_params(
                &p,
                20,
                40,
                20,
                40,
                Params {
                    box_thresh: 0.99,
                    ..params
                }
            )
            .is_empty()
        );
    }
    #[test]
    fn expansion_changes_bounds_without_splitting_components() {
        let mut p = vec![0.; 400];
        for y in 6..10 {
            for x in 5..13 {
                p[y * 20 + x] = 1.;
            }
        }
        let a = boxes(&p, 20, 20, 20, 20);
        let b = boxes_with_params(
            &p,
            20,
            20,
            20,
            20,
            Params {
                unclip_ratio: 0.,
                ..Params::default()
            },
        );
        assert_eq!(a.len(), b.len());
        assert!(a[0].polygon[0][0] < b[0].polygon[0][0]);
        assert!(a[0].polygon[1][0] > b[0].polygon[1][0]);
    }
    #[test]
    fn parameter_defaults_and_invalid_values() {
        let p: Params = serde_json::from_str("{\"bin_thresh\":0.4}").unwrap();
        assert_eq!(p.unclip_ratio, 1.5);
        assert!(p.valid());
        assert!(
            !Params {
                bin_thresh: f32::NAN,
                ..p
            }
            .valid()
        );
        assert!(
            !Params {
                bin_thresh: 1.,
                ..p
            }
            .valid()
        );
        assert!(
            !Params {
                unclip_ratio: -1.,
                ..p
            }
            .valid()
        );
        assert!(
            !Params {
                box_thresh: f32::INFINITY,
                ..p
            }
            .valid()
        );
    }
    #[test]
    fn matches_doctr_fixtures() {
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../testdata/rust_postprocess.json")).unwrap();
        for (index, c) in cases.as_array().unwrap().iter().enumerate() {
            let p: Vec<f32> = c["prob"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            let actual = boxes(
                &p,
                c["h"].as_u64().unwrap() as usize,
                c["w"].as_u64().unwrap() as usize,
                c["ih"].as_u64().unwrap() as usize,
                c["iw"].as_u64().unwrap() as usize,
            );
            let expected = c["boxes"].as_array().unwrap();
            assert_eq!(actual.len(), expected.len(), "fixture {index}");
            for (b, e) in actual.iter().zip(expected) {
                for (v, t) in b
                    .polygon
                    .iter()
                    .flatten()
                    .chain(std::iter::once(&b.objectness))
                    .zip(e.as_array().unwrap())
                {
                    assert!(
                        (*v - t.as_f64().unwrap() as f32).abs() < 1e-5,
                        "fixture {index}: {b:?} vs {e}"
                    );
                }
            }
        }
    }
    #[test]
    fn empty_is_empty() {
        assert!(boxes(&[0.; 64], 8, 8, 8, 8).is_empty());
    }
    #[test]
    fn opening_removes_single_pixel() {
        let mut p = [0.; 64];
        p[27] = 1.;
        assert!(boxes(&p, 8, 8, 8, 8).is_empty());
    }
    #[test]
    fn rectangle_expansion() {
        let mut p = vec![0.; 400];
        for y in 6..10 {
            for x in 5..13 {
                p[y * 20 + x] = 1.;
            }
        }
        let b = boxes(&p, 20, 20, 20, 20);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].polygon, [[3. / 20., 4. / 20.], [16. / 20., 13. / 20.]]);
    }
}
