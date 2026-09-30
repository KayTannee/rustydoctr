//! Bounded crop alternatives; preserves detections and uses only recognizer-produced text.
use crate::{crops::Mapping, detection::Word};
use image::RgbImage;
use serde::{Deserialize, Serialize};

pub const MAX_EXTRA_CROPS: usize = 32;
const MAX_EXTRA_PIXELS: usize = 1_000_000;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub rotation_deg: i16,
    pub text: String,
    pub confidence: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Decision {
    pub evidence: String,
    pub original_text: String,
    pub original_confidence: f32,
    pub selected_rotation_deg: Option<i16>,
    pub candidates: Vec<Candidate>,
}
#[derive(Clone, Copy)]
struct BoxPx {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}
impl BoxPx {
    fn width(self) -> f32 {
        self.x1 - self.x0
    }
    fn height(self) -> f32 {
        self.y1 - self.y0
    }
    fn cx(self) -> f32 {
        (self.x0 + self.x1) * 0.5
    }
    fn cy(self) -> f32 {
        (self.y0 + self.y1) * 0.5
    }
}
fn peers(boxes: &[BoxPx], i: usize, vertical: bool) -> Vec<usize> {
    let a = boxes[i];
    boxes
        .iter()
        .enumerate()
        .filter_map(|(j, &b)| {
            if i == j {
                return None;
            }
            let aligned = if vertical {
                b.height() >= 1.6 * b.width()
                    && (a.cx() - b.cx()).abs() < 0.4 * b.width()
                    && (a.cy() - b.cy()).abs() < 6. * b.width()
                    && a.width() / b.width() > 0.5
                    && a.width() / b.width() < 1.8
            } else {
                b.width() >= 1.3 * b.height()
                    && (a.cy() - b.cy()).abs() < 0.45 * b.height()
                    && (a.cx() - b.cx()).abs() < 6. * b.height()
                    && a.height() / b.height() > 0.35
                    && a.height() / b.height() < 1.8
            };
            aligned.then_some(j)
        })
        .collect()
}

pub fn prepare(
    image: &RgbImage,
    words: &mut [Word],
    crops: &mut Vec<RgbImage>,
    maps: &mut [Mapping],
) {
    let boxes: Vec<_> = words
        .iter()
        .map(|w| BoxPx {
            x0: w.polygon[0][0] * image.width() as f32,
            y0: w.polygon[0][1] * image.height() as f32,
            x1: w.polygon[1][0] * image.width() as f32,
            y1: w.polygon[1][1] * image.height() as f32,
        })
        .collect();
    let original_count = crops.len();
    let mut pixels = 0;
    for i in 0..words.len() {
        if maps[i].end - maps[i].start != 1 {
            continue;
        }
        let b = boxes[i];
        let horizontal = peers(&boxes, i, false);
        let mut alternatives = Vec::new();
        let evidence = if horizontal.len() >= 2 && b.width() < 0.8 * b.height() {
            // A narrow glyph inside a horizontal line inherits upright direction.
            // Borrow the neighbours' vertical extent, with a little blank side context.
            let mut tops: Vec<_> = horizontal.iter().map(|&j| boxes[j].y0).collect();
            let mut bottoms: Vec<_> = horizontal.iter().map(|&j| boxes[j].y1).collect();
            tops.sort_by(f32::total_cmp);
            bottoms.sort_by(f32::total_cmp);
            let margin = 0.15 * b.height();
            let x0 = (b.x0 - margin).floor().max(0.) as u32;
            let x1 = (b.x1 + margin).ceil().min(image.width() as f32) as u32;
            let y0 = tops[tops.len() / 2].min(b.y0).floor().max(0.) as u32;
            let y1 = bottoms[bottoms.len() / 2]
                .max(b.y1)
                .ceil()
                .min(image.height() as f32) as u32;
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            // Do not include a neighbour's ink box in the alternative.
            if boxes.iter().enumerate().any(|(j, p)| {
                j != i
                    && p.x0 < x1 as f32
                    && p.x1 > x0 as f32
                    && p.y0 < y1 as f32
                    && p.y1 > y0 as f32
            }) {
                continue;
            }
            if ((x1 - x0) as usize) * (y1 - y0) as usize > MAX_EXTRA_PIXELS - pixels {
                continue;
            }
            alternatives.push((
                image::imageops::crop_imm(image, x0, y0, x1 - x0, y1 - y0).to_image(),
                0,
            ));
            "horizontal_line_context"
        } else if horizontal.len() < 2
            && b.height() > 1.8 * b.width()
            && peers(&boxes, i, true).len() >= 2
        {
            let crop = &crops[maps[i].start];
            if 2 * crop.width() as usize * crop.height() as usize > MAX_EXTRA_PIXELS - pixels {
                continue;
            }
            alternatives.push((image::imageops::rotate90(crop), 90));
            alternatives.push((image::imageops::rotate270(crop), 270));
            "vertical_line_neighbours"
        } else {
            continue;
        };
        if crops.len() - original_count + alternatives.len() > MAX_EXTRA_CROPS {
            continue;
        }
        words[i].crop_decision = Some(Decision {
            evidence: evidence.into(),
            original_text: String::new(),
            original_confidence: 0.,
            selected_rotation_deg: None,
            candidates: vec![],
        });
        for (crop, angle) in alternatives {
            pixels += crop.width() as usize * crop.height() as usize;
            maps[i].alternatives.push((crops.len(), angle));
            crops.push(crop);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn word(x: f32, y: f32, w: f32, h: f32) -> Word {
        Word {
            thin_recovery: None,
            polygon: [[x / 500., y / 500.], [(x + w) / 500., (y + h) / 500.]],
            quadrilateral: None,
            objectness: 1.,
            text: String::new(),
            confidence: 0.,
            crop_decision: None,
        }
    }
    #[test]
    fn narrow_character_in_line_is_never_rotated() {
        let image = RgbImage::new(500, 500);
        let mut words = vec![
            word(40., 40., 50., 20.),
            word(105., 40., 6., 20.),
            word(125., 40., 50., 20.),
        ];
        let (mut crops, mut maps) = crate::crops::extract(&image, &words).unwrap();
        prepare(&image, &mut words, &mut crops, &mut maps);
        assert_eq!(maps[1].alternatives.len(), 1);
        assert_eq!(maps[1].alternatives[0].1, 0);
        assert!(maps[0].alternatives.is_empty());
    }
    #[test]
    fn isolated_tall_glyph_has_no_direction_evidence() {
        let image = RgbImage::new(500, 500);
        let mut words = vec![word(40., 40., 6., 20.)];
        let (mut crops, mut maps) = crate::crops::extract(&image, &words).unwrap();
        prepare(&image, &mut words, &mut crops, &mut maps);
        assert_eq!(crops.len(), 1);
    }
    #[test]
    fn vertical_group_gets_two_directions_and_selection_is_gated() {
        let image = RgbImage::new(500, 500);
        let mut words = vec![
            word(40., 40., 20., 40.),
            word(40., 95., 20., 40.),
            word(40., 150., 20., 40.),
        ];
        let (mut crops, mut maps) = crate::crops::extract(&image, &words).unwrap();
        prepare(&image, &mut words, &mut crops, &mut maps);
        assert_eq!(maps[1].alternatives.len(), 2);
        let mut parts = vec![("bad".into(), 0.4); crops.len()];
        for m in &maps {
            for &(i, angle) in &m.alternatives {
                parts[i] = if angle == 90 {
                    ("I".into(), 0.7)
                } else {
                    ("label".into(), 0.95)
                };
            }
        }
        crate::crops::remap(&mut words, &parts, &maps);
        assert_eq!(words[1].text, "label");
        assert_eq!(
            words[1]
                .crop_decision
                .as_ref()
                .unwrap()
                .selected_rotation_deg,
            Some(270)
        );
    }
    #[test]
    fn competing_confident_directions_abstain() {
        let image = RgbImage::new(500, 500);
        let mut words = vec![
            word(40., 40., 20., 40.),
            word(40., 95., 20., 40.),
            word(40., 150., 20., 40.),
        ];
        let (mut crops, mut maps) = crate::crops::extract(&image, &words).unwrap();
        prepare(&image, &mut words, &mut crops, &mut maps);
        let mut parts = vec![(">".into(), 0.5); crops.len()];
        for map in &maps {
            for &(index, angle) in &map.alternatives {
                parts[index] = (if angle == 90 { "VI" } else { "IA" }.into(), 0.999);
            }
        }
        crate::crops::remap(&mut words, &parts, &maps);
        assert_eq!(words[1].text, ">");
        assert_eq!(
            words[1]
                .crop_decision
                .as_ref()
                .unwrap()
                .selected_rotation_deg,
            None
        );
        assert_eq!(words[1].crop_decision.as_ref().unwrap().candidates.len(), 2);
    }
    #[test]
    fn alternatives_have_a_hard_page_cap() {
        let image = RgbImage::new(500, 500);
        let mut words = Vec::new();
        for y in (10..470).step_by(10) {
            words.extend([
                word(40., y as f32, 20., 6.),
                word(65., y as f32, 2., 6.),
                word(75., y as f32, 20., 6.),
            ]);
        }
        let (mut crops, mut maps) = crate::crops::extract(&image, &words).unwrap();
        let original = crops.len();
        prepare(&image, &mut words, &mut crops, &mut maps);
        assert_eq!(crops.len() - original, MAX_EXTRA_CROPS);
    }
}
