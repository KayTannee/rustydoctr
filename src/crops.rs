//! docTR-compatible upright extraction and wide-crop split/remap (Apache-2.0).
use crate::detection::Word;
use anyhow::{Result, ensure};
use image::RgbImage;
#[derive(Clone, Debug)]
pub struct Mapping {
    pub start: usize,
    pub end: usize,
    pub last_overlap: f64,
}

pub fn extract(image: &RgbImage, words: &[Word]) -> Result<(Vec<RgbImage>, Vec<Mapping>)> {
    let mut crops = Vec::new();
    let mut maps = Vec::new();
    for word in words {
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
        ensure!(x1 > x0 && y1 > y0, "Empty crop");
        let (cw, ch) = (x1 - x0, y1 - y0);
        let start = crops.len();
        let mut last_overlap = 0.;
        if cw as f64 / ch as f64 > 8. {
            let sw = ch * 6;
            let step = sw - sw / 2;
            let mut starts: Vec<u32> = (0..=cw - sw).step_by(step as usize).collect();
            if starts.last().unwrap() + sw < cw {
                starts.push(cw - sw);
            }
            if starts.len() > 1 {
                last_overlap =
                    (starts[starts.len() - 2] + sw - starts[starts.len() - 1]) as f64 / sw as f64;
            }
            for sx in starts {
                crops.push(image::imageops::crop_imm(image, x0 + sx, y0, sw, ch).to_image());
            }
        } else {
            crops.push(image::imageops::crop_imm(image, x0, y0, cw, ch).to_image());
        }
        maps.push(Mapping {
            start,
            end: crops.len(),
            last_overlap,
        });
    }
    Ok((crops, maps))
}
pub fn merge(a: &str, b: &str, ratio: f64) -> String {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().min(b.len()) <= 1 {
        return a.iter().chain(&b).collect();
    }
    let (ac, bc) = (&a[..a.len() - 1], &b[1..]);
    let scores: Vec<usize> = (1..=ac.len().min(bc.len()))
        .map(|n| {
            ac[ac.len() - n..]
                .iter()
                .zip(&bc[..n])
                .filter(|(a, b)| a != b)
                .count()
        })
        .collect();
    let expected = (b.len() as f64 * ratio).round_ties_even() as i64 - 3;
    let zeros: Vec<usize> = scores
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| if s == 0 { Some(i) } else { None })
        .collect();
    let best = if !zeros.is_empty() {
        *zeros
            .iter()
            .min_by_key(|&&i| (i as i64 - expected).abs())
            .unwrap()
    } else if expected < -1 {
        return a.iter().chain(&b).collect();
    } else if expected < 0 {
        return ac.iter().chain(bc).collect();
    } else {
        scores
            .iter()
            .enumerate()
            .min_by_key(|(i, s)| **s as i64 + (*i as i64 - expected).abs())
            .unwrap()
            .0
    };
    ac.iter().chain(&bc[best + 1..]).collect()
}
pub fn remap(words: &mut [Word], parts: &[(String, f32)], maps: &[Mapping]) {
    for (word, m) in words.iter_mut().zip(maps) {
        let slice = &parts[m.start..m.end];
        let mut text = slice[0].0.clone();
        for (i, (next, _)) in slice.iter().enumerate().skip(1) {
            text = merge(
                &text,
                next,
                if i + 1 == slice.len() {
                    m.last_overlap
                } else {
                    0.5
                },
            );
        }
        word.text = text;
        word.confidence = slice.iter().map(|p| p.1).sum::<f32>() / slice.len() as f32;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_doctr_merge_oracle() {
        let cases: Vec<(String, String, f64, String)> =
            serde_json::from_str(include_str!("../testdata/rust_crop_merge.json")).unwrap();
        for (a, b, ratio, expected) in cases {
            assert_eq!(merge(&a, &b, ratio), expected, "{a:?}, {b:?}, {ratio}");
        }
    }
    #[test]
    fn merging() {
        assert_eq!(merge("abcd", "cdefgh", 0.5), "abcefgh");
        assert_eq!(merge("abcdi", "cdefgh", 0.5), "abcdefgh");
        assert_eq!(merge("", "a", 0.5), "a");
    }
    #[test]
    fn splitting_covers_end() {
        let img = RgbImage::new(105, 10);
        let w = Word {
            quadrilateral: None,
            polygon: [[0., 0.], [1., 1.]],
            objectness: 1.,
            text: String::new(),
            confidence: 0.,
        };
        let (c, m) = extract(&img, &[w]).unwrap();
        assert_eq!(c.len(), 3);
        assert_eq!(c[0].width(), 60);
        assert_eq!(m[0].last_overlap, 0.75);
    }
}
