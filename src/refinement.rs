//! Experimental image-only dense-band selection. At most two fixed 1024 tiles.
use crate::detection::Word;
use image::RgbImage;
use serde::Serialize;

pub const SIZE: usize = 1024;
#[derive(Clone, Debug, Serialize)]
pub struct Tile {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub owner_start: u32,
    pub owner_end: u32,
}

pub fn select(image: &RgbImage, size: usize) -> Vec<Tile> {
    let (w, h) = (image.width() as usize, image.height() as usize);
    if w < 4 || h < 4 {
        return vec![];
    }
    // One byte per pixel and a reusable component stack, released before inference.
    let mut ink: Vec<u8> = image
        .pixels()
        .map(|p| {
            let gray = (4899 * p[0] as u32 + 9617 * p[1] as u32 + 1868 * p[2] as u32 + 8192) >> 14;
            u8::from(gray < 160)
        })
        .collect();
    let mut bins = vec![Vec::<(usize, usize)>::new(); h / 64 + 1];
    let mut stack = Vec::new();
    for seed in 0..ink.len() {
        if ink[seed] == 0 {
            continue;
        }
        ink[seed] = 0;
        stack.push(seed);
        let (mut x0, mut x1, mut y0, mut y1, mut area) = (w, 0, h, 0, 0);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
            area += 1;
            for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let j = yy * w + xx;
                    if ink[j] != 0 {
                        ink[j] = 0;
                        stack.push(j);
                    }
                }
            }
        }
        let (cw, ch) = (x1 - x0 + 1, y1 - y0 + 1);
        if area >= 3
            && ch >= 3
            && ch as f64 <= h as f64 * 0.04
            && cw as f64 <= w as f64 * 0.04
            && ch <= cw * 12
        {
            bins[(y0 + ch / 2) / 64].push((y0, ch));
        }
    }
    let mut groups: Vec<Vec<usize>> = vec![];
    for (k, bin) in bins.iter().enumerate() {
        if bin.len() < 20 {
            continue;
        }
        let mut heights: Vec<_> = bin.iter().map(|x| x.1).collect();
        heights.sort_unstable();
        let n = heights.len();
        let median = (heights[(n - 1) / 2] + heights[n / 2]) as f64 / 2.;
        if median * size as f64 / w.max(h) as f64 >= 9. {
            continue;
        }
        if let Some(g) = groups.last_mut()
            && *g.last().unwrap() + 1 == k
        {
            g.push(k);
            continue;
        }
        groups.push(vec![k]);
    }
    let Some(group) = groups
        .iter()
        .max_by_key(|g| g.iter().map(|&k| bins[k].len()).sum::<usize>())
    else {
        return vec![];
    };
    let y0 = group
        .iter()
        .flat_map(|&k| bins[k].iter())
        .map(|x| x.0)
        .min()
        .unwrap()
        .saturating_sub(24);
    let y1 = (group
        .iter()
        .flat_map(|&k| bins[k].iter())
        .map(|x| x.0 + x.1)
        .max()
        .unwrap()
        + 24)
        .min(h);
    // Bound extra source area; do not turn a whole dense page into an unbounded rescan.
    if y1 - y0 > h / 4 {
        return vec![];
    }
    let mid = w / 2;
    let overlap = w * 4 / 100;
    [(0, mid + overlap, 0, mid), (mid - overlap, w, mid, w)]
        .into_iter()
        .map(|(x, end, lo, hi)| Tile {
            x: x as u32,
            y: y0 as u32,
            width: (end - x) as u32,
            height: (y1 - y0) as u32,
            owner_start: lo as u32,
            owner_end: hi as u32,
        })
        .collect()
}

/// Reconcile before recognition. Empty tile detections retain the original words.
pub fn merge(base: &mut Vec<Word>, mut words: Vec<Word>, tile: &Tile, w: u32, h: u32) {
    if words.is_empty() {
        return;
    }
    for word in &mut words {
        for point in &mut word.polygon {
            point[0] = (point[0] * tile.width as f32 + tile.x as f32) / w as f32;
            point[1] = (point[1] * tile.height as f32 + tile.y as f32) / h as f32;
        }
    }
    let owns = |word: &Word| {
        let x = (word.polygon[0][0] + word.polygon[1][0]) * 0.5 * w as f32;
        let y = (word.polygon[0][1] + word.polygon[1][1]) * 0.5 * h as f32;
        x >= tile.owner_start as f32
            && x < (tile.owner_end as f32)
            && y >= tile.y as f32
            && y <= (tile.y + tile.height) as f32
    };
    words.retain(&owns);
    if words.is_empty() {
        return;
    }
    base.retain(|word| !owns(word));
    base.extend(words);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn word(x: f32, y: f32) -> Word {
        Word {
            quadrilateral: None,
            polygon: [[x - 0.01, y - 0.01], [x + 0.01, y + 0.01]],
            objectness: 1.,
            text: String::new(),
            confidence: 0.,
        }
    }
    #[test]
    fn merge_preserves_outside_words_and_owns_seam_once() {
        let left = Tile {
            x: 0,
            y: 50,
            width: 60,
            height: 20,
            owner_start: 0,
            owner_end: 50,
        };
        let right = Tile {
            x: 40,
            y: 50,
            width: 60,
            height: 20,
            owner_start: 50,
            owner_end: 100,
        };
        let mut base = vec![word(0.2, 0.1), word(0.2, 0.6), word(0.8, 0.6)];
        merge(&mut base, vec![], &left, 100, 100);
        assert_eq!(base.len(), 3);
        // Both tiles see x=50, but only the right tile owns its center.
        merge(
            &mut base,
            vec![word(20. / 60., 0.5), word(50. / 60., 0.5)],
            &left,
            100,
            100,
        );
        merge(
            &mut base,
            vec![word(10. / 60., 0.5), word(40. / 60., 0.5)],
            &right,
            100,
            100,
        );
        assert_eq!(base.len(), 4);
        assert!((base[0].polygon[0][1] - 0.09).abs() < 1e-6);
        assert_eq!(
            base.iter()
                .filter(|w| ((w.polygon[0][0] + w.polygon[1][0]) * 0.5 - 0.5).abs() < 1e-6)
                .count(),
            1
        );
    }
    #[test]
    fn empty_and_large_blocks_do_not_trigger() {
        assert!(select(&RgbImage::from_pixel(400, 600, image::Rgb([255; 3])), 1536).is_empty());
        assert!(select(&RgbImage::from_pixel(400, 600, image::Rgb([0; 3])), 1536).is_empty());
    }
    #[test]
    fn dense_small_text_selects_bounded_overlapping_tiles() {
        let mut image = RgbImage::from_pixel(1000, 1600, image::Rgb([255; 3]));
        for row in 0..3 {
            for col in 0..40 {
                for y in 1200 + row * 12..1206 + row * 12 {
                    for x in 20 + col * 20..24 + col * 20 {
                        image.put_pixel(x, y, image::Rgb([0; 3]));
                    }
                }
            }
        }
        let tiles = select(&image, 1536);
        assert_eq!(tiles.len(), 2);
        assert_eq!(tiles[0].owner_end, tiles[1].owner_start);
        assert!(tiles[0].x + tiles[0].width > tiles[1].x);
        assert!(tiles[0].height <= 400);
    }
}
