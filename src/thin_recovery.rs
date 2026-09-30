//! Conservative, bounded proposals from thin DB components discarded by opening.
use crate::detection::Word;
use serde::{Deserialize, Serialize};

pub const MAX_CROPS: usize = 32;
const MAX_PIXELS: usize = 1_000_000;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Evidence {
    pub source: String,
    pub map_size: [usize; 2],
    pub raw_box: [usize; 4],
    pub anchors: [[f64; 4]; 2],
}

pub struct Geometry {
    pub height: usize,
    pub width: usize,
    pub image_height: usize,
    pub image_width: usize,
}

pub fn propose(
    prob: &[f32],
    mask: &[bool],
    opened: &[bool],
    anchors: &[[f64; 4]],
    g: Geometry,
) -> Vec<Word> {
    let (h, w) = (g.height, g.width);
    let mut seen = vec![false; h * w];
    let mut stack = Vec::new();
    let mut result = Vec::new();
    // Vertical buckets avoid scanning every page word for each discarded component.
    let mut bins = vec![Vec::new(); h / 16 + 1];
    for (i, a) in anchors.iter().enumerate() {
        bins[((a[1] + a[3] / 2.) as usize / 16).min(h / 16)].push(i);
    }
    for seed in 0..mask.len() {
        if !mask[seed] || seen[seed] {
            continue;
        }
        seen[seed] = true;
        stack.push(seed);
        let (mut x0, mut y0, mut x1, mut y1) = (seed % w, seed / w, seed % w, seed / w);
        let (mut area, mut sum, mut survives) = (0usize, 0f64, false);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
            area += 1;
            sum += prob[i] as f64;
            survives |= opened[i];
            for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let j = yy * w + xx;
                    if mask[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        let (cw, ch) = ((x1 - x0 + 1) as f64, (y1 - y0 + 1) as f64);
        if survives
            || !(3. ..=60.).contains(&ch)
            || cw > ch * 0.6
            || area < 3
            || (area as f64) / (cw * ch) < 0.25
        {
            continue;
        }
        let (x, y) = (x0 as f64, y0 as f64);
        let lo = ((y + ch / 2. - 0.875 * ch).max(0.) as usize / 16).min(h / 16);
        let hi = ((y + ch / 2. + 0.875 * ch).ceil() as usize / 16).min(h / 16);
        let mut peers: Vec<_> = bins[lo..=hi]
            .iter()
            .flatten()
            .map(|&i| &anchors[i])
            .filter_map(|&a| {
                let [ax, ay, aw, ah] = a;
                let gap = (ax - x - cw).max(x - ax - aw).max(0.);
                ((0.4..=2.).contains(&(ch / ah))
                    && (y + ch / 2. - ay - ah / 2.).abs() <= 0.35 * ch.max(ah)
                    && gap <= 5. * ch.max(ah))
                .then_some((gap, a))
            })
            .collect();
        if peers.len() == 1 {
            let [ax, ay, aw, ah] = peers[0].1;
            let acy = ay + ah / 2.;
            let lo = ((acy - 0.35 * ah / 0.65).max(0.) as usize / 16).min(h / 16);
            let hi = ((acy + 0.35 * ah / 0.65).ceil() as usize / 16).min(h / 16);
            let bridge = bins[lo..=hi]
                .iter()
                .flatten()
                .filter_map(|&i| {
                    let b @ [bx, by, bw, bh] = anchors[i];
                    let gap = (bx - ax - aw).max(ax - bx - bw).max(0.);
                    ((bx + bw / 2. - ax - aw / 2.) * (ax + aw / 2. - x - cw / 2.) > 0.
                        && (bx >= ax + aw || bx + bw <= ax)
                        && (0.65..=1.6).contains(&(ah / bh))
                        && (acy - by - bh / 2.).abs() <= 0.35 * ah.max(bh)
                        && gap <= 5. * ah.max(bh))
                    .then_some((gap, b))
                })
                .min_by(|a, b| a.partial_cmp(b).unwrap());
            if let Some(bridge) = bridge {
                peers.push(bridge);
            }
        }
        if peers.len() < 2 {
            continue;
        }
        peers.sort_by(|a, b| {
            a.0.total_cmp(&b.0).then_with(|| {
                a.1.iter()
                    .zip(b.1.iter())
                    .map(|(a, b)| a.total_cmp(b))
                    .find(|o| !o.is_eq())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        let top = (peers[0].1[1] + peers[1].1[1]) / 2.;
        let bottom = (peers[0].1[1] + peers[0].1[3] + peers[1].1[1] + peers[1].1[3]) / 2.;
        let pad = 0.3 * ch.max(bottom - top);
        let score = (sum / area as f64) as f32;
        if score < 0.3 {
            continue;
        }
        let mut b = [
            [
                (x - pad).max(0.) / w as f64,
                (y.min(top) - pad).max(0.) / h as f64,
            ],
            [
                (x + cw + pad).min(w as f64) / w as f64,
                ((y + ch).max(bottom) + pad).min(h as f64) / h as f64,
            ],
        ];
        for p in &mut b {
            if g.image_height > g.image_width {
                p[0] = (p[0] - 0.5) * g.image_height as f64 / g.image_width as f64 + 0.5;
            } else {
                p[1] = (p[1] - 0.5) * g.image_width as f64 / g.image_height as f64 + 0.5;
            }
        }
        result.push(Word {
            polygon: b.map(|p| p.map(|v| v.clamp(0., 1.) as f32)),
            quadrilateral: None,
            objectness: score,
            text: String::new(),
            confidence: 0.,
            crop_decision: None,
            thin_recovery: Some(Evidence {
                source: "detector_map".into(),
                map_size: [h, w],
                raw_box: [x0, y0, cw as usize, ch as usize],
                anchors: [peers[0].1, peers[1].1],
            }),
        });
        result.sort_by(|a, b| b.objectness.total_cmp(&a.objectness));
        result.truncate(MAX_CROPS);
    }
    result.sort_by(|a, b| b.objectness.total_cmp(&a.objectness));
    result.truncate(MAX_CROPS);
    result
}

fn overlaps(a: &Word, b: &Word) -> bool {
    a.polygon[0][0] < b.polygon[1][0]
        && a.polygon[1][0] > b.polygon[0][0]
        && a.polygon[0][1] < b.polygon[1][1]
        && a.polygon[1][1] > b.polygon[0][1]
}

/// Append separate proposals only, while keeping the page's existing admission credit.
pub fn append(words: &mut Vec<Word>, mut candidates: Vec<Word>, width: u32, height: u32) {
    // Deduplicate in map order (tiles before full page), then prioritize confidence.
    let mut separate = Vec::new();
    for candidate in candidates.drain(..) {
        if candidate.polygon[1][0] <= candidate.polygon[0][0]
            || candidate.polygon[1][1] <= candidate.polygon[0][1]
        {
            continue;
        }
        if !words
            .iter()
            .chain(separate.iter())
            .any(|w| overlaps(w, &candidate))
        {
            separate.push(candidate);
        }
    }
    separate.sort_by(|a, b| b.objectness.total_cmp(&a.objectness));
    let (mut count, mut pixels) = (0, 0);
    for candidate in separate {
        if count == MAX_CROPS {
            break;
        }
        let b = candidate.polygon;
        let cw = (b[1][0] * width as f32).ceil() as usize
            - (b[0][0] * width as f32).floor() as usize
            + 2;
        let ch = (b[1][1] * height as f32).ceil() as usize
            - (b[0][1] * height as f32).floor() as usize
            + 2;
        if pixels + cw * ch > MAX_PIXELS {
            continue;
        }
        pixels += cw * ch;
        count += 1;
        words.push(candidate);
    }
}

pub fn retain_accepted(words: &mut Vec<Word>) {
    words.retain(|w| {
        w.thin_recovery.is_none()
            || (w.confidence >= 0.9
                && (1..=2).contains(&w.text.chars().count())
                && w.text.chars().all(char::is_alphanumeric))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Vec<f32> {
        let mut p = vec![0.; 100 * 100];
        for y in 40..50 {
            for x in (10..30).chain(60..80) {
                p[y * 100 + x] = 0.9;
            }
        }
        for y in 41..49 {
            for x in 44..46 {
                p[y * 100 + x] = 0.8;
            }
        }
        p
    }
    #[test]
    fn recovers_discarded_component_without_changing_original_boxes() {
        let p = fixture();
        let base = crate::detection::boxes(&p, 100, 100, 100, 100);
        let (unchanged, extra) =
            crate::detection::boxes_and_thin(&p, 100, 100, 100, 100, Default::default(), true);
        assert_eq!(
            serde_json::to_value(base).unwrap(),
            serde_json::to_value(unchanged).unwrap()
        );
        assert_eq!(extra.len(), 1);
        assert_eq!(
            extra[0].thin_recovery.as_ref().unwrap().raw_box,
            [44, 41, 2, 8]
        );
        assert!((extra[0].polygon[0][1] - 0.37).abs() < 1e-6);
        let mut p = p;
        for y in 0..100 {
            for x in 60..100 {
                p[y * 100 + x] = 0.;
            }
        }
        assert!(
            crate::detection::boxes_and_thin(&p, 100, 100, 100, 100, Default::default(), true)
                .1
                .is_empty()
        );
    }
    #[test]
    fn one_hop_requires_an_aligned_outward_neighbour() {
        let run = |second_x: usize, second_y: usize, signal: f32| {
            let mut p = vec![0.; 150 * 150];
            for y in 40..50 {
                for x in 32..52 {
                    p[y * 150 + x] = 0.9;
                }
            }
            for y in second_y..second_y + 10 {
                for x in second_x..second_x + 20 {
                    p[y * 150 + x] = 0.9;
                }
            }
            for y in 41..49 {
                for x in 20..22 {
                    p[y * 150 + x] = signal;
                }
            }
            crate::detection::boxes_and_thin(&p, 150, 150, 150, 150, Default::default(), true).1
        };
        assert_eq!(run(85, 40, 0.8).len(), 1);
        assert_eq!(run(60, 40, 0.8).len(), 1); // Two direct anchors still produce one crop.
        assert!(run(85, 75, 0.8).is_empty());
        assert!(run(120, 40, 0.8).is_empty());
        assert!(run(85, 40, 0.2).is_empty());
    }
    #[test]
    fn deduplicates_caps_and_filters_untrusted_recognition() {
        let candidate = crate::detection::boxes_and_thin(
            &fixture(),
            100,
            100,
            100,
            100,
            Default::default(),
            true,
        )
        .1
        .remove(0);
        let mut words = vec![];
        append(
            &mut words,
            vec![candidate.clone(), candidate.clone()],
            100,
            100,
        );
        assert_eq!(words.len(), 1);
        words[0].text = "-".into();
        words[0].confidence = 0.999;
        retain_accepted(&mut words);
        assert!(words.is_empty());
        let mut c = candidate;
        c.text = "I".into();
        c.confidence = 0.99;
        words.push(c);
        retain_accepted(&mut words);
        assert_eq!(words.len(), 1);
        let mut many = Vec::new();
        for i in 0..50 {
            let mut c = words[0].clone();
            c.polygon = [[i as f32 / 50., 0.1], [i as f32 / 50. + 0.01, 0.11]];
            many.push(c);
        }
        words.clear();
        append(&mut words, many, 100, 100);
        assert_eq!(words.len(), MAX_CROPS);
        words.clear();
        append(&mut words, vec![candidate_for_budget()], 100_000, 100_000);
        assert!(words.is_empty());
    }
    fn candidate_for_budget() -> Word {
        crate::detection::boxes_and_thin(&fixture(), 100, 100, 100, 100, Default::default(), true)
            .1
            .remove(0)
    }
}
