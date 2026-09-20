//! Aspect-preserving bilinear antialiased resize, uint8 rounding, padding and CHW normalization.
use image::RgbImage;

fn weights(input: usize, output: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = input as f64 / output as f64;
    let support = scale.max(1.0);
    (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5) as isize).max(0) as usize;
            let hi = ((center + support + 0.5) as usize).min(input);
            let mut w: Vec<_> = (lo..hi)
                .map(|j| {
                    (
                        j,
                        (1.0 - ((j as f64 + 0.5 - center) / support).abs()).max(0.0) as f32,
                    )
                })
                .collect();
            let total: f32 = w.iter().map(|x| x.1).sum();
            for x in &mut w {
                x.1 /= total;
            }
            w
        })
        .collect()
}

pub fn prepare(
    image: &RgbImage,
    height: usize,
    width: usize,
    symmetric: bool,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Vec<f32> {
    let (iw, ih) = (image.width() as usize, image.height() as usize);
    let ratio = ih as f64 / iw as f64;
    let (rh, rw) = if ratio > height as f64 / width as f64 {
        (height, ((height as f64 / ratio) as usize).max(1))
    } else {
        (((width as f64 * ratio) as usize).max(1), width)
    };
    let wx = weights(iw, rw);
    let wy = weights(ih, rh);
    let mut horizontal = vec![0f32; ih * rw * 3];
    for y in 0..ih {
        for (x, weights) in wx.iter().enumerate() {
            for c in 0..3 {
                horizontal[(y * rw + x) * 3 + c] = weights
                    .iter()
                    .map(|&(sx, w)| image.as_raw()[(y * iw + sx) * 3 + c] as f32 * w)
                    .sum();
            }
        }
    }
    let mut result = vec![0f32; 3 * height * width];
    for c in 0..3 {
        result[c * height * width..(c + 1) * height * width].fill(-mean[c] / std[c]);
    }
    let (ox, oy) = if symmetric {
        ((width - rw).div_ceil(2), (height - rh).div_ceil(2))
    } else {
        (0, 0)
    };
    for (y, weights) in wy.iter().enumerate() {
        for x in 0..rw {
            for c in 0..3 {
                let value: f32 = weights
                    .iter()
                    .map(|&(sy, w)| horizontal[(sy * rw + x) * 3 + c] * w)
                    .sum();
                let value = value.round_ties_even().clamp(0.0, 255.0) / 255.0;
                result[c * height * width + (y + oy) * width + x + ox] = (value - mean[c]) / std[c];
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn padding_and_channels() {
        let img = RgbImage::from_pixel(2, 4, image::Rgb([255, 128, 0]));
        let x = prepare(&img, 4, 4, true, &[0.; 3], &[1.; 3]);
        assert_eq!(x[0], 0.);
        assert_eq!(x[1], 1.);
        assert_eq!(x[3], 0.);
        assert!((x[17] - 128. / 255.).abs() < 1e-6);
    }
}
