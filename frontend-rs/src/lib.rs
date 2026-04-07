mod gpu;
mod utils;

use wasm_bindgen::prelude::*;

// ── Color space conversion math ─────────────────────────────────────────────
//
// All pixel data flowing through the wasm boundary is interpreted as
// Display P3. The editor reads source pixels via getImageData with
// `{ colorSpace: 'display-p3' }` and renders the same way.
//
// Our perceptual working space is **OKLab** (Björn Ottosson, 2020), which
// has substantially better hue uniformity than CIELAB — particularly
// across blues, where CIELAB famously drifts toward purple as L changes.
//
// To keep the user-facing slider numbers familiar (L still 0..100, chroma
// magnitudes around 0..160 like before), we scale OKLab values:
//   L_scaled = OKLab L * 100
//   a_scaled = OKLab a * 400
//   b_scaled = OKLab b * 400
// so chroma magnitudes line up with the previous CIELAB-shaped pipeline.

const L_SCALE: f32 = 100.0;
const AB_SCALE: f32 = 400.0;

// Linear Display P3 → LMS, precomputed as M_xyz_to_lms · M_p3_to_xyz.
const M_P3_TO_LMS: [[f32; 3]; 3] = [
    [0.4813371, 0.4620734, 0.0565038],
    [0.2288498, 0.6532486, 0.1179665],
    [0.0839833, 0.2242765, 0.6922382],
];

// LMS → linear Display P3, precomputed as M_xyz_to_p3 · M_lms_to_xyz.
const M_LMS_TO_P3: [[f32; 3]; 3] = [
    [3.1280366, -2.2571161, 0.1292834],
    [-1.0911312, 2.4133403, -0.3221775],
    [-0.0260093, -0.5080364, 1.5332833],
];

#[inline]
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert a single Display P3 RGB byte triple to scaled OKLab
/// (L*100, a*400, b*400).
fn rgb_bytes_to_lab(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let lr = srgb_to_linear(r as f32 / 255.0);
    let lg = srgb_to_linear(g as f32 / 255.0);
    let lb = srgb_to_linear(b as f32 / 255.0);

    // Linear P3 → LMS
    let l = M_P3_TO_LMS[0][0] * lr + M_P3_TO_LMS[0][1] * lg + M_P3_TO_LMS[0][2] * lb;
    let m = M_P3_TO_LMS[1][0] * lr + M_P3_TO_LMS[1][1] * lg + M_P3_TO_LMS[1][2] * lb;
    let s = M_P3_TO_LMS[2][0] * lr + M_P3_TO_LMS[2][1] * lg + M_P3_TO_LMS[2][2] * lb;

    // Nonlinear LMS' (cube root)
    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    // LMS' → OKLab
    let big_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    let aa = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    let bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;

    (big_l * L_SCALE, aa * AB_SCALE, bb * AB_SCALE)
}

/// Convert scaled OKLab back to a clamped 8-bit Display P3 RGB triple.
fn lab_to_rgb_bytes(l: f32, a: f32, b: f32) -> (u8, u8, u8) {
    let big_l = l / L_SCALE;
    let aa = a / AB_SCALE;
    let bb = b / AB_SCALE;

    // OKLab → LMS'
    let l_ = big_l + 0.3963377774 * aa + 0.2158037573 * bb;
    let m_ = big_l - 0.1055613458 * aa - 0.0638541728 * bb;
    let s_ = big_l - 0.0894841775 * aa - 1.2914855480 * bb;

    // LMS (cube)
    let lin_l = l_ * l_ * l_;
    let lin_m = m_ * m_ * m_;
    let lin_s = s_ * s_ * s_;

    // LMS → linear P3
    let lr = M_LMS_TO_P3[0][0] * lin_l + M_LMS_TO_P3[0][1] * lin_m + M_LMS_TO_P3[0][2] * lin_s;
    let lg = M_LMS_TO_P3[1][0] * lin_l + M_LMS_TO_P3[1][1] * lin_m + M_LMS_TO_P3[1][2] * lin_s;
    let lb = M_LMS_TO_P3[2][0] * lin_l + M_LMS_TO_P3[2][1] * lin_m + M_LMS_TO_P3[2][2] * lin_s;

    (
        (linear_to_srgb(lr.clamp(0.0, 1.0)) * 255.0).round() as u8,
        (linear_to_srgb(lg.clamp(0.0, 1.0)) * 255.0).round() as u8,
        (linear_to_srgb(lb.clamp(0.0, 1.0)) * 255.0).round() as u8,
    )
}

/// Precompute seam removal order using forward energy in LAB color space.
/// Returns a u32 array of size width*height where each value is the step at which
/// that pixel gets removed (1-indexed). Pixels surviving to the end get value = max_dimension.
/// direction: 0 = vertical seams (reduce width), 1 = horizontal seams (reduce height)
#[wasm_bindgen]
pub fn precompute_seam_order(
    image_data: &[u8],
    width: u32,
    height: u32,
    direction: u32,
) -> Vec<u32> {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    let w = width as usize;
    let h = height as usize;

    if direction == 0 {
        precompute_vertical_order(image_data, w, h)
    } else {
        // Transpose image, compute vertical order, map back
        let mut transposed = vec![0u8; image_data.len()];
        for y in 0..h {
            for x in 0..w {
                let src = (y * w + x) * 4;
                let dst = (x * h + y) * 4;
                transposed[dst..dst + 4].copy_from_slice(&image_data[src..src + 4]);
            }
        }
        let order_t = precompute_vertical_order(&transposed, h, w);
        // Map transposed coords back: order_t[tx * h + ty] → order[ty * w + tx]
        let mut order = vec![0u32; w * h];
        for y in 0..h {
            for x in 0..w {
                order[y * w + x] = order_t[x * h + y];
            }
        }
        order
    }
}

fn precompute_vertical_order(image_data: &[u8], w: usize, h: usize) -> Vec<u32> {
    // Convert to LAB, row-based for efficient removal
    // Convert each pixel to scaled OKLab once. OKLab gives a much
    // better perceptual distance for the energy function — particularly
    // around blues, where CIELAB drifts in hue.
    let mut rows: Vec<Vec<(f32, f32, f32)>> = Vec::with_capacity(h);
    for y in 0..h {
        let mut row = Vec::with_capacity(w);
        for x in 0..w {
            let i = (y * w + x) * 4;
            row.push(rgb_bytes_to_lab(
                image_data[i],
                image_data[i + 1],
                image_data[i + 2],
            ));
        }
        rows.push(row);
    }

    // Track original x indices for each row
    let mut indices: Vec<Vec<usize>> = (0..h).map(|_| (0..w).collect()).collect();
    let mut order = vec![0u32; w * h];
    let mut cur_w = w;

    for step in 1..w as u32 {
        if cur_w <= 1 {
            break;
        }

        // Build forward energy cost matrix with parent pointers for traceback
        // Reference: https://avikdas.com/2019/07/29/improved-seam-carving-with-forward-energy.html
        let mut cost = vec![0.0_f32; cur_w * h];
        // Parent direction: -1=came from above-left, 0=above, 1=above-right
        let mut parent = vec![0i8; cur_w * h];

        // Helper: compute C_U(x, y) = D[(x-1, y), (x+1, y)]
        // At boundaries, replace out-of-bounds pixel with current pixel
        let compute_c_u = |row: &[(f32, f32, f32)], x: usize, w: usize| -> f32 {
            if w <= 1 {
                0.0
            } else if x > 0 && x < w - 1 {
                lab_dist(row[x - 1], row[x + 1])
            } else if x == 0 {
                lab_dist(row[0], row[1])
            } else {
                lab_dist(row[w - 2], row[w - 1])
            }
        };

        // First row: M(x, 0) = C_U(x, 0)
        for (x, c) in cost[..cur_w].iter_mut().enumerate() {
            *c = compute_c_u(&rows[0], x, cur_w);
        }

        for y in 1..h {
            for x in 0..cur_w {
                let c_u = compute_c_u(&rows[y], x, cur_w);

                // C_L(x, y) = C_U(x, y) + D[(x, y-1), (x-1, y)]
                let c_l = if x > 0 {
                    c_u + lab_dist(rows[y - 1][x], rows[y][x - 1])
                } else {
                    f32::MAX / 2.0
                };

                // C_R(x, y) = C_U(x, y) + D[(x, y-1), (x+1, y)]
                let c_r = if x < cur_w - 1 {
                    c_u + lab_dist(rows[y - 1][x], rows[y][x + 1])
                } else {
                    f32::MAX / 2.0
                };

                // M(x, y) = min(M(x-1,y-1)+C_L, M(x,y-1)+C_U, M(x+1,y-1)+C_R)
                let from_left = if x > 0 {
                    cost[(y - 1) * cur_w + x - 1] + c_l
                } else {
                    f32::MAX / 2.0
                };
                let from_above = cost[(y - 1) * cur_w + x] + c_u;
                let from_right = if x < cur_w - 1 {
                    cost[(y - 1) * cur_w + x + 1] + c_r
                } else {
                    f32::MAX / 2.0
                };

                let min = from_above.min(from_left).min(from_right);
                cost[y * cur_w + x] = min;

                // Store parent direction for exact traceback
                if min == from_above {
                    parent[y * cur_w + x] = 0;
                } else if min == from_left {
                    parent[y * cur_w + x] = -1;
                } else {
                    parent[y * cur_w + x] = 1;
                }
            }
        }

        // Find minimum cost in last row
        let last_row_start = (h - 1) * cur_w;
        let mut min_x = 0;
        let mut min_cost = cost[last_row_start];
        for x in 1..cur_w {
            if cost[last_row_start + x] < min_cost {
                min_cost = cost[last_row_start + x];
                min_x = x;
            }
        }

        // Trace seam back from bottom to top using parent pointers
        let mut seam = vec![0_usize; h];
        seam[h - 1] = min_x;
        for y in (1..h).rev() {
            let x = seam[y];
            let dir = parent[y * cur_w + x];
            seam[y - 1] = (x as isize + dir as isize) as usize;
        }

        // Record removal order and remove from working buffers
        for y in 0..h {
            let sx = seam[y];
            let orig_x = indices[y][sx];
            order[y * w + orig_x] = step;
            rows[y].remove(sx);
            indices[y].remove(sx);
        }

        cur_w -= 1;
    }

    // Surviving pixels (last one per row) get order = w
    for y in 0..h {
        for &orig_x in &indices[y] {
            order[y * w + orig_x] = w as u32;
        }
    }

    order
}

/// Render a seam-carved image at any target size using the precomputed order map.
/// direction: 0 = vertical seams (target_size is width), 1 = horizontal seams (target_size is height)
#[wasm_bindgen]
pub fn render_seam_carved(
    image_data: &[u8],
    order: &[u32],
    orig_width: u32,
    orig_height: u32,
    target_size: u32,
    direction: u32,
) -> Vec<u8> {
    let w = orig_width as usize;
    let h = orig_height as usize;
    let target = target_size as usize;

    if direction == 0 {
        // Reducing width: keep pixels where order > seams_removed
        let seams_removed = w - target;
        let mut output = vec![0u8; target * h * 4];
        let mut out_idx = 0;

        for y in 0..h {
            for x in 0..w {
                if order[y * w + x] as usize > seams_removed {
                    let src = (y * w + x) * 4;
                    output[out_idx..out_idx + 4].copy_from_slice(&image_data[src..src + 4]);
                    out_idx += 4;
                }
            }
        }

        output
    } else {
        // Reducing height: keep pixels where order > seams_removed, per column
        let seams_removed = h - target;

        // Build surviving row indices per column
        let mut col_survivors: Vec<Vec<usize>> = vec![Vec::with_capacity(target); w];
        for x in 0..w {
            for y in 0..h {
                if order[y * w + x] as usize > seams_removed {
                    col_survivors[x].push(y);
                }
            }
        }

        // Output in row-major order
        let mut output = vec![0u8; w * target * 4];
        for (r, row) in output.chunks_exact_mut(w * 4).enumerate().take(target) {
            for (x, survivors) in col_survivors.iter().enumerate().take(w) {
                let orig_y = survivors[r];
                let src = (orig_y * w + x) * 4;
                let dst = x * 4;
                row[dst..dst + 4].copy_from_slice(&image_data[src..src + 4]);
            }
        }

        output
    }
}

/// Euclidean distance between two scaled OKLab colors
fn lab_dist(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    let dl = a.0 - b.0;
    let da = a.1 - b.1;
    let db = a.2 - b.2;
    (dl * dl + da * da + db * db).sqrt()
}

/// Encode an RGBA byte buffer as a PNG with the Display P3 ICC profile
/// embedded via an iCCP chunk so that color-managed viewers render the
/// pixels in the wide gamut they were authored in. Pixels are written
/// as-is — no color conversion happens here, since the buffer is already
/// in the target working space.
#[wasm_bindgen]
pub fn encode_png(image_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    // Display P3 ICC profile shipped with macOS. Copied verbatim from
    // /System/Library/ColorSync/Profiles/Display P3.icc.
    const P3_ICC: &[u8] = include_bytes!("display_p3.icc");

    // Build the iCCP chunk content per PNG spec:
    //   profile name (Latin-1, 1..=79 bytes) + 0x00 + compression method (0)
    //   + zlib-compressed profile.
    let mut iccp = Vec::with_capacity(P3_ICC.len() + 16);
    iccp.extend_from_slice(b"Display P3");
    iccp.push(0); // null terminator after the profile name
    iccp.push(0); // compression method: zlib
    {
        use flate2::{write::ZlibEncoder, Compression};
        use std::io::Write;
        let mut z = ZlibEncoder::new(&mut iccp, Compression::default());
        z.write_all(P3_ICC)
            .map_err(|e| JsValue::from_str(&format!("zlib: {e}")))?;
        z.finish()
            .map_err(|e| JsValue::from_str(&format!("zlib finish: {e}")))?;
    }

    let mut buf: Vec<u8> = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|e| JsValue::from_str(&format!("png header: {e}")))?;
        // iCCP must come before IDAT — write_chunk inserts at the current
        // position, which after write_header is right before the first IDAT.
        writer
            .write_chunk(png::chunk::iCCP, &iccp)
            .map_err(|e| JsValue::from_str(&format!("png iccp chunk: {e}")))?;
        writer
            .write_image_data(image_data)
            .map_err(|e| JsValue::from_str(&format!("png data: {e}")))?;
    }
    Ok(buf)
}

/// Linearly remap LAB lightness and chroma in a single pass.
///
/// - Lightness: input L in [0, 100] is mapped linearly to [l_min, l_max].
///   Defaults of (0, 100) are identity. (100, 0) inverts. (avg, avg) flattens.
/// - Chroma: input chroma magnitude (sqrt(a² + b²)) is rescaled by a factor
///   that linearly interpolates between c_min/100 (at chroma=0) and c_max/100
///   (at chroma=128, the practical sRGB max). Defaults of (0, 100) are identity
///   in the sense that c_max=100 means "100% of original" — the c_min knob lets
///   you bias the rescale toward desaturating darker chromas independently.
///   To desaturate fully use (0, 0); to boost use (0, 200); etc.
/// Build a "carve LUT" — for each output pixel after seam carving, the
/// linear index into the original source buffer that should appear there.
/// Used by the WebGPU rendering pipeline so the carve compute shader can
/// do an indirect lookup instead of recomputing surviving pixels.
///
/// Output layout is row-major over the carved dimensions:
///   length = target_w * target_h
///   for direction == 0 (reduce width):  target_w = target_size, target_h = orig_h
///   for direction == 1 (reduce height): target_w = orig_w, target_h = target_size
#[wasm_bindgen]
pub fn build_carve_lut(
    order: &[u32],
    orig_width: u32,
    orig_height: u32,
    target_size: u32,
    direction: u32,
) -> Vec<u32> {
    let w = orig_width as usize;
    let h = orig_height as usize;
    let target = target_size as usize;

    if direction == 0 {
        // Reducing width: keep pixels where order > seams_removed.
        let seams_removed = w - target;
        let mut lut = Vec::with_capacity(target * h);
        for y in 0..h {
            for x in 0..w {
                if order[y * w + x] as usize > seams_removed {
                    lut.push((y * w + x) as u32);
                }
            }
        }
        lut
    } else {
        // Reducing height. Walk each column, keep surviving rows in order,
        // then transpose into row-major output.
        let seams_removed = h - target;
        let mut col_survivors: Vec<Vec<usize>> = vec![Vec::with_capacity(target); w];
        for x in 0..w {
            for y in 0..h {
                if order[y * w + x] as usize > seams_removed {
                    col_survivors[x].push(y);
                }
            }
        }
        let mut lut = vec![0u32; w * target];
        for r in 0..target {
            for x in 0..w {
                let orig_y = col_survivors[x][r];
                lut[r * w + x] = (orig_y * w + x) as u32;
            }
        }
        lut
    }
}

/// Build a 101-bin histogram of L* values across the image (one bin per
/// integer L from 0 to 100) and a 161-bin histogram of chroma magnitudes
/// (sqrt(a² + b²)) clamped into 0..160. Returned as one flat array of
/// length 262 (101 + 161); the L bins come first, then the C bins.
///
/// The chroma range goes to 160 instead of 128 because Display P3's
/// gamut reaches further into Lab chroma than sRGB does (the most
/// saturated P3 reds/greens land around c≈140–150).
#[wasm_bindgen]
pub fn compute_lc_histogram(image_data: &[u8]) -> Vec<u32> {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    let mut hist = vec![0u32; 101 + 161];
    for i in (0..image_data.len()).step_by(4) {
        let (l, a, b) = rgb_bytes_to_lab(image_data[i], image_data[i + 1], image_data[i + 2]);
        let l_bin = l.round().clamp(0.0, 100.0) as usize;
        let chroma = (a * a + b * b).sqrt();
        let c_bin = chroma.round().clamp(0.0, 160.0) as usize;
        hist[l_bin] += 1;
        hist[101 + c_bin] += 1;
    }
    hist
}

#[wasm_bindgen]
pub fn remap_lab(
    image_data: &[u8],
    _width: u32,
    _height: u32,
    l_min: f32,
    l_max: f32,
    c_min: f32,
    c_max: f32,
    hue_deg: f32,
) -> Vec<u8> {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    let l_scale = (l_max - l_min) / 100.0;
    let c_min_n = c_min / 100.0;
    let c_max_n = c_max / 100.0;
    let c_diff_n = c_max_n - c_min_n;
    // Reference chroma magnitude for "100% chroma" — chosen to match
    // Display P3's practical maximum in our scaled OKLab units.
    const C_REF: f32 = 160.0;

    let hue_rad = hue_deg.to_radians();
    let cos_h = hue_rad.cos();
    let sin_h = hue_rad.sin();

    let mut output = Vec::with_capacity(image_data.len());

    for i in (0..image_data.len()).step_by(4) {
        let (mut l, mut a, mut b) =
            rgb_bytes_to_lab(image_data[i], image_data[i + 1], image_data[i + 2]);
        let alpha = image_data[i + 3];

        // Lightness remap.
        l = l_min + l * l_scale;

        // Chroma remap — parallel to L: input chroma magnitude (treated
        // as 0..1 against C_REF) is mapped linearly to [c_min_n, c_max_n],
        // and a/b are scaled so the resulting magnitude matches.
        // Skip near-zero chromas where the hue direction is undefined.
        let chroma = (a * a + b * b).sqrt();
        if chroma > 0.5 {
            let chroma_norm = chroma / C_REF;
            let out_chroma_norm = c_min_n + c_diff_n * chroma_norm;
            let factor = (out_chroma_norm * C_REF) / chroma;
            a *= factor;
            b *= factor;
        }

        // Hue rotation — rotate the (a, b) vector by hue_deg around origin.
        if hue_deg != 0.0 {
            let new_a = a * cos_h - b * sin_h;
            let new_b = a * sin_h + b * cos_h;
            a = new_a;
            b = new_b;
        }

        let (r, g, b8) = lab_to_rgb_bytes(l, a, b);
        output.push(r);
        output.push(g);
        output.push(b8);
        output.push(alpha);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pixel(r: u8, g: u8, b: u8) -> [u8; 4] {
        [r, g, b, 255]
    }

    #[test]
    fn test_seam_order_each_row_is_permutation() {
        // 6x4 image: red background with a green vertical stripe at x=3
        let w = 6;
        let h = 4;
        let mut data = Vec::with_capacity(w * h * 4);
        for _y in 0..h {
            for x in 0..w {
                let px = if x == 3 {
                    make_pixel(0, 255, 0)
                } else {
                    make_pixel(255, 0, 0)
                };
                data.extend_from_slice(&px);
            }
        }

        let order = precompute_vertical_order(&data, w, h);

        // Each row should have order values that form a permutation of 1..=w
        for y in 0..h {
            let mut row_orders: Vec<u32> = (0..w).map(|x| order[y * w + x]).collect();
            row_orders.sort();
            assert_eq!(
                row_orders,
                vec![1, 2, 3, 4, 5, 6],
                "Row {y}: expected permutation of 1..=6, got {row_orders:?}"
            );
        }
    }

    #[test]
    fn test_seam_order_not_just_left_to_right() {
        // Image with a gradient and a bright object in the middle
        // Left side: smooth dark gradient. Center: bright spot. Right: smooth dark gradient.
        // The seam carving should avoid the bright center and remove from the dark edges.
        let w = 10;
        let h = 6;
        let mut data = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                // Create a "landscape" with important center content
                let dist_from_center = ((x as f32 - 4.5).abs() + (y as f32 - 2.5).abs()) / 7.0;
                let brightness = (255.0 * (1.0 - dist_from_center).max(0.0)) as u8;
                // Add some variation based on position to prevent uniform regions
                let r = brightness.saturating_add((x * 17 % 30) as u8);
                let g = brightness.saturating_add((y * 23 % 20) as u8);
                let b = brightness.saturating_add(((x + y) * 13 % 25) as u8);
                data.extend_from_slice(&make_pixel(r, g, b));
            }
        }

        let order = precompute_vertical_order(&data, w, h);

        // Check that the order is not trivially sequential for each row
        for y in 0..h {
            let row_orders: Vec<u32> = (0..w).map(|x| order[y * w + x]).collect();
            let sequential: Vec<u32> = (1..=w as u32).collect();
            assert_ne!(
                row_orders, sequential,
                "Row {y}: order should not be trivially left-to-right sequential, got {row_orders:?}"
            );
            let reverse_sequential: Vec<u32> = (1..=w as u32).rev().collect();
            assert_ne!(
                row_orders, reverse_sequential,
                "Row {y}: order should not be trivially right-to-left sequential"
            );
        }
    }

    #[test]
    fn test_render_preserves_correct_pixels() {
        // 4x2 image with distinct pixels
        let w = 4;
        let h = 2;
        let data: Vec<u8> = vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255, // row 0
            128, 0, 0, 255, 0, 128, 0, 255, 0, 0, 128, 255, 128, 128, 0, 255, // row 1
        ];

        let order = precompute_vertical_order(&data, w, h);

        // Render at target=3 (remove 1 seam)
        let result = render_seam_carved(&data, &order, 4, 2, 3, 0);
        assert_eq!(result.len(), 3 * 2 * 4, "Output should be 3x2 pixels");

        // Render at target=1 (remove 3 seams)
        let result = render_seam_carved(&data, &order, 4, 2, 1, 0);
        assert_eq!(result.len(), 1 * 2 * 4, "Output should be 1x2 pixels");

        // Render at original size (remove 0 seams) should be identical to input
        let result = render_seam_carved(&data, &order, 4, 2, 4, 0);
        assert_eq!(
            result, data,
            "Rendering at original size should produce identical output"
        );
    }
}
