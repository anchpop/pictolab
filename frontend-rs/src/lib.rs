mod utils;

use wasm_bindgen::prelude::*;
use palette::{IntoColor, Lab, Srgb};

/// Precompute seam removal order using forward energy in LAB color space.
/// Returns a u32 array of size width*height where each value is the step at which
/// that pixel gets removed (1-indexed). Pixels surviving to the end get value = max_dimension.
/// direction: 0 = vertical seams (reduce width), 1 = horizontal seams (reduce height)
#[wasm_bindgen]
pub fn precompute_seam_order(image_data: &[u8], width: u32, height: u32, direction: u32) -> Vec<u32> {
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
    let mut rows: Vec<Vec<Lab>> = Vec::with_capacity(h);
    for y in 0..h {
        let mut row = Vec::with_capacity(w);
        for x in 0..w {
            let i = (y * w + x) * 4;
            let r = image_data[i] as f32 / 255.0;
            let g = image_data[i + 1] as f32 / 255.0;
            let b = image_data[i + 2] as f32 / 255.0;
            row.push(Srgb::new(r, g, b).into_color());
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

        // Build forward energy cost matrix
        let mut cost = vec![0.0_f32; cur_w * h];

        for y in 1..h {
            for x in 0..cur_w {
                // Cost of creating a new horizontal edge between (y, x-1) and (y, x+1)
                let c_u = if x > 0 && x < cur_w - 1 {
                    lab_dist(&rows[y][x - 1], &rows[y][x + 1])
                } else {
                    0.0
                };

                // Cost of creating edge between (y-1, x) and (y, x-1) -- for left path
                let c_l = if x > 0 {
                    c_u + lab_dist(&rows[y - 1][x], &rows[y][x - 1])
                } else {
                    f32::MAX / 2.0
                };

                // Cost of creating edge between (y-1, x) and (y, x+1) -- for right path
                let c_r = if x < cur_w - 1 {
                    c_u + lab_dist(&rows[y - 1][x], &rows[y][x + 1])
                } else {
                    f32::MAX / 2.0
                };

                let above_left = if x > 0 { cost[(y - 1) * cur_w + x - 1] + c_l } else { f32::MAX / 2.0 };
                let above = cost[(y - 1) * cur_w + x] + c_u;
                let above_right = if x < cur_w - 1 { cost[(y - 1) * cur_w + x + 1] + c_r } else { f32::MAX / 2.0 };

                cost[y * cur_w + x] = above.min(above_left).min(above_right);
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

        // Trace seam back from bottom to top using forward energy
        let mut seam = vec![0_usize; h];
        seam[h - 1] = min_x;
        for y in (1..h).rev() {
            let x = seam[y];
            let above = cost[(y - 1) * cur_w + x];
            let above_left = if x > 0 { cost[(y - 1) * cur_w + x - 1] } else { f32::MAX };

            if x > 0 {
                let c_u = if x > 0 && x < cur_w - 1 {
                    lab_dist(&rows[y][x - 1], &rows[y][x + 1])
                } else {
                    0.0
                };
                let c_l = c_u + lab_dist(&rows[y - 1][x], &rows[y][x - 1]);
                if (above_left + c_l - cost[y * cur_w + x]).abs() < 1e-3 {
                    seam[y - 1] = x - 1;
                    continue;
                }
            }
            {
                let c_u = if x > 0 && x < cur_w - 1 {
                    lab_dist(&rows[y][x - 1], &rows[y][x + 1])
                } else {
                    0.0
                };
                if (above + c_u - cost[y * cur_w + x]).abs() < 1e-3 {
                    seam[y - 1] = x;
                    continue;
                }
            }
            if x < cur_w - 1 {
                seam[y - 1] = x + 1;
            } else {
                seam[y - 1] = x;
            }
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
        for r in 0..target {
            for x in 0..w {
                let orig_y = col_survivors[x][r];
                let src = (orig_y * w + x) * 4;
                let dst = (r * w + x) * 4;
                output[dst..dst + 4].copy_from_slice(&image_data[src..src + 4]);
            }
        }

        output
    }
}

/// Euclidean distance between two LAB colors
fn lab_dist(a: &Lab, b: &Lab) -> f32 {
    let dl = a.l - b.l;
    let da = a.a - b.a;
    let db = a.b - b.b;
    (dl * dl + da * da + db * db).sqrt()
}

#[wasm_bindgen]
pub fn invert_lightness_lab(image_data: &[u8], _width: u32, _height: u32) -> Vec<u8> {
    // Initialize panic hook for better error messages in console
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    let mut output = Vec::with_capacity(image_data.len());

    // Process each pixel
    for i in (0..image_data.len()).step_by(4) {
        let r = image_data[i] as f32 / 255.0;
        let g = image_data[i + 1] as f32 / 255.0;
        let b = image_data[i + 2] as f32 / 255.0;
        let a = image_data[i + 3];

        // Convert RGB to LAB
        let rgb = Srgb::new(r, g, b);
        let mut lab: Lab = rgb.into_color();

        // Invert the lightness channel (L ranges from 0 to 100)
        lab.l = 100.0 - lab.l;

        // Convert back to RGB
        let rgb_out: Srgb = lab.into_color();

        // Clamp and convert back to u8
        output.push((rgb_out.red.clamp(0.0, 1.0) * 255.0) as u8);
        output.push((rgb_out.green.clamp(0.0, 1.0) * 255.0) as u8);
        output.push((rgb_out.blue.clamp(0.0, 1.0) * 255.0) as u8);
        output.push(a); // Keep alpha unchanged
    }

    output
}

#[wasm_bindgen]
pub fn equalize_lightness_lab(image_data: &[u8], _width: u32, _height: u32) -> Vec<u8> {
    // Initialize panic hook for better error messages in console
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    // First pass: compute average lightness
    let mut total_lightness = 0.0;
    let pixel_count = image_data.len() / 4;

    for i in (0..image_data.len()).step_by(4) {
        let r = image_data[i] as f32 / 255.0;
        let g = image_data[i + 1] as f32 / 255.0;
        let b = image_data[i + 2] as f32 / 255.0;

        let rgb = Srgb::new(r, g, b);
        let lab: Lab = rgb.into_color();
        total_lightness += lab.l;
    }

    let avg_lightness = total_lightness / pixel_count as f32;

    // Second pass: set all pixels to average lightness
    let mut output = Vec::with_capacity(image_data.len());

    for i in (0..image_data.len()).step_by(4) {
        let r = image_data[i] as f32 / 255.0;
        let g = image_data[i + 1] as f32 / 255.0;
        let b = image_data[i + 2] as f32 / 255.0;
        let a = image_data[i + 3];

        // Convert RGB to LAB
        let rgb = Srgb::new(r, g, b);
        let mut lab: Lab = rgb.into_color();

        // Set lightness to average
        lab.l = avg_lightness;

        // Convert back to RGB
        let rgb_out: Srgb = lab.into_color();

        // Clamp and convert back to u8
        output.push((rgb_out.red.clamp(0.0, 1.0) * 255.0) as u8);
        output.push((rgb_out.green.clamp(0.0, 1.0) * 255.0) as u8);
        output.push((rgb_out.blue.clamp(0.0, 1.0) * 255.0) as u8);
        output.push(a); // Keep alpha unchanged
    }

    output
}

#[wasm_bindgen]
pub fn laberation(image_data: &[u8], width: u32, height: u32, offset: i32) -> Vec<u8> {
    // Initialize panic hook for better error messages in console
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    let width = width as usize;
    let height = height as usize;

    // First pass: convert all pixels to LAB and store in separate channel arrays
    let mut l_channel = vec![0.0_f32; width * height];
    let mut a_channel = vec![0.0_f32; width * height];
    let mut b_channel = vec![0.0_f32; width * height];
    let mut alpha_channel = vec![0_u8; width * height];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            let pixel_idx = y * width + x;

            let r = image_data[idx] as f32 / 255.0;
            let g = image_data[idx + 1] as f32 / 255.0;
            let b = image_data[idx + 2] as f32 / 255.0;

            let rgb = Srgb::new(r, g, b);
            let lab: Lab = rgb.into_color();

            l_channel[pixel_idx] = lab.l;
            a_channel[pixel_idx] = lab.a;
            b_channel[pixel_idx] = lab.b;
            alpha_channel[pixel_idx] = image_data[idx + 3];
        }
    }

    // Second pass: reconstruct image with offset channels
    let mut output = Vec::with_capacity(image_data.len());

    for y in 0..height {
        for x in 0..width {
            let pixel_idx = y * width + x;

            // Sample L channel offset directly down
            let l_y = ((y as i32 + offset).max(0).min(height as i32 - 1)) as usize;
            let l_idx = l_y * width + x;
            let l = l_channel[l_idx];

            // Sample A channel offset up and to the right
            let a_x = ((x as i32 + offset).max(0).min(width as i32 - 1)) as usize;
            let a_y = ((y as i32 - offset).max(0).min(height as i32 - 1)) as usize;
            let a_idx = a_y * width + a_x;
            let a = a_channel[a_idx];

            // Sample B channel offset up and to the left
            let b_x = ((x as i32 - offset).max(0).min(width as i32 - 1)) as usize;
            let b_y = ((y as i32 - offset).max(0).min(height as i32 - 1)) as usize;
            let b_idx = b_y * width + b_x;
            let b = b_channel[b_idx];

            // Reconstruct LAB color
            let lab = Lab::new(l, a, b);
            let rgb_out: Srgb = lab.into_color();

            // Clamp and convert back to u8
            output.push((rgb_out.red.clamp(0.0, 1.0) * 255.0) as u8);
            output.push((rgb_out.green.clamp(0.0, 1.0) * 255.0) as u8);
            output.push((rgb_out.blue.clamp(0.0, 1.0) * 255.0) as u8);
            output.push(alpha_channel[pixel_idx]);
        }
    }

    output
}

#[wasm_bindgen]
pub fn boost_chroma_lab(image_data: &[u8], _width: u32, _height: u32, factor: f32) -> Vec<u8> {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    let mut output = Vec::with_capacity(image_data.len());

    for i in (0..image_data.len()).step_by(4) {
        let r = image_data[i] as f32 / 255.0;
        let g = image_data[i + 1] as f32 / 255.0;
        let b = image_data[i + 2] as f32 / 255.0;
        let a = image_data[i + 3];

        let rgb = Srgb::new(r, g, b);
        let mut lab: Lab = rgb.into_color();

        // Scale the A and B channels to boost chroma
        lab.a *= factor;
        lab.b *= factor;

        let rgb_out: Srgb = lab.into_color();

        output.push((rgb_out.red.clamp(0.0, 1.0) * 255.0) as u8);
        output.push((rgb_out.green.clamp(0.0, 1.0) * 255.0) as u8);
        output.push((rgb_out.blue.clamp(0.0, 1.0) * 255.0) as u8);
        output.push(a);
    }

    output
}
