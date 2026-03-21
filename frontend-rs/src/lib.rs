mod utils;

use palette::{IntoColor, Lab, Srgb};
use wasm_bindgen::prelude::*;

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

        // Build forward energy cost matrix with parent pointers for traceback
        // Reference: https://avikdas.com/2019/07/29/improved-seam-carving-with-forward-energy.html
        let mut cost = vec![0.0_f32; cur_w * h];
        // Parent direction: -1=came from above-left, 0=above, 1=above-right
        let mut parent = vec![0i8; cur_w * h];

        // Helper: compute C_U(x, y) = D[(x-1, y), (x+1, y)]
        // At boundaries, replace out-of-bounds pixel with current pixel
        let compute_c_u = |row: &[Lab], x: usize, w: usize| -> f32 {
            if w <= 1 {
                0.0
            } else if x > 0 && x < w - 1 {
                lab_dist(&row[x - 1], &row[x + 1])
            } else if x == 0 {
                lab_dist(&row[0], &row[1])
            } else {
                lab_dist(&row[w - 2], &row[w - 1])
            }
        };

        // First row: M(x, 0) = C_U(x, 0)
        for x in 0..cur_w {
            cost[x] = compute_c_u(&rows[0], x, cur_w);
        }

        for y in 1..h {
            for x in 0..cur_w {
                let c_u = compute_c_u(&rows[y], x, cur_w);

                // C_L(x, y) = C_U(x, y) + D[(x, y-1), (x-1, y)]
                let c_l = if x > 0 {
                    c_u + lab_dist(&rows[y - 1][x], &rows[y][x - 1])
                } else {
                    f32::MAX / 2.0
                };

                // C_R(x, y) = C_U(x, y) + D[(x, y-1), (x+1, y)]
                let c_r = if x < cur_w - 1 {
                    c_u + lab_dist(&rows[y - 1][x], &rows[y][x + 1])
                } else {
                    f32::MAX / 2.0
                };

                // M(x, y) = min(M(x-1,y-1)+C_L, M(x,y-1)+C_U, M(x+1,y-1)+C_R)
                let from_left = if x > 0 { cost[(y - 1) * cur_w + x - 1] + c_l } else { f32::MAX / 2.0 };
                let from_above = cost[(y - 1) * cur_w + x] + c_u;
                let from_right = if x < cur_w - 1 { cost[(y - 1) * cur_w + x + 1] + c_r } else { f32::MAX / 2.0 };

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
