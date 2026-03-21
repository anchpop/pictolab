mod utils;

use wasm_bindgen::prelude::*;
use palette::{IntoColor, Lab, Srgb};

/// Seam carving with forward energy, computed in LAB color space.
/// Converts to LAB once, carves seams on the LAB buffer, converts back at the end.
#[wasm_bindgen]
pub fn seam_carve_lab(image_data: &[u8], width: u32, height: u32, seams_to_remove: u32) -> Vec<u8> {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    let w = width as usize;
    let h = height as usize;
    let seams = seams_to_remove as usize;

    // Convert entire image to LAB once
    let mut lab_img: Vec<Lab> = Vec::with_capacity(w * h);
    let mut alpha_img: Vec<u8> = Vec::with_capacity(w * h);

    for i in (0..image_data.len()).step_by(4) {
        let r = image_data[i] as f32 / 255.0;
        let g = image_data[i + 1] as f32 / 255.0;
        let b = image_data[i + 2] as f32 / 255.0;
        let rgb = Srgb::new(r, g, b);
        lab_img.push(rgb.into_color());
        alpha_img.push(image_data[i + 3]);
    }

    let mut cur_w = w;

    for _ in 0..seams.min(cur_w.saturating_sub(1)) {
        // Build forward energy cost matrix
        // Forward energy considers the cost of new edges created by removing a pixel
        let mut cost = vec![0.0_f32; cur_w * h];

        // First row: zero cost
        // (cost is already zero from initialization)

        for y in 1..h {
            for x in 0..cur_w {
                let idx = |yy: usize, xx: usize| -> &Lab {
                    &lab_img[yy * cur_w + xx]
                };

                // Cost of creating a new horizontal edge between (y, x-1) and (y, x+1)
                let c_u = if x > 0 && x < cur_w - 1 {
                    lab_dist(idx(y, x - 1), idx(y, x + 1))
                } else {
                    0.0
                };

                // Cost of creating edge between (y-1, x) and (y, x-1) -- for left path
                let c_l = if x > 0 {
                    c_u + lab_dist(idx(y - 1, x), idx(y, x - 1))
                } else {
                    f32::MAX / 2.0
                };

                // Cost of creating edge between (y-1, x) and (y, x+1) -- for right path
                let c_r = if x < cur_w - 1 {
                    c_u + lab_dist(idx(y - 1, x), idx(y, x + 1))
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

        // Trace seam back from bottom to top
        let mut seam = vec![0_usize; h];
        seam[h - 1] = min_x;
        for y in (1..h).rev() {
            let x = seam[y];
            let above = cost[(y - 1) * cur_w + x];
            let above_left = if x > 0 { cost[(y - 1) * cur_w + x - 1] } else { f32::MAX };

            // Retrace using the same forward energy logic
            if x > 0 {
                let c_u = if x > 0 && x < cur_w - 1 {
                    lab_dist(&lab_img[y * cur_w + x - 1], &lab_img[y * cur_w + x + 1])
                } else { 0.0 };
                let c_l = c_u + lab_dist(&lab_img[(y - 1) * cur_w + x], &lab_img[y * cur_w + x - 1]);
                if (above_left + c_l - cost[y * cur_w + x]).abs() < 1e-3 {
                    seam[y - 1] = x - 1;
                    continue;
                }
            }
            {
                let c_u = if x > 0 && x < cur_w - 1 {
                    lab_dist(&lab_img[y * cur_w + x - 1], &lab_img[y * cur_w + x + 1])
                } else { 0.0 };
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

        // Remove the seam from lab_img and alpha_img
        for y in (0..h).rev() {
            let row_start = y * cur_w;
            let remove_x = seam[y];
            lab_img.remove(row_start + remove_x);
            alpha_img.remove(row_start + remove_x);
        }

        cur_w -= 1;
    }

    // Convert LAB back to RGBA
    let new_width = cur_w;
    let mut output = Vec::with_capacity(new_width * h * 4);

    for y in 0..h {
        for x in 0..new_width {
            let lab = lab_img[y * new_width + x];
            let rgb_out: Srgb = lab.into_color();
            output.push((rgb_out.red.clamp(0.0, 1.0) * 255.0) as u8);
            output.push((rgb_out.green.clamp(0.0, 1.0) * 255.0) as u8);
            output.push((rgb_out.blue.clamp(0.0, 1.0) * 255.0) as u8);
            output.push(alpha_img[y * new_width + x]);
        }
    }

    output
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
