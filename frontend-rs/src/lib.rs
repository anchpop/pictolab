mod utils;

use wasm_bindgen::prelude::*;
use palette::{IntoColor, Lab, Srgb};

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
