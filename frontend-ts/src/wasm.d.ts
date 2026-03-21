declare module 'frontend-rs' {
  export function invert_lightness_lab(
    image_data: Uint8Array,
    width: number,
    height: number
  ): Uint8Array;

  export function equalize_lightness_lab(
    image_data: Uint8Array,
    width: number,
    height: number
  ): Uint8Array;

  export function laberation(
    image_data: Uint8Array,
    width: number,
    height: number,
    offset: number
  ): Uint8Array;

  export function precompute_seam_order(
    image_data: Uint8Array,
    width: number,
    height: number,
    direction: number
  ): Uint32Array;

  export function render_seam_carved(
    image_data: Uint8Array,
    order: Uint32Array,
    orig_width: number,
    orig_height: number,
    target_size: number,
    direction: number
  ): Uint8Array;

  export function boost_chroma_lab(
    image_data: Uint8Array,
    width: number,
    height: number,
    factor: number
  ): Uint8Array;
}
