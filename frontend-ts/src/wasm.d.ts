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
}
