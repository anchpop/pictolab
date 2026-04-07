declare module 'frontend-rs' {
  export function compute_lc_histogram(image_data: Uint8Array): Uint32Array;

  export function encode_png(
    image_data: Uint8Array,
    width: number,
    height: number
  ): Uint8Array;

  export function remap_lab(
    image_data: Uint8Array,
    width: number,
    height: number,
    l_min: number,
    l_max: number,
    c_min: number,
    c_max: number,
    hue_deg: number
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

  export function precompute_seam_order_gpu(
    image_data: Uint8Array,
    width: number,
    height: number,
    direction: number
  ): Promise<Uint32Array>;

  export function gpu_init(canvas: HTMLCanvasElement): Promise<void>;

  export function gpu_set_source(
    image_data: Uint8Array,
    width: number,
    height: number
  ): void;

  export function gpu_set_source_linear_f16(
    f16_data: Uint8Array,
    width: number,
    height: number
  ): void;

  export function gpu_set_carve_lut(
    lut: Uint32Array,
    target_w: number,
    target_h: number
  ): void;

  export function gpu_set_squish_dims(target_w: number, target_h: number): void;

  export function gpu_render(
    l_min: number,
    l_max: number,
    c_min: number,
    c_max: number,
    hue_deg: number,
    show_hdr: number,
    time: number
  ): void;

  export function gpu_dispose(): void;

  export function gpu_readback_rgba8(): Promise<Uint8Array>;

  export function gpu_readback_hdr_pq_u16(depth: number): Promise<Uint16Array>;

  export function gpu_readback_linear_f16(): Promise<Uint8Array>;

  export function build_carve_lut(
    order: Uint32Array,
    orig_width: number,
    orig_height: number,
    target_size: number,
    direction: number
  ): Uint32Array;

  export function is_webgpu_available(): boolean;
}
