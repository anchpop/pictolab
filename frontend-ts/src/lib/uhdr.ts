// Thin wrapper around our vendored libultrahdr emscripten encoder
// (in vendor/uhdr-enc/). Takes a half-float RGBA image (linear extended
// Display P3 by default) and emits an Ultra HDR JPEG — a regular SDR
// JPEG with a gain-map JPEG embedded via MPF + Adobe gain-map XMP.

// Color gamut codes (mirror UHDR_CG_*).
export const CG_BT_709 = 0;
export const CG_DISPLAY_P3 = 1;
export const CG_BT_2100 = 2;

// Color transfer codes (mirror UHDR_CT_*).
export const CT_LINEAR = 0;
export const CT_HLG = 1;
export const CT_PQ = 2;
export const CT_SRGB = 3;

export interface EncodeOptions {
  baseQuality: number;
  gainMapQuality: number;
  colorGamut: number;
  colorTransfer: number;
  multiChannelGainmap: number;
}

const defaultOptions: EncodeOptions = {
  baseQuality: 90,
  gainMapQuality: 95,
  colorGamut: CG_DISPLAY_P3,
  colorTransfer: CT_LINEAR,
  multiChannelGainmap: 1,
};

let modulePromise: Promise<any> | null = null;

async function getModule() {
  if (!modulePromise) {
    const mod: any = await import(
      /* @vite-ignore */ '../../vendor/uhdr-enc/uhdr_enc.js' as any
    );
    modulePromise = mod.default();
  }
  return modulePromise;
}

// Encode raw f16 RGBA pixels (8 bytes/pixel) into an Ultra HDR JPEG.
export async function encodeUhdr(
  pixelsF16: Uint8Array,
  width: number,
  height: number,
  opts: Partial<EncodeOptions> = {}
): Promise<Uint8Array> {
  const options = { ...defaultOptions, ...opts };
  const module = await getModule();
  const out: Uint8Array | null = module.encode(
    pixelsF16,
    width,
    height,
    options
  );
  if (!out) throw new Error('Ultra HDR encode failed');
  return new Uint8Array(out);
}
