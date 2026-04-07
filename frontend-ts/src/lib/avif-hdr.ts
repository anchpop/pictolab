// Thin wrapper around our vendored fork of the jSquash AVIF encoder
// (in vendor/avif-enc/enc/avif_enc.js). Forked solely to expose libavif's
// CICP fields so we can write nclx-tagged HDR AVIFs (BT.2020 + PQ).
//
// The vendored module is a single-threaded build — multi-threaded needs
// SharedArrayBuffer which requires COOP/COEP headers we don't set.

// CICP code points (ITU-T H.273)
export const CP_BT709 = 1;
export const CP_BT2020 = 9;
export const CP_DISPLAY_P3 = 12;

export const TC_SRGB = 13;
export const TC_PQ = 16; // SMPTE ST 2084
export const TC_HLG = 18;

export const MC_IDENTITY = 0;
export const MC_BT601 = 6;
export const MC_BT2020_NCL = 9;

export interface EncodeOptions {
  quality: number;
  qualityAlpha: number;
  tileRowsLog2: number;
  tileColsLog2: number;
  speed: number;
  subsample: number;
  chromaDeltaQ: boolean;
  sharpness: number;
  tune: number;
  denoiseLevel: number;
  enableSharpYUV: boolean;
  bitDepth: 8 | 10 | 12;
  cicpColorPrimaries: number;
  cicpTransferCharacteristics: number;
  cicpMatrixCoefficients: number;
  fullRange: boolean;
}

const defaultOptions: EncodeOptions = {
  quality: 60,
  qualityAlpha: -1,
  tileRowsLog2: 0,
  tileColsLog2: 0,
  speed: 6,
  subsample: 1,
  chromaDeltaQ: false,
  sharpness: 0,
  tune: 0,
  denoiseLevel: 0,
  enableSharpYUV: false,
  bitDepth: 8,
  cicpColorPrimaries: -1,
  cicpTransferCharacteristics: -1,
  cicpMatrixCoefficients: -1,
  fullRange: false,
};

// Import the wasm as a fingerprinted asset URL so vite emits it into
// dist/. The emscripten glue's auto-locate logic doesn't survive the
// bundler reshuffle, so we override locateFile() to point at the URL
// vite gives us.
import avifWasmUrl from '../../vendor/avif-enc/enc/avif_enc.wasm?url';

let modulePromise: Promise<any> | null = null;

async function getModule() {
  if (!modulePromise) {
    const mod: any = await import('../../vendor/avif-enc/enc/avif_enc.js' as any);
    modulePromise = mod.default({
      locateFile: (path: string) =>
        path.endsWith('.wasm') ? avifWasmUrl : path,
    });
  }
  return modulePromise;
}

export async function encodeAvif(
  pixels: Uint8Array | Uint16Array,
  width: number,
  height: number,
  opts: Partial<EncodeOptions> = {}
): Promise<Uint8Array> {
  const options = { ...defaultOptions, ...opts };
  const module = await getModule();
  // The C++ binding takes a std::string built from a Uint8Array view of
  // the underlying buffer — works equally well for u8 and u16 RGBA.
  const u8 = new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength);
  const out: Uint8Array | null = module.encode(u8, width, height, options);
  if (!out) throw new Error('AVIF encode failed');
  // Copy out of the module's heap so we can return after the encoder
  // releases the memory.
  return new Uint8Array(out);
}
