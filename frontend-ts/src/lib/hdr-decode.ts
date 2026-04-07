// HDR-aware image decoder. Sniffs container magic bytes and routes
// HDR-capable formats (AVIF, Ultra HDR JPEG) through our vendored
// libavif / libultrahdr emscripten builds, returning linear extended
// Display P3 half-float RGBA pixels. SDR fallback (PNG / regular JPEG)
// is the caller's responsibility — this module returns null for them.

import avifWasmUrl from '../../vendor/avif-enc/dec/avif_dec.wasm?url';
import uhdrWasmUrl from '../../vendor/uhdr-enc/uhdr_enc.wasm?url';

// CICP (ITU-T H.273) transfer characteristics codes we care about.
const TC_BT709 = 1;
const TC_SRGB = 13;
const TC_PQ = 16; // SMPTE ST 2084
const TC_HLG = 18;

// CICP color primaries.
const CP_BT709 = 1;
const CP_BT2020 = 9;
const CP_DISPLAY_P3 = 12;

export interface HdrDecodeResult {
  // True for real HDR (above-1 / wide-gamut). False means we decoded
  // through a wasm path that produced quantized SDR — Editor should use
  // gpu_set_source for these.
  hdr: boolean;
  // Linear extended Display P3, half-float RGBA, tightly packed
  // (8 bytes/pixel). Only present for hdr=true sources.
  pixels: Uint8Array | null;
  // 8-bit Display P3 RGBA proxy for the seam worker + histogram. Always
  // populated. For hdr=true this is a tonemapped quantization of the f16
  // pixels; for hdr=false it's the actual decoded SDR pixels.
  sdrPixels: Uint8Array;
  width: number;
  height: number;
}

let avifModulePromise: Promise<any> | null = null;
async function getAvifModule() {
  if (!avifModulePromise) {
    const mod: any = await import('../../vendor/avif-enc/dec/avif_dec.js' as any);
    avifModulePromise = mod.default({
      locateFile: (path: string) =>
        path.endsWith('.wasm') ? avifWasmUrl : path,
    });
  }
  return avifModulePromise;
}

let uhdrModulePromise: Promise<any> | null = null;
async function getUhdrModule() {
  if (!uhdrModulePromise) {
    const mod: any = await import('../../vendor/uhdr-enc/uhdr_enc.js' as any);
    uhdrModulePromise = mod.default({
      locateFile: (path: string) =>
        path.endsWith('.wasm') ? uhdrWasmUrl : path,
    });
  }
  return uhdrModulePromise;
}

// Sniff: AVIF files start with `....ftypavif` / `....ftypavis`.
function isAvif(bytes: Uint8Array): boolean {
  if (bytes.length < 12) return false;
  if (bytes[4] !== 0x66 || bytes[5] !== 0x74 || bytes[6] !== 0x79 || bytes[7] !== 0x70) {
    return false; // not "ftyp"
  }
  const brand = String.fromCharCode(bytes[8], bytes[9], bytes[10], bytes[11]);
  return brand === 'avif' || brand === 'avis';
}

// Cheap JPEG sniff. We let libultrahdr's `isUltraHdr` decide whether
// it's a *gain-mapped* one — regular JPEGs return false.
function isJpeg(bytes: Uint8Array): boolean {
  return bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff;
}

// f32 → IEEE 754 half (round-to-nearest-even, with infinity/NaN handling).
function f32ToF16(val: number): number {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = val;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  const frac = x & 0x7fffff;
  if (exp <= 0) {
    if (exp < -10) return sign;
    const f = (frac | 0x800000) >>> (1 - exp);
    return sign | ((f + 0x1000) >>> 13);
  }
  if (exp >= 31) {
    if (((x >>> 23) & 0xff) === 0xff) {
      return sign | 0x7c00 | (frac ? 1 : 0);
    }
    return sign | 0x7c00;
  }
  return sign | (exp << 10) | ((frac + 0x1000) >>> 13);
}

// Quantize a linear extended Display P3 f32 RGBA buffer down to 8-bit
// Display P3 RGBA. Highlights >1 are softened with a Reinhard-ish curve
// so the seam-energy / histogram passes still see meaningful structure
// in the bright areas instead of a flat clip.
function quantizeToSdrP3(linF32: Float32Array): Uint8Array {
  const n = linF32.length / 4;
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    let r = Math.max(linF32[i * 4], 0);
    let g = Math.max(linF32[i * 4 + 1], 0);
    let b = Math.max(linF32[i * 4 + 2], 0);
    // Simple per-channel Reinhard tonemap: x / (1 + x).
    r = r / (1 + r);
    g = g / (1 + g);
    b = b / (1 + b);
    // Compensate for the SDR midtone shift the tonemap introduces by
    // rescaling so 0.5 (where most SDR content sits) maps back near 0.5.
    const k = 1 / 0.5; // 0.5 -> 0.333 after Reinhard, ×2 -> 0.666; close enough
    r = Math.min(r * k, 1);
    g = Math.min(g * k, 1);
    b = Math.min(b * k, 1);
    // sRGB / Display P3 OETF.
    const enc = (x: number) =>
      x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
    out[i * 4] = Math.round(enc(r) * 255);
    out[i * 4 + 1] = Math.round(enc(g) * 255);
    out[i * 4 + 2] = Math.round(enc(b) * 255);
    out[i * 4 + 3] = Math.round(Math.min(Math.max(linF32[i * 4 + 3], 0), 1) * 255);
  }
  return out;
}

// Pack a Float32Array RGBA buffer into the f16 layout the WASM expects.
function packF16(f32: Float32Array): Uint8Array {
  const out = new Uint8Array(f32.length * 2);
  for (let i = 0; i < f32.length; i++) {
    const h = f32ToF16(f32[i]);
    out[i * 2] = h & 0xff;
    out[i * 2 + 1] = h >>> 8;
  }
  return out;
}

// PQ inverse EOTF (SMPTE ST 2084): non-linear [0,1] → linear cd/m² / 10000
function pqEotf(e: number): number {
  const m1 = 0.1593017578125;
  const m2 = 78.84375;
  const c1 = 0.8359375;
  const c2 = 18.8515625;
  const c3 = 18.6875;
  const ep = Math.pow(Math.max(e, 0), 1 / m2);
  const num = Math.max(ep - c1, 0);
  const den = c2 - c3 * ep;
  return Math.pow(num / den, 1 / m1);
}

// HLG inverse OETF (BT.2100). Returns scene-linear in [0, 12].
function hlgInverseOetf(e: number): number {
  const a = 0.17883277;
  const b = 0.28466892;
  const c = 0.55991073;
  if (e <= 0.5) return (e * e) / 3;
  return (Math.exp((e - c) / a) + b) / 12;
}

// sRGB / Display-P3 inverse OETF (same transfer function).
function srgbInverse(e: number): number {
  const s = Math.sign(e);
  const a = Math.abs(e);
  if (a <= 0.04045) return s * a / 12.92;
  return s * Math.pow((a + 0.055) / 1.055, 2.4);
}

// BT.2020 → Display P3 (linear). Precomputed M_p3_to_xyz^-1 · M_xyz_from_2020.
const BT2020_TO_P3 = [
  1.34357825,  -0.28217967, -0.06139858,
  -0.06529745, 1.07578791,  -0.01049046,
  0.00282179,  -0.01959849, 1.01677670,
];

function applyMatrix3(m: number[], r: number, g: number, b: number): [number, number, number] {
  return [
    m[0] * r + m[1] * g + m[2] * b,
    m[3] * r + m[4] * g + m[5] * b,
    m[6] * r + m[7] * g + m[8] * b,
  ];
}

// Convert decoded AVIF u16 RGBA pixels into linear extended Display P3 f32,
// honoring the source CICP. depth is 10 or 12.
function avifToLinearP3(
  rgba: Uint16Array,
  width: number,
  height: number,
  depth: number,
  primaries: number,
  transfer: number
): Float32Array {
  const max = (1 << depth) - 1;
  const n = width * height;
  const out = new Float32Array(n * 4);

  let toLinear: (e: number) => number;
  if (transfer === TC_PQ) {
    // 203 nits = SDR reference white per BT.2408.
    const refWhite = 203 / 10000;
    toLinear = (e: number) => pqEotf(e) / refWhite;
  } else if (transfer === TC_HLG) {
    toLinear = hlgInverseOetf;
  } else if (transfer === TC_SRGB || transfer === TC_BT709) {
    toLinear = srgbInverse;
  } else {
    toLinear = (e: number) => e; // assume already linear
  }

  const wide = primaries === CP_BT2020;

  for (let i = 0; i < n; i++) {
    let r = toLinear(rgba[i * 4] / max);
    let g = toLinear(rgba[i * 4 + 1] / max);
    let b = toLinear(rgba[i * 4 + 2] / max);
    if (wide) {
      [r, g, b] = applyMatrix3(BT2020_TO_P3, r, g, b);
    }
    out[i * 4] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = rgba[i * 4 + 3] / max;
  }
  return out;
}

// Decode an AVIF buffer through libavif. Falls back to the SDR caller
// when the file is plain BT.709 sRGB (no need for our HDR pipeline).
async function decodeAvif(buf: Uint8Array): Promise<HdrDecodeResult | null> {
  const mod = await getAvifModule();
  // Always request 12 bits — libavif will upsample 8-bit content for
  // free, and 10/12-bit HDR survives unchanged.
  const result: any = mod.decode(buf, 12);
  if (!result) return null;
  const transfer: number = result.transferCharacteristics;
  const primaries: number = result.colorPrimaries;
  const sourceDepth: number = result.sourceDepth ?? 8;
  // Plain SDR? Let the caller take the cheap canvas path.
  if (
    sourceDepth === 8 &&
    (transfer === TC_SRGB || transfer === TC_BT709 || transfer === 2 /* unspec */) &&
    (primaries === CP_BT709 || primaries === CP_DISPLAY_P3 || primaries === 2)
  ) {
    return null;
  }
  const f32 = avifToLinearP3(
    result.data,
    result.width,
    result.height,
    12,
    primaries,
    transfer
  );
  return {
    hdr: true,
    pixels: packF16(f32),
    sdrPixels: quantizeToSdrP3(f32),
    width: result.width,
    height: result.height,
  };
}

// Decode an Ultra HDR JPEG through libultrahdr. Returns null if the JPEG
// isn't gain-mapped (libultrahdr's `isUltraHdr` says no), so the caller
// can fall back to the regular JPEG path.
async function decodeUhdr(buf: Uint8Array): Promise<HdrDecodeResult | null> {
  const mod = await getUhdrModule();
  if (!mod.isUltraHdr(buf)) return null;
  const result: any = mod.decode(buf);
  if (!result) return null;
  // libultrahdr returned linear half-float RGBA in P3 already — exactly
  // what gpu_set_source_linear_f16 wants. Build a quantized SDR proxy
  // from the same f16 data for the seam worker / histogram path.
  const f16 = result.data as Uint8Array;
  const n = result.width * result.height;
  const f32 = new Float32Array(n * 4);
  for (let i = 0; i < n * 4; i++) {
    const lo = f16[i * 2];
    const hi = f16[i * 2 + 1];
    // Inline f16→f32 (mirror frontend-rs/src/gpu.rs::f16_to_f32).
    const h = lo | (hi << 8);
    const sign = (h & 0x8000) >>> 15;
    const exp = (h & 0x7c00) >>> 10;
    const frac = h & 0x03ff;
    let v: number;
    if (exp === 0) v = (frac / 1024) * Math.pow(2, -14);
    else if (exp === 31) v = frac ? NaN : Infinity;
    else v = (1 + frac / 1024) * Math.pow(2, exp - 15);
    f32[i] = sign ? -v : v;
  }
  return {
    hdr: true,
    pixels: f16,
    sdrPixels: quantizeToSdrP3(f32),
    width: result.width,
    height: result.height,
  };
}

// HEIC magic-byte sniff. The container is ISOBMFF; first 8 bytes are
// the box length followed by "ftyp", then a 4-char major brand. iPhone
// HEICs use one of `heic`, `heix`, `mif1`, `msf1`, `heim`, `heis`, `hevc`.
function isHeic(bytes: Uint8Array): boolean {
  if (bytes.length < 12) return false;
  if (bytes[4] !== 0x66 || bytes[5] !== 0x74 || bytes[6] !== 0x79 || bytes[7] !== 0x70) {
    return false;
  }
  const brand = String.fromCharCode(bytes[8], bytes[9], bytes[10], bytes[11]);
  return (
    brand === 'heic' ||
    brand === 'heix' ||
    brand === 'mif1' ||
    brand === 'msf1' ||
    brand === 'heim' ||
    brand === 'heis' ||
    brand === 'hevc' ||
    brand === 'hevx'
  );
}

let libheifPromise: Promise<any> | null = null;
async function getLibheif() {
  if (!libheifPromise) {
    // The wasm-bundle variant inlines the .wasm as base64 inside the JS,
    // so vite doesn't need to know about a separate wasm asset.
    libheifPromise = import('libheif-js/wasm-bundle' as any).then((m) => m.default ?? m);
  }
  return libheifPromise;
}

// libheif enum constants we care about (mirror the C header).
const HEIF_COLORSPACE_RGB = 1;
const HEIF_CHROMA_INTERLEAVED_RGBA = 11;
const HEIF_CHROMA_INTERLEAVED_RRGGBBAA_LE = 15;
const HEIF_CHANNEL_INTERLEAVED = 10;

// Drop into libheif's C API to pull HDR HEIC pixels at full bit depth.
// `image.display(...)` only ever gives 8-bit sRGB tonemapped output, so
// for iPhone HDR HEICs we have to do this dance ourselves: ask libheif
// for 16-bit interleaved RGBA + the source nclx CICP, then run the same
// PQ/HLG inverse + BT.2020→P3 conversion as the AVIF path.
async function decodeHeicViaCApi(buf: Uint8Array): Promise<HdrDecodeResult | null> {
  const libheif: any = await getLibheif();
  const M: any = libheif.Module ?? libheif;
  if (typeof M._heif_context_alloc !== 'function') return null;

  const readU32 = (ptr: number) => M.HEAPU32[ptr >>> 2];
  const errOk = (errPtr: number) => readU32(errPtr) === 0;

  // All malloc'd handles to free in finally.
  let inPtr = 0,
    errPtr = 0,
    handlePtrPtr = 0,
    nclxPtrPtr = 0,
    imgPtrPtr = 0,
    stridePtr = 0,
    optsPtr = 0,
    ctx = 0,
    handle = 0,
    img = 0;

  try {
    inPtr = M._malloc(buf.length);
    if (!inPtr) return null;
    M.HEAPU8.set(buf, inPtr);

    ctx = M._heif_context_alloc();
    if (!ctx) return null;

    errPtr = M._malloc(12); // sizeof(heif_error)

    M._heif_context_read_from_memory_without_copy(errPtr, ctx, inPtr, buf.length, 0);
    if (!errOk(errPtr)) return null;

    handlePtrPtr = M._malloc(4);
    M._heif_context_get_primary_image_handle(errPtr, ctx, handlePtrPtr);
    if (!errOk(errPtr)) return null;
    handle = readU32(handlePtrPtr);
    if (!handle) return null;

    const lumaBits = M._heif_image_handle_get_luma_bits_per_pixel(handle);

    // Pull nclx CICP if present. Defaults match unspecified-but-likely-sRGB.
    let primaries = CP_BT709;
    let transfer = TC_SRGB;
    nclxPtrPtr = M._malloc(4);
    M.HEAPU32[nclxPtrPtr >>> 2] = 0;
    M._heif_image_handle_get_nclx_color_profile(errPtr, handle, nclxPtrPtr);
    if (errOk(errPtr)) {
      const nclxPtr = readU32(nclxPtrPtr);
      if (nclxPtr) {
        // struct heif_color_profile_nclx { uint8_t version; (3pad)
        //   int color_primaries;            // offset 4
        //   int transfer_characteristics;   // offset 8
        //   int matrix_coefficients;        // offset 12
        //   uint8_t full_range_flag; ... }
        primaries = readU32(nclxPtr + 4);
        transfer = readU32(nclxPtr + 8);
        M._heif_nclx_color_profile_free(nclxPtr);
      }
    }

    optsPtr = M._heif_decoding_options_alloc();
    // libheif's default for `convert_hdr_to_8bit` is false, which is
    // exactly what we want, so we don't poke at the struct.

    imgPtrPtr = M._malloc(4);
    const chroma =
      lumaBits > 8 ? HEIF_CHROMA_INTERLEAVED_RRGGBBAA_LE : HEIF_CHROMA_INTERLEAVED_RGBA;
    M._heif_decode_image(errPtr, handle, imgPtrPtr, HEIF_COLORSPACE_RGB, chroma, optsPtr);
    if (!errOk(errPtr)) return null;
    img = readU32(imgPtrPtr);
    if (!img) return null;

    const width = M._heif_image_get_width(img, HEIF_CHANNEL_INTERLEAVED);
    const height = M._heif_image_get_height(img, HEIF_CHANNEL_INTERLEAVED);
    const bppRange = M._heif_image_get_bits_per_pixel_range(img, HEIF_CHANNEL_INTERLEAVED);
    stridePtr = M._malloc(4);
    const planePtr = M._heif_image_get_plane_readonly(img, HEIF_CHANNEL_INTERLEAVED, stridePtr);
    if (!planePtr) return null;
    const stride = readU32(stridePtr);

    // Bit depth per channel — 8 for SDR, 10/12 for HDR. libheif aligns
    // 10/12-bit values into 16-bit slots, value lives in the low bits.
    const depth = bppRange > 0 ? bppRange : lumaBits;

    // Pull the pixels into a contiguous typed array. The source is row-
    // padded so we copy a row at a time.
    let f32: Float32Array;
    if (depth > 8) {
      const u16Row = width * 4;
      const u16 = new Uint16Array(width * height * 4);
      for (let y = 0; y < height; y++) {
        const srcOff = (planePtr + y * stride) >>> 1;
        u16.set(M.HEAPU16.subarray(srcOff, srcOff + u16Row), y * u16Row);
      }
      f32 = avifToLinearP3(u16, width, height, depth, primaries, transfer);
    } else {
      const u8 = new Uint8Array(width * height * 4);
      for (let y = 0; y < height; y++) {
        const srcOff = planePtr + y * stride;
        u8.set(M.HEAPU8.subarray(srcOff, srcOff + width * 4), y * width * 4);
      }
      // Reuse the linearizer by widening 8 → 16 with the right shift so
      // avifToLinearP3's `/max` math holds.
      const u16 = new Uint16Array(u8.length);
      for (let i = 0; i < u8.length; i++) u16[i] = u8[i] << 8;
      // Treat as 16-bit; primaries/transfer still apply.
      f32 = avifToLinearP3(u16, width, height, 16, primaries, transfer);
    }

    return {
      hdr: depth > 8,
      pixels: depth > 8 ? packF16(f32) : null,
      sdrPixels: quantizeToSdrP3(f32),
      width,
      height,
    };
  } catch (err) {
    console.warn('HEIC C-API decode failed:', err);
    return null;
  } finally {
    if (img) M._heif_image_release(img);
    if (handle) M._heif_image_handle_release(handle);
    if (ctx) M._heif_context_free(ctx);
    if (optsPtr) M._heif_decoding_options_free(optsPtr);
    if (inPtr) M._free(inPtr);
    if (errPtr) M._free(errPtr);
    if (handlePtrPtr) M._free(handlePtrPtr);
    if (nclxPtrPtr) M._free(nclxPtrPtr);
    if (imgPtrPtr) M._free(imgPtrPtr);
    if (stridePtr) M._free(stridePtr);
  }
}

// Last-resort SDR fallback that uses libheif's high-level display() API,
// which always tonemaps to 8-bit sRGB. We only land here if the C-API
// path fails for some reason (e.g. an HEIC variant libheif's high-level
// JS wrapper handles but the C API trips on).
async function decodeHeicViaDisplay(buf: Uint8Array): Promise<HdrDecodeResult | null> {
  const libheif: any = await getLibheif();
  const decoder = new libheif.HeifDecoder();
  const images = decoder.decode(buf);
  if (!images || images.length === 0) return null;
  const image = images[0];
  const width = image.get_width();
  const height = image.get_height();
  const sdrPixels = new Uint8ClampedArray(width * height * 4);
  await new Promise<void>((resolve, reject) => {
    image.display({ data: sdrPixels, width, height }, (out: any) => {
      if (!out) reject(new Error('libheif display() failed'));
      else resolve();
    });
  });
  return {
    hdr: false,
    pixels: null,
    sdrPixels: new Uint8Array(sdrPixels.buffer, sdrPixels.byteOffset, sdrPixels.byteLength),
    width,
    height,
  };
}

async function decodeHeic(buf: Uint8Array): Promise<HdrDecodeResult | null> {
  const viaC = await decodeHeicViaCApi(buf);
  if (viaC) return viaC;
  return decodeHeicViaDisplay(buf);
}

// Top-level dispatcher: try every HDR decoder we have. Returns null on
// SDR or on any failure so callers can fall through to their existing
// 8-bit path.
export async function decodeHdr(buf: Uint8Array): Promise<HdrDecodeResult | null> {
  try {
    if (isAvif(buf)) return await decodeAvif(buf);
    if (isJpeg(buf)) return await decodeUhdr(buf);
    if (isHeic(buf)) return await decodeHeic(buf);
  } catch (err) {
    console.warn('HDR decode failed, falling back to SDR:', err);
  }
  return null;
}
