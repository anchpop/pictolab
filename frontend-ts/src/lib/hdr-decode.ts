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
  // Linear extended Display P3, half-float RGBA, tightly packed
  // (8 bytes/pixel). Suitable for gpu_set_source_linear_f16.
  pixels: Uint8Array;
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
  return { pixels: packF16(f32), width: result.width, height: result.height };
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
  // what gpu_set_source_linear_f16 wants.
  return {
    pixels: result.data as Uint8Array,
    width: result.width,
    height: result.height,
  };
}

// Top-level dispatcher: try every HDR decoder we have. Returns null on
// SDR or on any failure so callers can fall through to their existing
// 8-bit path.
export async function decodeHdr(buf: Uint8Array): Promise<HdrDecodeResult | null> {
  try {
    if (isAvif(buf)) return await decodeAvif(buf);
    if (isJpeg(buf)) return await decodeUhdr(buf);
  } catch (err) {
    console.warn('HDR decode failed, falling back to SDR:', err);
  }
  return null;
}
