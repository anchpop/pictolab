import { useEffect, useRef, useState } from 'react';
import { ArrowLeftRight, Crop as CropIcon, Download, Eye, ImagePlus, Loader2, Minus, Plus, RotateCcw, RotateCw, X } from 'lucide-react';
import ImageDropZone from '@/components/ImageDropZone';
import { Button } from '@/components/ui/button';
import { CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { DualSlider } from '@/components/ui/dual-slider';
import { Segmented } from '@/components/ui/segmented';
import { Collapsible } from '@/components/ui/collapsible';
import { cn } from '@/lib/utils';

type Direction = 'width' | 'height';
type ResizeMode = 'squish' | 'carve';
type AspectRatio = 'free' | 'lock' | '1:1' | '4:3' | '3:4' | '16:9' | '9:16';

interface SourceState {
  url: string;
  // For SDR sources this is 8-bit RGBA in Display P3. For HDR sources
  // (`hdr: true`) it is tightly packed half-float RGBA in linear extended
  // Display P3 (8 bytes/pixel), ready for gpu_set_source_linear_f16.
  data: Uint8Array;
  // 8-bit Display P3 RGBA proxy used by the seam worker + histogram. For
  // SDR sources this is the same buffer as `data`; for HDR sources it's
  // a tonemapped quantization of the f16 pixels.
  sdrData: Uint8Array;
  w: number;
  h: number;
  hdr: boolean;
  // True if any pixel has alpha < 255. Used to warn the user that JPEG
  // export silently drops the alpha channel.
  hasAlpha: boolean;
}

type CropRect = { x: number; y: number; w: number; h: number };

type OutputTransform = {
  active: boolean;
  m: [number, number, number, number, number, number];
  dstW: number;
  dstH: number;
};

type Point = { x: number; y: number };

// Scan an 8-bit RGBA buffer for any non-opaque pixel. Cheap O(n) walk.
function bufferHasAlpha(rgba: Uint8Array): boolean {
  for (let i = 3; i < rgba.length; i += 4) {
    if (rgba[i] !== 255) return true;
  }
  return false;
}

type ExifOrientation = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;

function normalizeOrientation(value: unknown): ExifOrientation {
  const n = Number(value);
  return Number.isInteger(n) && n >= 2 && n <= 8 ? (n as ExifOrientation) : 1;
}

function getOrientedSize(width: number, height: number, orientation: ExifOrientation) {
  return orientation >= 5 ? { width: height, height: width } : { width, height };
}

function mapOrientedCoords(
  x: number,
  y: number,
  width: number,
  height: number,
  orientation: ExifOrientation
): [number, number] {
  switch (orientation) {
    case 2: return [width - 1 - x, y];
    case 3: return [width - 1 - x, height - 1 - y];
    case 4: return [x, height - 1 - y];
    case 5: return [y, x];
    case 6: return [height - 1 - y, x];
    case 7: return [height - 1 - y, width - 1 - x];
    case 8: return [y, width - 1 - x];
    default: return [x, y];
  }
}

function orientPackedRgba(
  data: Uint8Array,
  width: number,
  height: number,
  bytesPerPixel: number,
  orientation: ExifOrientation
): { data: Uint8Array; width: number; height: number } {
  if (orientation === 1) return { data, width, height };
  const outSize = getOrientedSize(width, height, orientation);
  const out = new Uint8Array(outSize.width * outSize.height * bytesPerPixel);

  // Fast paths via typed-array element copies (one indexed assignment per
  // pixel instead of subarray + set, ~5x faster on a 12MP photo). bpp=4 →
  // Uint32 (one element per pixel); bpp=8 → two Uint32 copies per pixel.
  // Skip when the source byteOffset isn't 4-aligned: a subarray()-backed
  // Uint8Array can land on an odd offset, and aliasing it as Uint32Array
  // would throw. The generic byte loop below handles that case correctly.
  if ((bytesPerPixel === 4 || bytesPerPixel === 8) && data.byteOffset % 4 === 0) {
    // Vertical flip is just reversed-row-order; pure subarray copies.
    if (orientation === 4) {
      const rowBytes = width * bytesPerPixel;
      for (let y = 0; y < height; y++) {
        out.set(
          data.subarray(y * rowBytes, (y + 1) * rowBytes),
          (height - 1 - y) * rowBytes
        );
      }
      return { data: out, width: outSize.width, height: outSize.height };
    }

    if (bytesPerPixel === 4) {
      const src32 = new Uint32Array(data.buffer, data.byteOffset, width * height);
      const dst32 = new Uint32Array(out.buffer);
      const dw = outSize.width;
      for (let y = 0; y < height; y++) {
        const srcRow = y * width;
        for (let x = 0; x < width; x++) {
          const [dx, dy] = mapOrientedCoords(x, y, width, height, orientation);
          dst32[dy * dw + dx] = src32[srcRow + x];
        }
      }
      return { data: out, width: outSize.width, height: outSize.height };
    }

    // bytesPerPixel === 8: two Uint32s per pixel.
    const src32 = new Uint32Array(data.buffer, data.byteOffset, width * height * 2);
    const dst32 = new Uint32Array(out.buffer);
    const dw = outSize.width;
    for (let y = 0; y < height; y++) {
      const srcRow = y * width;
      for (let x = 0; x < width; x++) {
        const [dx, dy] = mapOrientedCoords(x, y, width, height, orientation);
        const sIdx = (srcRow + x) * 2;
        const dIdx = (dy * dw + dx) * 2;
        dst32[dIdx] = src32[sIdx];
        dst32[dIdx + 1] = src32[sIdx + 1];
      }
    }
    return { data: out, width: outSize.width, height: outSize.height };
  }

  // Generic fallback (any unusual bpp).
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [dx, dy] = mapOrientedCoords(x, y, width, height, orientation);
      const srcOff = (y * width + x) * bytesPerPixel;
      const dstOff = (dy * outSize.width + dx) * bytesPerPixel;
      out.set(data.subarray(srcOff, srcOff + bytesPerPixel), dstOff);
    }
  }
  return {
    data: out,
    width: outSize.width,
    height: outSize.height,
  };
}

type JpegMetadata = {
  orientation: ExifOrientation;
  storedSize: { width: number; height: number } | null;
};

function parseExifOrientationFromApp1(
  buf: Uint8Array,
  segmentStart: number,
  segmentEnd: number
): ExifOrientation | null {
  if (segmentStart + 14 > segmentEnd) return null;
  if (
    buf[segmentStart] !== 0x45 || buf[segmentStart + 1] !== 0x78 ||
    buf[segmentStart + 2] !== 0x69 || buf[segmentStart + 3] !== 0x66 ||
    buf[segmentStart + 4] !== 0x00 || buf[segmentStart + 5] !== 0x00
  ) {
    return null;
  }

  const tiff = segmentStart + 6;
  const byteOrderA = buf[tiff];
  const byteOrderB = buf[tiff + 1];
  const little = byteOrderA === 0x49 && byteOrderB === 0x49;
  const big = byteOrderA === 0x4d && byteOrderB === 0x4d;
  if (!little && !big) return null;

  const read16 = (offset: number): number | null => {
    if (offset + 1 >= segmentEnd) return null;
    return little
      ? (buf[offset] | (buf[offset + 1] << 8))
      : ((buf[offset] << 8) | buf[offset + 1]);
  };

  const read32 = (offset: number): number | null => {
    if (offset + 3 >= segmentEnd) return null;
    return little
      ? (
        buf[offset] |
        (buf[offset + 1] << 8) |
        (buf[offset + 2] << 16) |
        (buf[offset + 3] << 24)
      ) >>> 0
      : (
        (buf[offset] << 24) |
        (buf[offset + 1] << 16) |
        (buf[offset + 2] << 8) |
        buf[offset + 3]
      ) >>> 0;
  };

  if (read16(tiff + 2) !== 42) return null;
  const ifdOffset = read32(tiff + 4);
  if (ifdOffset === null) return null;
  const ifd0 = tiff + ifdOffset;
  const entryCount = read16(ifd0);
  if (entryCount === null) return null;

  for (let j = 0; j < entryCount; j++) {
    const entry = ifd0 + 2 + j * 12;
    if (entry + 11 >= segmentEnd) return null;
    const tag = read16(entry);
    const type = read16(entry + 2);
    const count = read32(entry + 4);
    if (tag === 0x0112 && type === 3 && count === 1) {
      const value = read16(entry + 8);
      return value === null ? null : normalizeOrientation(value);
    }
  }
  return null;
}

// Scan the JPEG metadata once to get both the raw stored SOF dims and the
// EXIF Orientation tag. We own this parse instead of delegating to exifr so
// the critical JPEG upload path behaves the same across browsers.
function scanJpegMetadata(buf: Uint8Array): JpegMetadata | null {
  if (buf.length < 4 || buf[0] !== 0xff || buf[1] !== 0xd8) return null;
  let i = 2;
  let orientation: ExifOrientation = 1;
  let storedSize: { width: number; height: number } | null = null;

  while (i < buf.length) {
    while (i < buf.length && buf[i] === 0xff) i++;
    if (i >= buf.length) break;
    const marker = buf[i++];

    // Start-of-scan switches into entropy-coded data where segment parsing no
    // longer applies. EOI means there is nothing left to inspect.
    if (marker === 0xda || marker === 0xd9) break;
    // TEM and restart markers have no payload length.
    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
    if (i + 1 >= buf.length) break;

    const size = (buf[i] << 8) | buf[i + 1];
    if (size < 2 || i + size > buf.length) break;
    const segmentStart = i + 2;
    const segmentEnd = i + size;

    if (
      storedSize === null &&
      marker >= 0xc0 && marker <= 0xcf &&
      marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc &&
      segmentStart + 4 < segmentEnd
    ) {
      storedSize = {
        height: (buf[segmentStart + 1] << 8) | buf[segmentStart + 2],
        width: (buf[segmentStart + 3] << 8) | buf[segmentStart + 4],
      };
    } else if (marker === 0xe1 && orientation === 1) {
      const parsed = parseExifOrientationFromApp1(buf, segmentStart, segmentEnd);
      if (parsed !== null) orientation = parsed;
    }

    i = segmentEnd;
  }

  return { orientation, storedSize };
}

// We only apply manual orientation in our HDR path for JPEGs, where we own
// the EXIF parse. Non-JPEG formats rely on their decoders/browser path to
// honor orientation internally, so there is no extra post-decode rotation
// to apply. libultrahdr — our JPEG HDR decoder — returns raw stored pixels
// (verified empirically), so we apply the full EXIF orientation unchanged.
function pickHdrManualRotation(jpegMeta: JpegMetadata | null): ExifOrientation {
  return jpegMeta ? jpegMeta.orientation : 1;
}

function orientHdrDecodeResult<T extends {
  hdr: boolean;
  pixels: Uint8Array | null;
  sdrPixels: Uint8Array;
  width: number;
  height: number;
}>(decoded: T, orientation: ExifOrientation): T {
  if (orientation === 1) return decoded;
  const sdr = orientPackedRgba(decoded.sdrPixels, decoded.width, decoded.height, 4, orientation);
  if (decoded.hdr && decoded.pixels) {
    const hdr = orientPackedRgba(decoded.pixels, decoded.width, decoded.height, 8, orientation);
    return {
      ...decoded,
      pixels: hdr.data,
      sdrPixels: sdr.data,
      width: sdr.width,
      height: sdr.height,
    };
  }
  return {
    ...decoded,
    sdrPixels: sdr.data,
    width: sdr.width,
    height: sdr.height,
  };
}

// Safari iOS caps navigator.hardwareConcurrency at 2 regardless of actual
// core count, so this catches all iPhones (and any genuinely low-core
// device). Used to defer carve slider state updates until release.
const LOW_CORE_DEVICE =
  typeof navigator !== 'undefined' && navigator.hardwareConcurrency <= 2;
const GPU_RECOVERY_DELAY_MS = 250;

const ASPECT_OPTIONS: { value: AspectRatio; label: string }[] = [
  { value: 'free', label: 'Free' },
  { value: 'lock', label: 'Lock' },
  { value: '1:1', label: '1:1' },
  { value: '4:3', label: '4:3' },
  { value: '3:4', label: '3:4' },
  { value: '16:9', label: '16:9' },
  { value: '9:16', label: '9:16' },
];

// Numeric ratio (W/H) for a named aspect, or null for "free".
function aspectRatioValue(ar: AspectRatio, src: SourceState): number | null {
  switch (ar) {
    case 'free': return null;
    case 'lock': return src.w / src.h;
    case '1:1': return 1;
    case '4:3': return 4 / 3;
    case '3:4': return 3 / 4;
    case '16:9': return 16 / 9;
    case '9:16': return 9 / 16;
  }
}

// Photo-app-style angle slider: a strip of tick marks that the user
// clicks/drags through, with a number bubble over the active tick.
// Drag is captured pointer-style so it survives leaving the strip
// while held.
function TickSlider({
  value,
  min,
  max,
  onChange,
  disabled = false,
  tickStep = 1,
  majorTickStep = 5,
  snapStep = 0.1,
  trackClassName,
  activeTickClassName = 'bg-amber-500',
  activeMarkerClassName = 'bg-amber-500',
  title = 'Drag to adjust · double-click to reset',
  resetTitle = 'Reset to 0°',
  formatValue = (v: number) => `${v.toFixed(1)}°`,
}: {
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
  disabled?: boolean;
  tickStep?: number;
  majorTickStep?: number;
  snapStep?: number;
  trackClassName?: string;
  activeTickClassName?: string;
  activeMarkerClassName?: string;
  title?: string;
  resetTitle?: string;
  formatValue?: (v: number) => string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const setFromPointer = (clientX: number) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const f = (clientX - rect.left) / rect.width;
    const clamped = Math.max(0, Math.min(1, f));
    const raw = min + clamped * (max - min);
    const snapped = Math.round(raw / snapStep) * snapStep;
    onChange(snapped);
  };

  const ticks: { v: number; major: boolean }[] = [];
  for (let v = min; v <= max; v += tickStep) {
    const rounded = Math.round(v * 1000) / 1000;
    const major = Math.abs(rounded / majorTickStep - Math.round(rounded / majorTickStep)) < 1e-6;
    ticks.push({ v: rounded, major });
  }

  const fraction = Math.max(0, Math.min(1, (value - min) / (max - min)));

  return (
    <div
      className={
        'flex flex-1 flex-col items-center gap-1 select-none ' +
        (disabled ? 'opacity-50' : '')
      }
    >
      {/* Number bubble over the current value */}
      <div className="relative h-5 w-full">
        <div
          className="absolute -translate-x-1/2 rounded-full border border-border bg-background px-2 py-0.5 font-mono text-[10px] text-foreground shadow-sm"
          style={{ left: `${fraction * 100}%` }}
        >
          {formatValue(value)}
        </div>
      </div>
      {/* Tick strip */}
      <div
        ref={ref}
        className={cn(
          'relative h-7 w-full touch-none rounded-md bg-zinc-900 px-2',
          trackClassName,
          disabled ? 'cursor-default' : 'cursor-pointer'
        )}
        onPointerDown={(e) => {
          if (disabled) return;
          dragging.current = true;
          (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
          setFromPointer(e.clientX);
        }}
        onPointerMove={(e) => {
          if (disabled) return;
          if (!dragging.current) return;
          setFromPointer(e.clientX);
        }}
        onPointerUp={(e) => {
          if (disabled) return;
          dragging.current = false;
          (e.currentTarget as HTMLDivElement).releasePointerCapture(e.pointerId);
        }}
        onDoubleClick={() => {
          if (!disabled) onChange(0);
        }}
        title={title}
      >
        {/* Center reference dot (sits at value = midpoint) */}
        <div
          className="absolute top-0 h-1 w-1 -translate-x-1/2 rounded-full bg-muted-foreground/60"
          style={{ left: '50%' }}
        />
        {ticks.map((t) => {
          const f = (t.v - min) / (max - min);
          // Highlight ticks between center and current value.
          const lit =
            (value >= 0 && t.v >= 0 && t.v <= value) ||
            (value <= 0 && t.v <= 0 && t.v >= value);
          return (
            <div
              key={t.v}
              className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
              style={{ left: `${f * 100}%` }}
            >
              <div
                className={cn(
                  lit
                    ? activeTickClassName
                    : t.major
                    ? 'bg-zinc-300'
                    : 'bg-zinc-500',
                  t.major ? 'h-4 w-px' : 'h-2 w-px'
                )}
              />
            </div>
          );
        })}
        {/* Active position marker */}
        <div
          className={cn('absolute top-0 bottom-0 w-px', activeMarkerClassName)}
          style={{ left: `${fraction * 100}%` }}
        />
      </div>
      {/* Reset to 0° */}
      <button
        type="button"
        onClick={() => onChange(0)}
        disabled={disabled || value === 0}
        title={resetTitle}
        className="rounded-full p-1 text-muted-foreground hover:text-foreground disabled:cursor-default disabled:opacity-30"
      >
        <X className="h-3 w-3" />
      </button>
    </div>
  );
}

function normalizeQuarterTurns(rotate90: number): number {
  return ((rotate90 % 4) + 4) % 4;
}

function clampCropRect(crop: CropRect | null, targetW: number, targetH: number): CropRect | null {
  if (!crop) return null;
  const x = Math.max(0, Math.min(targetW - 1, crop.x));
  const y = Math.max(0, Math.min(targetH - 1, crop.y));
  const maxW = Math.max(1, targetW - x);
  const maxH = Math.max(1, targetH - y);
  const w = Math.max(1, Math.min(maxW, crop.w));
  const h = Math.max(1, Math.min(maxH, crop.h));
  if (x <= 0 && y <= 0 && w >= targetW && h >= targetH) return null;
  return { x, y, w, h };
}

function scaleCropRect(
  crop: CropRect | null,
  prevW: number,
  prevH: number,
  nextW: number,
  nextH: number
): CropRect | null {
  if (!crop || prevW <= 0 || prevH <= 0) return crop;
  const sx = nextW / prevW;
  const sy = nextH / prevH;
  return clampCropRect(
    {
      x: crop.x * sx,
      y: crop.y * sy,
      w: crop.w * sx,
      h: crop.h * sy,
    },
    nextW,
    nextH
  );
}

function scalePoint(
  point: Point | null,
  prevW: number,
  prevH: number,
  nextW: number,
  nextH: number
): Point | null {
  if (!point || prevW <= 0 || prevH <= 0) return point;
  return {
    x: point.x * (nextW / prevW),
    y: point.y * (nextH / prevH),
  };
}

function mapPreviewPointToSource(transform: OutputTransform, point: Point): Point {
  const [m00, m01, m10, m11, t0, t1] = transform.m;
  return {
    x: m00 * point.x + m01 * point.y + t0,
    y: m10 * point.x + m11 * point.y + t1,
  };
}

function mapSourcePointToPreview(transform: OutputTransform, point: Point): Point {
  const [m00, m01, m10, m11, t0, t1] = transform.m;
  const sx = point.x - t0;
  const sy = point.y - t1;
  return {
    x: m00 * sx + m10 * sy,
    y: m01 * sx + m11 * sy,
  };
}

function recenterCropRect(
  crop: CropRect | null,
  previewTransform: OutputTransform,
  anchor: Point | null
): CropRect | null {
  if (!crop || !anchor) return crop;
  const center = mapSourcePointToPreview(previewTransform, anchor);
  return clampCropRect(
    {
      x: center.x - crop.w / 2,
      y: center.y - crop.h / 2,
      w: crop.w,
      h: crop.h,
    },
    previewTransform.dstW,
    previewTransform.dstH
  );
}

function resolveRotationTransform(args: {
  targetW: number;
  targetH: number;
  straightenDeg: number;
  rotate90: number;
}): OutputTransform {
  const { targetW, targetH, straightenDeg, rotate90 } = args;
  const cx = targetW / 2;
  const cy = targetH / 2;

  const turns = normalizeQuarterTurns(rotate90);
  const theta = ((turns * 90 + straightenDeg) * Math.PI) / 180;
  const swap = turns % 2 === 1;
  const baseW = swap ? targetH : targetW;
  const baseH = swap ? targetW : targetH;

  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);

  // Solve for the largest scale factor s ∈ (0, 1] such that the four
  // dst corners, mapped to src via R(θ), all stay inside the source.
  let s = 1;
  for (const sgnX of [-1, 1]) {
    for (const sgnY of [-1, 1]) {
      const dx = (sgnX * baseW) / 2;
      const dy = (sgnY * baseH) / 2;
      const ox = cosT * dx - sinT * dy;
      const oy = sinT * dx + cosT * dy;
      if (ox > 1e-9) s = Math.min(s, (targetW - cx) / ox);
      else if (ox < -1e-9) s = Math.min(s, -cx / ox);
      if (oy > 1e-9) s = Math.min(s, (targetH - cy) / oy);
      else if (oy < -1e-9) s = Math.min(s, -cy / oy);
    }
  }
  s = Math.max(0, s);

  const dstW = Math.max(1, Math.round(baseW * s));
  const dstH = Math.max(1, Math.round(baseH * s));

  // Forward dst→src: src = C + R(θ) · (dst - dstCenter)
  //   sx = cx + cosT·(dx - dstW/2) - sinT·(dy - dstH/2)
  //   sy = cy + sinT·(dx - dstW/2) + cosT·(dy - dstH/2)
  const m00 = cosT;
  const m01 = -sinT;
  const m10 = sinT;
  const m11 = cosT;
  const t0 = cx - cosT * (dstW / 2) + sinT * (dstH / 2);
  const t1 = cy - sinT * (dstW / 2) - cosT * (dstH / 2);

  const active = straightenDeg !== 0 || turns !== 0;
  return { active, m: [m00, m01, m10, m11, t0, t1], dstW, dstH };
}

function applyCropToTransform(base: OutputTransform, crop: CropRect | null): OutputTransform {
  const clampedCrop = clampCropRect(crop, base.dstW, base.dstH);
  if (!clampedCrop) return base;
  const [m00, m01, m10, m11, t0, t1] = base.m;
  return {
    active: true,
    m: [
      m00,
      m01,
      m10,
      m11,
      m00 * clampedCrop.x + m01 * clampedCrop.y + t0,
      m10 * clampedCrop.x + m11 * clampedCrop.y + t1,
    ],
    dstW: Math.max(1, Math.round(clampedCrop.w)),
    dstH: Math.max(1, Math.round(clampedCrop.h)),
  };
}

// Resolve the final GPU output transform for the current crop + rotation
// controls. Crop lives in the rotated/straightened preview space and is
// applied as the final stage after the base rotation transform.
function resolveOutputTransform(args: {
  targetW: number;
  targetH: number;
  crop: CropRect | null;
  straightenDeg: number;
  rotate90: number;
}): OutputTransform {
  const base = resolveRotationTransform(args);
  return applyCropToTransform(base, args.crop);
}

// Largest (W, H) ≤ (src.w, src.h) that satisfies the ratio. Used for the
// initial snap when an aspect is selected; subsequent slider drags
// re-derive the other dim from the dragged one.
function snapSquish(ratio: number, src: SourceState): [number, number] {
  const srcRatio = src.w / src.h;
  if (ratio >= srcRatio) {
    return [src.w, Math.round(src.w / ratio)];
  }
  return [Math.round(src.h * ratio), src.h];
}

// Carve can only change one dimension, so for a given direction the ratio
// must satisfy r ≤ srcRatio (direction=width — vertical seams shrink the
// width) or r ≥ srcRatio (direction=height — horizontal seams shrink the
// height). Returns null when the ratio isn't directly reachable; the
// caller should hide the button rather than offer an inverted alias,
// since the inverted ratio has its own button anyway.
function snapCarve(
  ratio: number,
  src: SourceState,
  dir: Direction
): { w: number; h: number } | null {
  const srcRatio = src.w / src.h;
  if (dir === 'width' && ratio <= srcRatio) {
    return { w: Math.max(1, Math.round(src.h * ratio)), h: src.h };
  }
  if (dir === 'height' && ratio >= srcRatio) {
    return { w: src.w, h: Math.max(1, Math.round(src.w / ratio)) };
  }
  return null;
}

function Editor() {
  const [source, setSource] = useState<SourceState | null>(null);
  const [direction, setDirection] = useState<Direction>('width');
  const [resizeMode, setResizeMode] = useState<ResizeMode>('carve');
  const [aspect, setAspect] = useState<AspectRatio>('free');
  const [targetW, setTargetW] = useState(0);
  const [targetH, setTargetH] = useState(0);
  const [isPrecomputing, setIsPrecomputing] = useState(false);
  const [ready, setReady] = useState(false);
  const [lRange, setLRange] = useState<[number, number]>([0, 100]);
  const [cRange, setCRange] = useState<[number, number]>([0, 100]);
  const [hue, setHue] = useState(0);
  const [showHdr, setShowHdr] = useState(false);
  // null = still detecting, true/false = known result.
  const [gpuAvailable, setGpuAvailable] = useState<boolean | null>(null);
  const [hdrPresentationActive, setHdrPresentationActive] = useState<boolean | null>(null);
  const [resizeOpen, setResizeOpen] = useState(false);
  const [lightnessOpen, setLightnessOpen] = useState(false);
  const [chromaOpen, setChromaOpen] = useState(false);
  const [hueOpen, setHueOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(true);
  const [targetWDraft, setTargetWDraft] = useState<number | null>(null);
  const [targetHDraft, setTargetHDraft] = useState<number | null>(null);

  // Straighten + 90° rotation. Composed at the end of the pipeline by a
  // Lanczos rotation pass on the GPU.
  const [cropTool, setCropTool] = useState(false);
  const [cropRect, setCropRect] = useState<CropRect | null>(null);
  const [cropDrag, setCropDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const [straightenDeg, setStraightenDeg] = useState(0);
  const [rotate90, setRotate90] = useState(0);
  const straightenDegRef = useRef(0);
  const rotate90Ref = useRef(0);
  const cropRectRef = useRef<CropRect | null>(null);
  const cropAnchorRef = useRef<Point | null>(null);
  const cropToolRef = useRef(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const requestIdRef = useRef(0);
  // Tracks whether the persistent GPU context has been initialized against
  // the current canvas element.
  const gpuInitedRef = useRef(false);
  const gpuRecoveryRef = useRef<Promise<boolean> | null>(null);
  const [gpuLost, setGpuLost] = useState(false);
  // Ref mirror of gpuLost so recoverGpu (which is not a useCallback and
  // closes over the latest state via refs) can read it synchronously.
  const gpuLostRef = useRef(false);

  // Cached pieces of the pipeline.
  const wasmRef = useRef<typeof import('frontend-rs') | null>(null);
  const orderRef = useRef<Uint32Array | null>(null);
  // Direction the cached `orderRef` was computed for. Used to skip the
  // worker re-run when toggling resize mode without changing direction.
  const orderDirectionRef = useRef<Direction | null>(null);
  const resizeModeRef = useRef<ResizeMode>('carve');
  const aspectRef = useRef<AspectRatio>('free');
  const sourceRef = useRef<SourceState | null>(null);
  // Combined L (101 bins) + chroma (161 bins) histogram for the current
  // source. Used to display how many pixels would clip under the current
  // L/C remap settings.
  const lcHistRef = useRef<Uint32Array | null>(null);
  const directionRef = useRef<Direction>('width');
  const targetWRef = useRef(0);
  const targetHRef = useRef(0);
  const lRangeRef = useRef<[number, number]>([0, 100]);
  const cRangeRef = useRef<[number, number]>([0, 100]);
  const hueRef = useRef(0);
  const showHdrRef = useRef(false);
  const hdrRafRef = useRef<number | null>(null);

  // Mirror straighten/rotate90/crop state into refs and re-render. The
  // refs are what `render()` reads, so any change to these inputs has
  // to land in the refs *before* we kick the GPU.
  useEffect(() => {
    straightenDegRef.current = straightenDeg;
    rotate90Ref.current = rotate90;
    cropRectRef.current = cropRect;
    cropToolRef.current = cropTool;
    if (sourceRef.current && ready) render();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [straightenDeg, rotate90, cropRect, cropTool, ready]);

  useEffect(() => {
    if (!cropRectRef.current || !cropAnchorRef.current) return;
    const previewTransform = resolveRotationTransform({
      targetW: targetWRef.current,
      targetH: targetHRef.current,
      straightenDeg,
      rotate90,
    });
    const nextCrop = recenterCropRect(cropRectRef.current, previewTransform, cropAnchorRef.current);
    if (!nextCrop) return;
    const prevCrop = cropRectRef.current;
    const changed =
      Math.abs(nextCrop.x - prevCrop.x) > 0.25 ||
      Math.abs(nextCrop.y - prevCrop.y) > 0.25 ||
      Math.abs(nextCrop.w - prevCrop.w) > 0.25 ||
      Math.abs(nextCrop.h - prevCrop.h) > 0.25;
    if (!changed) return;
    cropRectRef.current = nextCrop;
    setCropRect(nextCrop);
  }, [straightenDeg, rotate90]);

  useEffect(() => {
    if (!resizeOpen) return;
    ensureCarvePrecompute(() => {
      applyResize();
      render();
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resizeOpen, resizeMode, direction, source]);

  // Detect WebGPU + load wasm.
  useEffect(() => {
    (async () => {
      const wasm = await import('frontend-rs');
      wasmRef.current = wasm;
      let available = false;
      try {
        if (wasm.is_webgpu_available()) {
          const adapter = await navigator.gpu?.requestAdapter();
          if (adapter) available = true;
        }
      } catch {
        /* no gpu */
      }
      setGpuAvailable(available);
    })();
    return () => {
      workerRef.current?.terminate();
      try {
        wasmRef.current?.gpu_dispose();
      } catch {
        /* ignore */
      }
    };
  }, []);

  const getWorker = () => {
    if (!workerRef.current) {
      workerRef.current = new Worker(
        new URL('../workers/seam-worker.ts', import.meta.url),
        { type: 'module' }
      );
    }
    return workerRef.current;
  };

  // Push the current resize state into the GPU context. Branches on the
  // active resize mode: squish writes a Lanczos dims uniform; carve
  // requires a precomputed seam order and rebuilds + uploads the LUT.
  const applyResize = () => {
    const wasm = wasmRef.current;
    const src = sourceRef.current;
    if (!wasm || !src) return;
    const tw = targetWRef.current;
    const th = targetHRef.current;
    if (resizeModeRef.current === 'carve') {
      const order = orderRef.current;
      const dir = directionRef.current;
      // The seam order is direction-specific. If we're mid-direction-
      // switch and the precompute hasn't landed yet, bail rather than
      // build a LUT against the wrong axis (which produces garbage).
      if (!order || orderDirectionRef.current !== dir) return;
      const dirNum = dir === 'width' ? 0 : 1;
      const target = dir === 'width' ? tw : th;
      const lut = wasm.build_carve_lut(order, src.w, src.h, target, dirNum);
      wasm.gpu_set_carve_lut(lut, tw, th);
    } else {
      wasm.gpu_set_squish_dims(tw, th);
    }
  };

  const uploadSourceToGpu = (
    wasm: typeof import('frontend-rs'),
    src: SourceState
  ) => {
    if (src.hdr) {
      wasm.gpu_set_source_linear_f16(src.data, src.w, src.h);
    } else {
      wasm.gpu_set_source(src.data, src.w, src.h);
    }
  };

  const configureGpuForSource = async (src: SourceState, preserveDims: boolean) => {
    const wasm = wasmRef.current;
    const c = canvasRef.current;
    if (!wasm || !c) return;

    c.width = src.w;
    c.height = src.h;

    if (!gpuInitedRef.current) {
      await wasm.gpu_init(c);
      gpuInitedRef.current = true;
    }
    setHdrPresentationActive(wasm.gpu_is_hdr_presentation_active());
    uploadSourceToGpu(wasm, src);

    const haveDims = preserveDims && targetWRef.current > 0 && targetHRef.current > 0;
    const nextW = haveDims ? targetWRef.current : src.w;
    const nextH = haveDims ? targetHRef.current : src.h;

    if (!haveDims) {
      targetWRef.current = nextW;
      targetHRef.current = nextH;
      setTargetW(nextW);
      setTargetH(nextH);
      setTargetWDraft(null);
      setTargetHDraft(null);
    }

    // Carve needs the seam order before it can build the LUT. If we're
    // recovering before that exists, fall back to a squish preview so the
    // canvas repaints instead of staying blank.
    if (
      resizeModeRef.current === 'carve' &&
      (!haveDims || !orderRef.current || orderDirectionRef.current !== directionRef.current)
    ) {
      wasm.gpu_set_squish_dims(nextW, nextH);
    } else {
      applyResize();
    }

    setReady(true);
    requestAnimationFrame(render);
  };

  const recoverGpu = async (reason: string, cause: unknown): Promise<boolean> => {
    if (gpuRecoveryRef.current) return gpuRecoveryRef.current;
    // Once we've given up, stop trying — the source-mount effect clears
    // gpuLost whenever the user loads a new image, giving the next attempt
    // a fresh start.
    if (gpuLostRef.current) return false;

    const job = (async () => {
      const wasm = wasmRef.current;
      const src = sourceRef.current;
      console.error(`GPU ${reason} failed; attempting recovery`, cause);
      if (!wasm || !src) return false;
      setReady(false);
      const resumeHdrLoop = showHdrRef.current;
      stopHdrLoop();

      // Give transient WebGPU/context hiccups a moment to settle so we don't
      // immediately thrash through repeated dispose/init cycles.
      await new Promise((resolve) => window.setTimeout(resolve, GPU_RECOVERY_DELAY_MS));

      try {
        wasm.gpu_dispose();
      } catch {
        /* ignore */
      }
      gpuInitedRef.current = false;

      try {
        await configureGpuForSource(src, true);
        if (resumeHdrLoop) startHdrLoop();
        return true;
      } catch (err) {
        // A single failed recovery is enough to surface the banner:
        // render() now bails cleanly after dispose, so nothing else will
        // retrigger recovery until the user loads a new source. Waiting
        // for repeated failures just leaves the editor silently frozen.
        console.error('GPU recovery failed:', err);
        gpuLostRef.current = true;
        setGpuLost(true);
        return false;
      } finally {
        gpuRecoveryRef.current = null;
      }
    })();

    gpuRecoveryRef.current = job;
    return job;
  };

  const syncGpuTransform = (): OutputTransform => {
    const wasm = wasmRef.current;
    const transform = resolveOutputTransform({
      targetW: targetWRef.current,
      targetH: targetHRef.current,
      crop: cropToolRef.current ? null : cropRectRef.current,
      straightenDeg: straightenDegRef.current,
      rotate90: rotate90Ref.current,
    });
    if (!wasm || !gpuInitedRef.current) return transform;

    if (transform.active) {
      try {
        wasm.gpu_set_rotation(
          transform.m[0],
          transform.m[1],
          transform.m[2],
          transform.m[3],
          transform.m[4],
          transform.m[5],
          transform.dstW,
          transform.dstH
        );
      } catch (err) {
        console.error('gpu_set_rotation failed:', err);
      }
    } else {
      try {
        wasm.gpu_clear_rotation();
      } catch {
        /* ignore — first render before output_tex exists */
      }
    }

    return transform;
  };

  // Issue a render with the current L/C/hue params. Cheap; safe to call
  // from any callback.
  const render = () => {
    const wasm = wasmRef.current;
    if (!wasm || !gpuInitedRef.current || gpuRecoveryRef.current) return;
    if (gpuLostRef.current) return;

    try {
      const tw = targetWRef.current;
      const th = targetHRef.current;
      const transform = syncGpuTransform();
      const showTransformedPreview = transform.active;

      // Sync the canvas attribute + CSS *before* the GPU submit. Setting
      // canvas.width/height clears the contents, so doing it after a
      // render would wipe the freshly-drawn frame and the next paint
      // wouldn't land until the next render() call.
      const c = canvasRef.current;
      const src = sourceRef.current;
      if (c && src) {
        if (showTransformedPreview) {
          // Transform active: canvas attribute matches the rotated/cropped dst
          // dims so the present pass blits 1:1; CSS fills the wrapper
          // (which has its aspect overridden to dstW/dstH).
          if (c.width !== transform.dstW) c.width = transform.dstW;
          if (c.height !== transform.dstH) c.height = transform.dstH;
          c.style.width = '100%';
          c.style.height = '100%';
        } else {
          if (c.width !== src.w) c.width = src.w;
          if (c.height !== src.h) c.height = src.h;
          c.style.width = `${(tw / src.w) * 100}%`;
          c.style.height = `${(th / src.h) * 100}%`;
        }
      }

      const [lMin, lMax] = lRangeRef.current;
      const [cMin, cMax] = cRangeRef.current;
      const t = performance.now() / 1000;
      wasm.gpu_render(lMin, lMax, cMin, cMax, hueRef.current, showHdrRef.current ? 1 : 0, t);
    } catch (err) {
      void recoverGpu('render', err);
    }
  };

  // Animation loop for HDR view: re-renders every frame so the marker
  // color oscillates. Started/stopped by handleHdrToggle.
  const startHdrLoop = () => {
    if (hdrRafRef.current !== null) return;
    const tick = () => {
      if (!showHdrRef.current) {
        hdrRafRef.current = null;
        return;
      }
      render();
      hdrRafRef.current = requestAnimationFrame(tick);
    };
    hdrRafRef.current = requestAnimationFrame(tick);
  };
  const stopHdrLoop = () => {
    if (hdrRafRef.current !== null) {
      cancelAnimationFrame(hdrRafRef.current);
      hdrRafRef.current = null;
    }
  };

  const handleHdrToggle = () => {
    const next = !showHdrRef.current;
    showHdrRef.current = next;
    setShowHdr(next);
    if (next) startHdrLoop();
    else {
      stopHdrLoop();
      render();
    }
  };

  // After the source state is set and React mounts the canvas, init the
  // GPU context (if needed), upload the source, and kick off precompute.
  useEffect(() => {
    const src = source;
    if (!src) return;
    // A new source is a fresh chance: clear any prior "GPU lost" state so the
    // user can try uploading again after a previous failure.
    gpuLostRef.current = false;
    setGpuLost(false);
    let cancelled = false;
    (async () => {
      try {
        await configureGpuForSource(src, false);
      } catch (err) {
        if (!cancelled) {
          void recoverGpu('init/upload', err);
        }
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source]);

  // Run seam-order precompute via worker. Used by the carve path; the
  // squish path doesn't need this.
  const runPrecompute = (src: SourceState, dir: Direction, onDone?: () => void) => {
    setIsPrecomputing(true);

    const id = ++requestIdRef.current;
    const dirNum = dir === 'width' ? 0 : 1;

    const worker = getWorker();
    worker.onmessage = (e: MessageEvent) => {
      const { order, requestId } = e.data;
      if (requestId !== requestIdRef.current) return;

      orderRef.current = order;
      orderDirectionRef.current = dir;
      setIsPrecomputing(false);
      onDone?.();
    };

    worker.postMessage({
      imageData: src.sdrData,
      width: src.w,
      height: src.h,
      direction: dirNum,
      requestId: id,
      useGPU: true,
    });
  };

  const ensureCarvePrecompute = (onDone?: () => void) => {
    const src = sourceRef.current;
    if (!src || resizeModeRef.current !== 'carve') {
      onDone?.();
      return;
    }
    if (isPrecomputing) return;
    if (orderRef.current && orderDirectionRef.current === directionRef.current) {
      onDone?.();
      return;
    }
    runPrecompute(src, directionRef.current, onDone);
  };

  const handleImageSelect = async (imageUrl: string) => {
    const wasm = wasmRef.current;
    if (!wasm) return;

    // Pull the bytes once so we can both feed the HDR decoder and (on
    // fallback) hand them to the browser image loader.
    let buf: Uint8Array;
    try {
      const resp = await fetch(imageUrl);
      buf = new Uint8Array(await resp.arrayBuffer());
    } catch (err) {
      console.error('image fetch failed:', err);
      return;
    }
    const jpegMeta = scanJpegMetadata(buf);

    // Try the HDR path first. Returns null for SDR / unsupported formats
    // so we can fall through to the canvas path.
    try {
      const { decodeHdr } = await import('@/lib/hdr-decode');
      const decodedRaw = await decodeHdr(buf);
      const decoded = decodedRaw
        ? orientHdrDecodeResult(decodedRaw, pickHdrManualRotation(jpegMeta))
        : null;
      if (decoded) {
        const src: SourceState = {
          url: imageUrl,
          // For real HDR sources `data` is the f16 buffer; for wasm-decoded
          // SDR sources (e.g. HEIC) it's the same 8-bit buffer as sdrData.
          data: decoded.hdr ? decoded.pixels! : decoded.sdrPixels,
          sdrData: decoded.sdrPixels,
          w: decoded.width,
          h: decoded.height,
          hdr: decoded.hdr,
          hasAlpha: bufferHasAlpha(decoded.sdrPixels),
        };
        setSource(src);
        sourceRef.current = src;
        // Histogram + seam worker run on the (tonemapped or actual) SDR proxy.
        lcHistRef.current = wasm.compute_lc_histogram(src.sdrData);
        return;
      }
    } catch (err) {
      console.warn('HDR decode threw, falling back to SDR:', err);
    }

    // SDR path: decode through a 2D canvas in display-p3 space.
    const blob = new Blob([buf as BlobPart]);
    let objectUrl: string | null = null;
    // Let the browser apply EXIF rotation. Trying to do it manually with
    // `imageOrientation: 'none'` is unreliable on iOS Safari, which bakes the
    // rotation into transcoded JPEG pixels yet still ships the original EXIF
    // tag, causing double-rotation. `'from-image'` is consistent across
    // browsers, so we treat the resulting bitmap as already correctly oriented.
    const decodeSource = async (): Promise<ImageBitmap | HTMLImageElement> => {
      if (typeof createImageBitmap === 'function') {
        try {
          return await createImageBitmap(blob, { imageOrientation: 'from-image' } as ImageBitmapOptions);
        } catch {
          // Fall back to <img> below.
        }
      }
      objectUrl = URL.createObjectURL(blob);
      return await new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('image decode failed'));
        img.src = objectUrl!;
      });
    };
    try {
      const image = await decodeSource();
      const isBitmap = typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap;
      const outWidth = isBitmap
        ? image.width
        : (image as HTMLImageElement).naturalWidth || image.width;
      const outHeight = isBitmap
        ? image.height
        : (image as HTMLImageElement).naturalHeight || image.height;
      const tmp = document.createElement('canvas');
      tmp.width = outWidth;
      tmp.height = outHeight;
      const ctx = tmp.getContext('2d', { colorSpace: 'display-p3' })!;
      ctx.drawImage(image, 0, 0);
      const data = ctx.getImageData(0, 0, outWidth, outHeight, {
        colorSpace: 'display-p3',
      });
      const sdrData = new Uint8Array(data.data);
      const src: SourceState = {
        url: imageUrl,
        data: sdrData,
        sdrData,
        w: outWidth,
        h: outHeight,
        hdr: false,
        hasAlpha: bufferHasAlpha(sdrData),
      };
      setSource(src);
      sourceRef.current = src;

      // Compute the L+chroma histogram once so the clipping readouts are live.
      lcHistRef.current = wasm.compute_lc_histogram(src.sdrData);
      if (isBitmap) image.close();
    } catch (err) {
      console.error('SDR decode failed:', err);
    } finally {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    }
  };

  const handleDirectionChange = (dir: Direction) => {
    setDirection(dir);
    directionRef.current = dir;
    setTargetWDraft(null);
    setTargetHDraft(null);
    const src = sourceRef.current;
    if (!src) return;
    // Carve constrains the *other* dimension to the source size — snap
    // it back when switching axes.
    if (dir === 'width') {
      targetHRef.current = src.h;
      setTargetH(src.h);
    } else {
      targetWRef.current = src.w;
      setTargetW(src.w);
    }
    // Re-apply any active aspect ratio against the new direction. If it
    // becomes infeasible in this direction (carve only), drop back to free.
    if (aspectRef.current !== 'free') {
      const snapped = snapToAspect(src, aspectRef.current, resizeModeRef.current, dir);
      if (snapped) {
        setDims(snapped.w, snapped.h);
      } else {
        aspectRef.current = 'free';
        setAspect('free');
      }
    }
    if (resizeOpen && resizeModeRef.current === 'carve' && orderDirectionRef.current !== dir) {
      runPrecompute(src, dir, () => {
        applyResize();
        render();
      });
    } else {
      applyResize();
      render();
    }
  };

  // Push (w, h) into state + refs together. Used by all the dim-changing
  // helpers so the squish/carve aspect-locked paths can update both at once.
  const setDims = (w: number, h: number) => {
    const prevPreview = resolveRotationTransform({
      targetW: targetWRef.current,
      targetH: targetHRef.current,
      straightenDeg: straightenDegRef.current,
      rotate90: rotate90Ref.current,
    });
    const nextPreview = resolveRotationTransform({
      targetW: w,
      targetH: h,
      straightenDeg: straightenDegRef.current,
      rotate90: rotate90Ref.current,
    });
    const nextCrop = scaleCropRect(
      cropRectRef.current,
      prevPreview.dstW,
      prevPreview.dstH,
      nextPreview.dstW,
      nextPreview.dstH
    );
    const nextAnchor = scalePoint(
      cropAnchorRef.current,
      targetWRef.current,
      targetHRef.current,
      w,
      h
    );
    targetWRef.current = w;
    targetHRef.current = h;
    cropRectRef.current = nextCrop;
    cropAnchorRef.current = nextAnchor;
    setTargetW(w);
    setTargetH(h);
    setTargetWDraft(null);
    setTargetHDraft(null);
    setCropRect(nextCrop);
    setCropDrag(null);
  };

  // In carve mode an active aspect ratio fully determines (W, H), so a
  // direct slider/input edit means the user wants out of that constraint.
  // Drop the aspect lock back to "free" so the dim they're dragging is
  // actually editable.
  const breakAspectIfCarve = () => {
    if (resizeModeRef.current === 'carve' && aspectRef.current !== 'free') {
      aspectRef.current = 'free';
      setAspect('free');
    }
  };

  const handleTargetWChange = (size: number) => {
    const src = sourceRef.current;
    if (!src) return;
    breakAspectIfCarve();
    let w = Math.max(1, Math.min(src.w, Math.round(size)));
    let h = targetHRef.current;
    const ratio = aspectRatioValue(aspectRef.current, src);
    if (ratio !== null && resizeModeRef.current === 'squish') {
      h = Math.max(1, Math.min(src.h, Math.round(w / ratio)));
      // If the derived H clipped against source, back-solve W to keep ratio.
      if (h === src.h) w = Math.max(1, Math.min(src.w, Math.round(h * ratio)));
    }
    setDims(w, h);
    applyResize();
    render();
  };

  const handleTargetHChange = (size: number) => {
    const src = sourceRef.current;
    if (!src) return;
    breakAspectIfCarve();
    let h = Math.max(1, Math.min(src.h, Math.round(size)));
    let w = targetWRef.current;
    const ratio = aspectRatioValue(aspectRef.current, src);
    if (ratio !== null && resizeModeRef.current === 'squish') {
      w = Math.max(1, Math.min(src.w, Math.round(h * ratio)));
      if (w === src.w) h = Math.max(1, Math.min(src.h, Math.round(w / ratio)));
    }
    setDims(w, h);
    applyResize();
    render();
  };

  // Apply the current aspect ratio constraint to the dims given the current
  // mode + direction. In carve mode this may flip the direction.
  // Returns null when the requested aspect can't be reached in carve mode
  // for the current direction (even after inverting). The caller falls
  // back to leaving the dims unchanged in that case.
  const snapToAspect = (
    src: SourceState,
    ar: AspectRatio,
    mode: ResizeMode,
    dir: Direction
  ): { w: number; h: number } | null => {
    const ratio = aspectRatioValue(ar, src);
    if (ratio === null) {
      return { w: targetWRef.current || src.w, h: targetHRef.current || src.h };
    }
    if (mode === 'squish') {
      const [w, h] = snapSquish(ratio, src);
      return { w, h };
    }
    return snapCarve(ratio, src, dir);
  };

  const handleAspectChange = (next: AspectRatio) => {
    const src = sourceRef.current;
    if (!src) return;
    const snapped = snapToAspect(src, next, resizeModeRef.current, directionRef.current);
    if (!snapped) return; // infeasible — leave the prior aspect selected
    setAspect(next);
    aspectRef.current = next;
    setDims(snapped.w, snapped.h);
    applyResize();
    render();
  };

  const handleResizeModeChange = (mode: ResizeMode) => {
    if (mode === resizeModeRef.current) return;
    setResizeMode(mode);
    resizeModeRef.current = mode;
    setTargetWDraft(null);
    setTargetHDraft(null);
    const src = sourceRef.current;
    if (!src) return;
    // Switching to carve constrains the inactive dimension to source.
    if (mode === 'carve') {
      if (directionRef.current === 'width') {
        targetHRef.current = src.h;
        setTargetH(src.h);
      } else {
        targetWRef.current = src.w;
        setTargetW(src.w);
      }
    }
    // Lock collapses to "no resize" in carve mode (since carve can only
    // change one dim, the only solution that preserves source aspect is
    // the source dims). Drop it.
    if (mode === 'carve' && aspectRef.current === 'lock') {
      aspectRef.current = 'free';
      setAspect('free');
    }
    // Re-apply any remaining active aspect against the new mode.
    if (aspectRef.current !== 'free') {
      const snapped = snapToAspect(src, aspectRef.current, mode, directionRef.current);
      if (snapped) {
        setDims(snapped.w, snapped.h);
      } else {
        aspectRef.current = 'free';
        setAspect('free');
      }
    }
    if (resizeOpen && mode === 'carve' && orderDirectionRef.current !== directionRef.current) {
      runPrecompute(src, directionRef.current, () => {
        applyResize();
        render();
      });
    } else {
      applyResize();
      render();
    }
  };

  const handleLRangeChange = (next: [number, number]) => {
    setLRange(next);
    lRangeRef.current = next;
    if (ready) render();
  };

  const handleCRangeChange = (next: [number, number]) => {
    setCRange(next);
    cRangeRef.current = next;
    if (ready) render();
  };

  const handleHueChange = (next: number) => {
    setHue(next);
    hueRef.current = next;
    if (ready) render();
  };

  // Bump exposure by N stops. Composes with the current L transform:
  // a stop is a 2× factor in *linear* light, which corresponds to a 2^(1/3)
  // factor in L*. The remap is affine in L*, so we just multiply both
  // endpoints by k = 2^(stops/3) to apply the same scale to L_out.
  const adjustExposure = (stops: number) => {
    const k = Math.pow(2, stops / 3);
    const next: [number, number] = [lRangeRef.current[0] * k, lRangeRef.current[1] * k];
    const snapped: [number, number] = [Math.round(next[0]), Math.round(next[1])];
    handleLRangeChange(snapped);
  };

  type ExportFormat = 'jpeg' | 'avif';
  const [exportFormat, setExportFormat] = useState<ExportFormat>('avif');
  const [downloading, setDownloading] = useState(false);
  const [smallerFile, setSmallerFile] = useState(false);
  const [lossless, setLossless] = useState(false);

  // Crop state. cropRect is in target output pixel coordinates
  // (the same space as targetW × targetH). null = no crop.
  const handleDownload = async () => {
    const wasm = wasmRef.current;
    const src = sourceRef.current;
    if (!wasm || !src) return;
    setDownloading(true);
    try {
      const transform = syncGpuTransform();
      // Make sure the texture has the latest params (and turn off the
      // HDR-view marker pixels for the export).
      const [lMin, lMax] = lRangeRef.current;
      const [cMin, cMax] = cRangeRef.current;
      wasm.gpu_render(lMin, lMax, cMin, cMax, hueRef.current, 0, 0);

      const outW = transform.active ? transform.dstW : targetWRef.current;
      const outH = transform.active ? transform.dstH : targetHRef.current;

      let blob: Blob;
      let filename: string;
      if (exportFormat === 'jpeg') {
        // Linear extended Display P3 half-floats → Ultra HDR JPEG.
        // libultrahdr derives the SDR base + gain map from the HDR input.
        const f16 = await wasm.gpu_readback_linear_f16();
        // Scan the buffer for the actual peak linear value so we can
        // tell libultrahdr what hdr_capacity_max should be. Without this
        // the encoder writes the default 10000-nit headroom into the
        // metadata regardless of content; decoders then under-apply the
        // gain map and the JPEG looks dim. f16 layout: 2 bytes per
        // channel, 4 channels per pixel; only RGB matter.
        let peak = 1.0;
        for (let i = 0; i < f16.length; i += 8) {
          for (let c = 0; c < 3; c++) {
            const lo = f16[i + c * 2];
            const hi = f16[i + c * 2 + 1];
            const h = lo | (hi << 8);
            const sign = (h & 0x8000) >>> 15;
            const exp = (h & 0x7c00) >>> 10;
            const frac = h & 0x3ff;
            let v: number;
            if (exp === 0) v = (frac / 1024) * Math.pow(2, -14);
            else if (exp === 31) continue;
            else v = (1 + frac / 1024) * Math.pow(2, exp - 15);
            if (sign) v = -v;
            if (v > peak) peak = v;
          }
        }
        // 1.0 in linear extended Display P3 = SDR diffuse white = 203 nits
        // per BT.2408. Clamp to libultrahdr's accepted [203, 10000] range.
        const peakNits = Math.max(203, Math.min(10000, peak * 203));
        const { encodeUhdr } = await import('@/lib/uhdr');
        const jpeg = await encodeUhdr(f16, outW, outH, {
          targetDisplayPeakNits: peakNits,
          baseQuality: smallerFile ? 70 : 90,
          gainMapQuality: smallerFile ? 80 : 95,
        });
        blob = new Blob([new Uint8Array(jpeg)], { type: 'image/jpeg' });
        filename = 'pictolab.jpg';
      } else {
        // 10-bit BT.2020 PQ readback for HDR AVIF.
        const rgba16 = await wasm.gpu_readback_hdr_pq_u16(10);
        const { encodeAvif, CP_BT2020, TC_PQ, MC_BT2020_NCL, MC_IDENTITY } = await import('@/lib/avif-hdr');
        const avif = await encodeAvif(rgba16, outW, outH, {
          bitDepth: 10,
          // Lossless: q=100, 4:4:4 subsampling, identity matrix so the
          // RGB samples are stored verbatim (no YUV round-trip).
          // Otherwise: higher than the libavif default (~60) — for HDR
          // master output we want chroma transitions in saturated
          // highlights to stay clean, and 60 leaves visible
          // banding/blocking on bright sky.
          quality: lossless ? 100 : smallerFile ? 55 : 80,
          subsample: lossless ? 0 : 1,
          cicpColorPrimaries: CP_BT2020,
          cicpTransferCharacteristics: TC_PQ,
          cicpMatrixCoefficients: lossless ? MC_IDENTITY : MC_BT2020_NCL,
          // gpu_readback_hdr_pq_u16 quantizes [0,1] linearly to the full
          // 0..(2^depth-1) code-point range, so the AVIF must declare
          // full range too. Telling libavif "limited" here would make
          // decoders re-expand the data and brighten everything.
          fullRange: true,
        });
        blob = new Blob([new Uint8Array(avif)], { type: 'image/avif' });
        filename = 'pictolab.avif';
      }
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('export failed:', err);
    } finally {
      setDownloading(false);
      render();
    }
  };

  const handleNewImage = () => {
    setSource(null);
    // Clear any prior terminal GPU-lost state so the user sees the upload
    // drop zone instead of the banner — the banner's own copy tells them
    // to pick a different image, and this is that path.
    gpuLostRef.current = false;
    setGpuLost(false);
    setTargetWDraft(null);
    setTargetHDraft(null);
    sourceRef.current = null;
    orderRef.current = null;
    orderDirectionRef.current = null;
    lcHistRef.current = null;
    setReady(false);
    setLRange([0, 100]);
    setCRange([0, 100]);
    setHue(0);
    setResizeMode('carve');
    resizeModeRef.current = 'carve';
    setAspect('free');
    aspectRef.current = 'free';
    lRangeRef.current = [0, 100];
    cRangeRef.current = [0, 100];
    hueRef.current = 0;
    setCropTool(false);
    setCropRect(null);
    setCropDrag(null);
    setStraightenDeg(0);
    setRotate90(0);
    straightenDegRef.current = 0;
    rotate90Ref.current = 0;
    cropRectRef.current = null;
    cropAnchorRef.current = null;
    cropToolRef.current = false;
  };

  const outW = targetW;
  const outH = targetH;

  // Live clipping percentages for the current L remap.
  const lClipping = (() => {
    const hist = lcHistRef.current;
    if (!hist || !source) return null;
    const total = source.w * source.h;
    const [lMin, lMax] = lRange;
    const scale = (lMax - lMin) / 100;
    let dark = 0;
    let light = 0;
    for (let l = 0; l <= 100; l++) {
      const out = lMin + l * scale;
      if (out < 0) dark += hist[l];
      else if (out > 100) light += hist[l];
    }
    return { dark: (dark / total) * 100, light: (light / total) * 100 };
  })();

  // Approximate chroma clipping. Mirrors the per-pixel mapping in remap_lab.
  const cClipping = (() => {
    const hist = lcHistRef.current;
    if (!hist || !source) return null;
    const total = source.w * source.h;
    const [cMin, cMax] = cRange;
    let boosted = 0;
    let inverted = 0;
    for (let c = 1; c <= 160; c++) {
      const chromaNorm = c / 160;
      const outChromaNorm = (cMin + (cMax - cMin) * chromaNorm) / 100;
      const outChroma = outChromaNorm * 160;
      if (outChroma > 160) boosted += hist[101 + c];
      else if (outChroma < 0) inverted += hist[101 + c];
    }
    return {
      boosted: (boosted / total) * 100,
      inverted: (inverted / total) * 100,
    };
  })();

  const gpuMissing = gpuAvailable === false;
  const hdrPresentationMissing = hdrPresentationActive === false;

  // Resolve the rotation pass at render time so the JSX can adjust the
  // wrapper aspect / dimension label / crop overlay visibility. Cheap.
  const editPreviewTransformNow = source
    ? resolveRotationTransform({
        targetW,
        targetH,
        straightenDeg,
        rotate90,
      })
    : { active: false, m: [1, 0, 0, 1, 0, 0] as [number, number, number, number, number, number], dstW: 0, dstH: 0 };
  const outputTransformNow = source
    ? resolveOutputTransform({
        targetW,
        targetH,
        crop: cropRect,
        straightenDeg,
        rotate90,
      })
    : { active: false, m: [1, 0, 0, 1, 0, 0] as [number, number, number, number, number, number], dstW: 0, dstH: 0 };
  const previewTransformNow = cropTool ? editPreviewTransformNow : outputTransformNow;
  const previewTransformActive = previewTransformNow.active;
  const deferCarveSliderUpdates = LOW_CORE_DEVICE && resizeMode === 'carve';
  const widthSliderValue = deferCarveSliderUpdates ? (targetWDraft ?? targetW) : targetW;
  const heightSliderValue = deferCarveSliderUpdates ? (targetHDraft ?? targetH) : targetH;

  return (
    <div className="flex h-full min-h-screen flex-col bg-background text-foreground lg:h-screen lg:overflow-hidden">
      <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-3 sm:px-6 sm:py-4">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">Pictolab</h1>
          </div>
          <div className="flex items-center gap-2">
            {source && (
              <Button variant="outline" size="sm" onClick={handleNewImage}>
                <ImagePlus className="mr-1 h-4 w-4" />
                New image
              </Button>
            )}
            <Button variant="outline" size="sm" asChild>
              <a
                href="https://github.com/anchpop/pictolab"
                target="_blank"
                rel="noopener noreferrer"
                title="View source on GitHub"
              >
                <svg
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  className="h-4 w-4"
                  aria-hidden="true"
                >
                  <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.1.79-.25.79-.56 0-.27-.01-1-.02-1.96-3.2.69-3.87-1.54-3.87-1.54-.52-1.33-1.28-1.69-1.28-1.69-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.03 1.76 2.7 1.25 3.36.96.1-.75.4-1.25.73-1.54-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.18-3.09-.12-.29-.51-1.46.11-3.05 0 0 .97-.31 3.18 1.18.92-.26 1.91-.39 2.89-.39.98 0 1.97.13 2.89.39 2.21-1.49 3.18-1.18 3.18-1.18.62 1.59.23 2.76.11 3.05.74.8 1.18 1.83 1.18 3.09 0 4.42-2.69 5.39-5.25 5.68.41.36.78 1.06.78 2.13 0 1.54-.01 2.78-.01 3.16 0 .31.21.67.8.55C20.21 21.39 23.5 17.08 23.5 12 23.5 5.65 18.35.5 12 .5z" />
                </svg>
              </a>
            </Button>
          </div>
        </header>

      <div className="flex flex-1 flex-col lg:min-h-0 lg:flex-row">
        {/* ── Main preview area ──────────────────────────────────────── */}
        <main className="flex flex-1 flex-col bg-background lg:min-h-0">
        <div
          className="flex flex-1 items-center justify-center overflow-auto p-4 sm:p-8"
          style={{
            // Dark canvas with a faint dot grid so the preview area reads
            // as a workspace, not a flat panel.
            backgroundColor: '#1a1a1f',
            backgroundImage:
              'radial-gradient(circle, rgba(255,255,255,0.12) 1px, transparent 1px)',
            backgroundSize: '18px 18px',
          }}
        >
          {gpuMissing ? (
            <div className="max-w-md text-center text-sm text-muted-foreground">
              WebGPU is required but unavailable on this device. Pictolab needs a
              browser with WebGPU enabled (recent Chrome, Edge, or Safari on iOS 26).
            </div>
          ) : gpuLost ? (
            <div className="max-w-md text-center text-sm text-muted-foreground">
              The GPU context was lost and could not be recovered. Reload the page
              to try again, or pick a different image.
            </div>
          ) : !source ? (
            <div className="w-full max-w-2xl">
              <ImageDropZone onImageSelect={handleImageSelect} />
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <div
                className="relative overflow-hidden rounded-lg border border-border shadow-sm"
                style={{
                  aspectRatio: previewTransformActive
                    ? `${previewTransformNow.dstW} / ${previewTransformNow.dstH}`
                    : `${source.w} / ${source.h}`,
                  // Fit inside both 90vw and 50vh (mobile) / 75vh (desktop)
                  // by computing the largest width that satisfies both.
                  width: previewTransformActive
                    ? `min(90vw, 60vh * ${previewTransformNow.dstW / previewTransformNow.dstH})`
                    : `min(90vw, 60vh * ${source.w / source.h})`,
                  // Checkerboard so transparent pixels are obvious as
                  // such instead of getting blended with the (white)
                  // card background.
                  backgroundColor: '#ffffff',
                  backgroundImage:
                    'linear-gradient(45deg, #d8d8d8 25%, transparent 25%), linear-gradient(-45deg, #d8d8d8 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #d8d8d8 75%), linear-gradient(-45deg, transparent 75%, #d8d8d8 75%)',
                  backgroundSize: '16px 16px',
                  backgroundPosition: '0 0, 0 8px, 8px -8px, -8px 0',
                }}
              >
                <canvas
                  ref={canvasRef}
                  className="absolute top-0 left-0 block"
                  style={{
                    width: `${(targetW / source.w) * 100}%`,
                    height: `${(targetH / source.h) * 100}%`,
                  }}
                />
                {/* Crop overlay: positioned identically to the canvas
                    so its coordinate space matches the rendered image
                    pixel-for-pixel. Captures pointer events only when
                    the crop tool is active; otherwise it's strictly
                    decorative (drawn marquee for an existing crop). */}
                {cropTool && (
                  <div
                    className="absolute top-0 left-0"
                    style={{
                      width: previewTransformActive ? '100%' : `${(targetW / source.w) * 100}%`,
                      height: previewTransformActive ? '100%' : `${(targetH / source.h) * 100}%`,
                      cursor: cropTool ? 'crosshair' : 'default',
                      pointerEvents: cropTool ? 'auto' : 'none',
                      touchAction: cropTool ? 'none' : 'auto',
                    }}
                    onPointerDown={(e) => {
                      if (!cropTool) return;
                      const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
                      const fx = (e.clientX - rect.left) / rect.width;
                      const fy = (e.clientY - rect.top) / rect.height;
                      const px = Math.max(0, Math.min(previewTransformNow.dstW, fx * previewTransformNow.dstW));
                      const py = Math.max(0, Math.min(previewTransformNow.dstH, fy * previewTransformNow.dstH));
                      (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
                      setCropDrag({ x0: px, y0: py, x1: px, y1: py });
                    }}
                    onPointerMove={(e) => {
                      if (!cropTool || !cropDrag) return;
                      const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
                      const fx = (e.clientX - rect.left) / rect.width;
                      const fy = (e.clientY - rect.top) / rect.height;
                      const px = Math.max(0, Math.min(previewTransformNow.dstW, fx * previewTransformNow.dstW));
                      const py = Math.max(0, Math.min(previewTransformNow.dstH, fy * previewTransformNow.dstH));
                      setCropDrag({ ...cropDrag, x1: px, y1: py });
                    }}
                    onPointerUp={(e) => {
                      if (!cropTool || !cropDrag) return;
                      (e.currentTarget as HTMLDivElement).releasePointerCapture(e.pointerId);
                      const x = Math.min(cropDrag.x0, cropDrag.x1);
                      const y = Math.min(cropDrag.y0, cropDrag.y1);
                      const w = Math.abs(cropDrag.x1 - cropDrag.x0);
                      const h = Math.abs(cropDrag.y1 - cropDrag.y0);
                      setCropDrag(null);
                      // Ignore stray clicks (need at least a few px in
                      // both dimensions for it to be a real selection).
                      // Crop mode stays on after a successful drag so
                      // the rotate/straighten panel below remains
                      // accessible — the user dismisses it with Done.
                      if (w >= 4 && h >= 4) {
                        const nextCrop = { x, y, w, h };
                        cropAnchorRef.current = mapPreviewPointToSource(previewTransformNow, {
                          x: x + w / 2,
                          y: y + h / 2,
                        });
                        cropRectRef.current = nextCrop;
                        setCropRect(nextCrop);
                      }
                    }}
                  >
                    {(() => {
                      // Build the visible marquee. While dragging, use
                      // the live drag rect; otherwise use the committed
                      // cropRect.
                      let rx: number, ry: number, rw: number, rh: number;
                      if (cropDrag) {
                        rx = Math.min(cropDrag.x0, cropDrag.x1);
                        ry = Math.min(cropDrag.y0, cropDrag.y1);
                        rw = Math.abs(cropDrag.x1 - cropDrag.x0);
                        rh = Math.abs(cropDrag.y1 - cropDrag.y0);
                      } else if (cropRect) {
                        rx = cropRect.x; ry = cropRect.y; rw = cropRect.w; rh = cropRect.h;
                      } else {
                        return null;
                      }
                      return (
                        <svg
                          className="absolute inset-0 h-full w-full"
                          viewBox={`0 0 ${previewTransformNow.dstW} ${previewTransformNow.dstH}`}
                          preserveAspectRatio="none"
                        >
                          {/* Mask: dim everything except the rect. */}
                          <defs>
                            <mask id="crop-mask">
                              <rect width={previewTransformNow.dstW} height={previewTransformNow.dstH} fill="white" />
                              <rect x={rx} y={ry} width={rw} height={rh} fill="black" />
                            </mask>
                          </defs>
                          <rect
                            width={previewTransformNow.dstW}
                            height={previewTransformNow.dstH}
                            fill="rgba(0,0,0,0.5)"
                            mask="url(#crop-mask)"
                          />
                          <rect
                            x={rx}
                            y={ry}
                            width={rw}
                            height={rh}
                            fill="none"
                            stroke="white"
                            strokeWidth={Math.max(1, Math.min(previewTransformNow.dstW, previewTransformNow.dstH) / 400)}
                            vectorEffect="non-scaling-stroke"
                          />
                        </svg>
                      );
                    })()}
                  </div>
                )}
                {isPrecomputing && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/30 backdrop-blur-[2px]">
                    <div className="flex items-center gap-2 rounded-md bg-background/90 px-3 py-2 text-sm text-foreground shadow-lg">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Precomputing seam order…
                    </div>
                  </div>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant={cropTool ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => {
                    setCropTool((v) => !v);
                    setCropDrag(null);
                  }}
                  title="Click and drag on the image to crop"
                >
                  <CropIcon className="mr-1 h-4 w-4" />
                  {cropTool ? 'Done' : cropRect ? 'Re-crop' : 'Crop'}
                </Button>
                {cropRect && !cropTool && (
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => {
                      setCropRect(null);
                      cropRectRef.current = null;
                      cropAnchorRef.current = null;
                      setStraightenDeg(0);
                      setRotate90(0);
                      straightenDegRef.current = 0;
                      rotate90Ref.current = 0;
                    }}
                    title="Remove crop"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                )}
              </div>
              <p className="text-xs text-muted-foreground">
                {outputTransformNow.active
                  ? `${outputTransformNow.dstW} × ${outputTransformNow.dstH} (transformed, from ${outW} × ${outH})`
                  : cropRect
                  ? `${Math.round(cropRect.w)} × ${Math.round(cropRect.h)} (cropped from ${outW} × ${outH})`
                  : `${outW} × ${outH}`}
              </p>
              {cropTool && (
                <div className="flex w-full max-w-lg items-center gap-3">
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-9 w-9 shrink-0"
                    onClick={() => setRotate90((((rotate90 - 1) % 4) + 4) % 4)}
                    title="Rotate 90° counter-clockwise"
                  >
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                  <TickSlider
                    value={straightenDeg}
                    min={-45}
                    max={45}
                    onChange={setStraightenDeg}
                  />
                  <Button
                    variant="outline"
                    size="icon"
                    className="h-9 w-9 shrink-0"
                    onClick={() => setRotate90((((rotate90 + 1) % 4) + 4) % 4)}
                    title="Rotate 90° clockwise"
                  >
                    <RotateCw className="h-4 w-4" />
                  </Button>
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      {/* ── Right sidebar ──────────────────────────────────────────── */}
      <aside className="w-full shrink-0 overflow-y-auto border-t border-border bg-card p-4 lg:h-full lg:min-h-0 lg:w-80 lg:border-t-0 lg:border-l">
        {!source ? (
          gpuMissing ? (
            <div className="text-sm text-muted-foreground">WebGPU unavailable.</div>
          ) : (
            <div className="space-y-3 text-sm text-foreground">
              <p>Drop an image to start editing.</p>
              <div className="space-y-3">
                <p className="font-medium">Why Pictolab?</p>

                <div>
                  <p className="font-medium">High-quality resizing</p>
                  <p className="text-muted-foreground">
                    Lanczos resampling, widely considered the gold standard for downscaling
                    photos, preserving sharpness without the ringing or blur of simpler filters.
                    Plus content-aware resize (seam carving) when you need to change aspect
                    ratio without squashing the subject.
                  </p>
                </div>

                <div>
                  <p className="font-medium">Photometrically correct color</p>
                  <p className="text-muted-foreground">
                    Brightness and tone adjustments happen in OKLab, a perceptually uniform
                    color space, so changes look the way your eyes expect instead of
                    skewing hue and saturation.
                  </p>
                </div>

                <div>
                  <p className="font-medium">Fast, local, GPU-accelerated</p>
                  <p className="text-muted-foreground">
                    Built on WebGPU. Everything runs on your machine. Your images never
                    leave the browser.
                  </p>
                </div>

                <div>
                  <p className="font-medium">End-to-end HDR</p>
                  <p className="text-muted-foreground">
                    Full HDR pipeline with true 10-bit color, including HEIC input and HDR
                    output.
                  </p>
                </div>

                <div>
                  <p className="font-medium">AVIF and JPEG export</p>
                  <p className="text-muted-foreground">
                    AVIF is the default: lossless mode beats PNG on file size, and lossy
                    mode beats JPEG at the same quality. For maximum compatibility we also
                    support Ultra HDR JPEG export.
                  </p>
                </div>
              </div>
            </div>
          )
        ) : (
          <div className="space-y-6 pb-24">
            <Button
              variant={showHdr ? 'default' : 'outline'}
              size="sm"
              className="w-full"
              onClick={handleHdrToggle}
              title="Highlight pixels that exceed SDR (will clip on standard displays)"
            >
              <Eye className="mr-1 h-4 w-4" />
              HDR view
            </Button>

            {hdrPresentationMissing && (
              <div className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-[11px] leading-snug text-amber-900">
                Your browser does not support HDR. The preview will be SDR, but the exported file will still be HDR.
              </div>
            )}

            <section>
              <Collapsible
                label={<CardTitle>Resize</CardTitle>}
                open={resizeOpen}
                onOpenChange={setResizeOpen}
                className="space-y-4"
              >
                <Segmented<ResizeMode>
                  value={resizeMode}
                  onValueChange={handleResizeModeChange}
                  options={[
                    {
                      value: 'squish',
                      label: 'Squish',
                      title: 'Lanczos resampling — fast, classic resize',
                    },
                    {
                      value: 'carve',
                      label: 'Content aware',
                      title: 'Content-aware resize via seam carving',
                      disabled: isPrecomputing,
                    },
                  ]}
                />

                <div className="flex flex-wrap gap-1 pt-3">
                  {ASPECT_OPTIONS.filter((opt) => {
                    // Lock is meaningless in carve mode — the only ratio
                    // that preserves source aspect is the source itself.
                    if (resizeMode === 'carve' && opt.value === 'lock') return false;
                    // Hide any aspect that can't be directly reached in
                    // the current carve direction. The inverted ratio
                    // has its own button (e.g. 9:16 vs 16:9), so we'd
                    // just be duplicating it.
                    if (
                      source &&
                      resizeMode === 'carve' &&
                      opt.value !== 'free'
                    ) {
                      const r = aspectRatioValue(opt.value, source);
                      if (r === null || snapCarve(r, source, direction) === null) {
                        return false;
                      }
                    }
                    return true;
                  }).map((opt) => (
                    <Button
                      key={opt.value}
                      variant={aspect === opt.value ? 'default' : 'outline'}
                      size="sm"
                      className="h-7 flex-1 px-2 text-xs"
                      onClick={() => handleAspectChange(opt.value)}
                      disabled={isPrecomputing}
                    >
                      {opt.label}
                    </Button>
                  ))}
                </div>
                {resizeMode === 'carve' && (
                  <Segmented<Direction>
                    value={direction}
                    onValueChange={handleDirectionChange}
                    options={[
                      { value: 'width', label: 'Width', disabled: isPrecomputing },
                      { value: 'height', label: 'Height', disabled: isPrecomputing },
                    ]}
                  />
                )}

                {ready && (
                  <>
                    {/* Width slider — always present in squish mode; only
                        when direction=width in carve mode. */}
                    {(resizeMode === 'squish' || direction === 'width') && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Label>Width</Label>
                          <Input
                            type="number"
                            value={targetW}
                            min={1}
                            max={source.w}
                            disabled={isPrecomputing}
                            onChange={(e) => handleTargetWChange(Number(e.target.value))}
                            className="h-6 w-16 px-1.5 py-0 text-right font-mono text-xs"
                          />
                        </div>
                        <Slider
                          min={1}
                          max={source.w}
                          step={1}
                          value={[widthSliderValue]}
                          disabled={isPrecomputing}
                          onValueChange={(v) => {
                            if (deferCarveSliderUpdates) setTargetWDraft(v[0]);
                            else handleTargetWChange(v[0]);
                          }}
                          onValueCommit={(v) => {
                            if (deferCarveSliderUpdates) handleTargetWChange(v[0]);
                          }}
                        />
                      </div>
                    )}

                    {/* Height slider — same logic, mirrored. */}
                    {(resizeMode === 'squish' || direction === 'height') && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Label>Height</Label>
                          <Input
                            type="number"
                            value={targetH}
                            min={1}
                            max={source.h}
                            disabled={isPrecomputing}
                            onChange={(e) => handleTargetHChange(Number(e.target.value))}
                            className="h-6 w-16 px-1.5 py-0 text-right font-mono text-xs"
                          />
                        </div>
                        <Slider
                          min={1}
                          max={source.h}
                          step={1}
                          value={[heightSliderValue]}
                          disabled={isPrecomputing}
                          onValueChange={(v) => {
                            if (deferCarveSliderUpdates) setTargetHDraft(v[0]);
                            else handleTargetHChange(v[0]);
                          }}
                          onValueCommit={(v) => {
                            if (deferCarveSliderUpdates) handleTargetHChange(v[0]);
                          }}
                        />
                      </div>
                    )}
                  </>
                )}
                {resizeMode === 'carve' && (
                  <p className="text-[11px] leading-snug text-muted-foreground">
                    <a
                      href="https://avikdas.com/2019/07/29/improved-seam-carving-with-forward-energy.html"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline decoration-dotted hover:text-foreground"
                    >
                      Seam carving with forward energy
                    </a>
                    {' '}— removes low-energy seams instead of resampling.
                  </p>
                )}
              </Collapsible>
            </section>

            <section>
              <Collapsible
                label={<CardTitle>Lightness</CardTitle>}
                open={lightnessOpen}
                onOpenChange={setLightnessOpen}
                className="space-y-4"
              >
                <DualSlider
                  min={-50}
                  max={150}
                  step={1}
                  value={lRange}
                  onValueChange={handleLRangeChange}
                  safeRange={[0, 100]}
                  snapPoints={[0, 100]}
                />
                <div className="flex w-full items-center gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 flex-1"
                    onClick={() => handleLRangeChange([lRangeRef.current[1], lRangeRef.current[0]])}
                    title="Invert (swap handles)"
                  >
                    <ArrowLeftRight className="h-3 w-3" />
                    Invert
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 flex-1"
                    onClick={() => adjustExposure(-0.5)}
                    title="−½ stop"
                  >
                    <Minus className="h-3 w-3" />
                    ½ EV
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 flex-1"
                    onClick={() => adjustExposure(0.5)}
                    title="+½ stop"
                  >
                    <Plus className="h-3 w-3" />
                    ½ EV
                  </Button>
                </div>
                {lClipping && (lClipping.dark > 0 || lClipping.light > 0) ? (
                  <p className="text-[11px] leading-snug text-destructive">
                    Clipping: {lClipping.dark.toFixed(1)}% shadows · {lClipping.light.toFixed(1)}% highlights
                  </p>
                ) : null}
              </Collapsible>
            </section>

            <section>
              <Collapsible
                label={<CardTitle>Chroma</CardTitle>}
                open={chromaOpen}
                onOpenChange={setChromaOpen}
                className="space-y-4"
              >
                <DualSlider
                  min={-50}
                  max={150}
                  step={1}
                  value={cRange}
                  onValueChange={handleCRangeChange}
                  safeRange={[0, 100]}
                  snapPoints={[0, 100]}
                />
                {cClipping && (cClipping.boosted > 0 || cClipping.inverted > 0) ? (
                  <p className="text-[11px] leading-snug text-destructive">
                    {cClipping.boosted > 0 && <>Clipping: {cClipping.boosted.toFixed(1)}% boosted past gamut</>}
                    {cClipping.boosted > 0 && cClipping.inverted > 0 && <> · </>}
                    {cClipping.inverted > 0 && <>{cClipping.inverted.toFixed(1)}% hue-inverted</>}
                  </p>
                ) : null}
              </Collapsible>
            </section>

            <section>
              <Collapsible
                label={<CardTitle>Hue</CardTitle>}
                open={hueOpen}
                onOpenChange={setHueOpen}
                className="space-y-4"
              >
                <div className="space-y-2">
                  <TickSlider
                    value={hue}
                    min={-180}
                    max={180}
                    onChange={handleHueChange}
                    tickStep={15}
                    majorTickStep={45}
                    snapStep={1}
                    trackClassName="bg-secondary"
                    activeTickClassName="bg-primary/80"
                    activeMarkerClassName="bg-primary/80"
                    title="Drag to rotate hue · double-click to reset"
                    resetTitle="Reset hue rotation to 0°"
                    formatValue={(v) => `${v > 0 ? '+' : ''}${Math.round(v)}°`}
                  />
                </div>
              </Collapsible>
            </section>

            <section>
              <Collapsible
                label={<CardTitle>Export</CardTitle>}
                open={exportOpen}
                onOpenChange={setExportOpen}
                className="space-y-4"
              >
              <Segmented<ExportFormat>
                value={exportFormat}
                onValueChange={setExportFormat}
                options={[
                  {
                    value: 'jpeg',
                    label: 'JPEG',
                    title: 'Ultra HDR JPEG (SDR base + gain map). Widely supported.',
                  },
                  {
                    value: 'avif',
                    label: 'AVIF',
                    title: '10-bit BT.2020 PQ AVIF. Smaller, real HDR.',
                  },
                ]}
              />
              {exportFormat === 'jpeg' && source.hasAlpha && (
                <p className="rounded border border-amber-300 bg-amber-50 px-2 py-1.5 text-[11px] leading-snug text-amber-900">
                  This image has transparency, which JPEG can't store —
                  transparent pixels will be flattened to a background
                  color. Use AVIF to preserve alpha.
                </p>
              )}
              <div className="sticky bottom-0 z-10 -mx-4 bg-card/95 px-4 py-3 backdrop-blur supports-[backdrop-filter]:bg-card/80">
                <Button
                  className="w-full"
                  onClick={handleDownload}
                  disabled={!ready || downloading}
                >
                  {downloading ? (
                    <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                  ) : (
                    <Download className="mr-1 h-4 w-4" />
                  )}
                  Export {exportFormat === 'jpeg' ? 'JPEG' : 'AVIF'}
                </Button>
              </div>
              <div className="flex items-center justify-between pt-1">
                <Label htmlFor="smaller-file" className="text-xs font-normal text-muted-foreground">
                  Smaller file size (lower quality)
                </Label>
                <Switch
                  id="smaller-file"
                  checked={smallerFile && !(exportFormat === 'avif' && lossless)}
                  disabled={exportFormat === 'avif' && lossless}
                  onCheckedChange={setSmallerFile}
                />
              </div>
              {exportFormat === 'avif' && (
                <div className="flex items-center justify-between">
                  <Label htmlFor="lossless" className="text-xs font-normal text-muted-foreground">
                    Lossless (much larger file)
                  </Label>
                  <Switch
                    id="lossless"
                    checked={lossless}
                    onCheckedChange={setLossless}
                  />
                </div>
              )}
              </Collapsible>
            </section>
          </div>
        )}
      </aside>
      </div>
    </div>
  );
}

export default Editor;
