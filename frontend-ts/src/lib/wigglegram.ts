// Wigglegram pipeline: a three-lens wigglegram lens exposes three
// side-by-side sub-frames onto one photo. Split them, match the outer
// frames' tone to the center one, and find the per-frame translation that
// locks a chosen pivot point — the parallax left everywhere else is what
// makes the 1-2-3-2 loop read as 3D.

export interface Pivot {
  /** Fractions of a single sub-frame, 0..1. */
  x: number;
  y: number;
}

export interface Shift {
  dx: number;
  dy: number;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Gray {
  g: Float32Array;
  w: number;
  h: number;
}

/** The three sub-frame rects: equal thirds, inset at the seams. */
export function frameRects(img: ImageBitmap, insetFrac: number): Rect[] {
  const w3 = Math.floor(img.width / 3);
  const pad = Math.round(w3 * insetFrac);
  return [0, 1, 2].map((i) => ({ x: i * w3 + pad, y: 0, w: w3 - 2 * pad, h: img.height }));
}

/** Grayscale copy of a source rect, scaled to targetW wide. */
function grayPatch(img: ImageBitmap, rect: Rect, targetW: number): Gray {
  const h = Math.max(1, Math.round(rect.h * (targetW / rect.w)));
  const c = new OffscreenCanvas(targetW, h);
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(img, rect.x, rect.y, rect.w, rect.h, 0, 0, targetW, h);
  const d = ctx.getImageData(0, 0, targetW, h).data;
  const g = new Float32Array(targetW * h);
  for (let i = 0; i < g.length; i++) {
    g[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
  }
  return { g, w: targetW, h };
}

/**
 * Zero-mean NCC template match around the pivot: the (dx, dy) that best
 * moves `frm` onto `ref`, searching ±radius around (initDx, initDy).
 */
function nccShift(
  ref: Gray,
  frm: Gray,
  pivot: Pivot,
  radius: number,
  initDx: number,
  initDy: number
): Shift {
  const pw = Math.round(ref.w * 0.35);
  const ph = Math.round(ref.h * 0.35);
  const cx = Math.round(ref.w * pivot.x);
  const cy = Math.round(ref.h * pivot.y);
  const x0 = Math.min(Math.max(0, cx - (pw >> 1)), ref.w - pw);
  const y0 = Math.min(Math.max(0, cy - (ph >> 1)), ref.h - ph);

  const tpl = new Float32Array(pw * ph);
  let tMean = 0;
  for (let y = 0; y < ph; y++) {
    for (let x = 0; x < pw; x++) {
      tMean += tpl[y * pw + x] = ref.g[(y0 + y) * ref.w + x0 + x];
    }
  }
  tMean /= tpl.length;
  let tNorm = 0;
  for (let i = 0; i < tpl.length; i++) {
    tpl[i] -= tMean;
    tNorm += tpl[i] * tpl[i];
  }
  tNorm = Math.sqrt(tNorm) || 1;

  let best = -2;
  let bestDx = initDx;
  let bestDy = initDy;
  for (let dy = initDy - radius; dy <= initDy + radius; dy++) {
    const fy = y0 - dy;
    if (fy < 0 || fy + ph > frm.h) continue;
    for (let dx = initDx - radius; dx <= initDx + radius; dx++) {
      const fx = x0 - dx;
      if (fx < 0 || fx + pw > frm.w) continue;
      let mean = 0;
      for (let y = 0; y < ph; y++) {
        const row = (fy + y) * frm.w + fx;
        for (let x = 0; x < pw; x++) mean += frm.g[row + x];
      }
      mean /= tpl.length;
      let dot = 0;
      let norm = 0;
      for (let y = 0; y < ph; y++) {
        const row = (fy + y) * frm.w + fx;
        for (let x = 0; x < pw; x++) {
          const v = frm.g[row + x] - mean;
          dot += v * tpl[y * pw + x];
          norm += v * v;
        }
      }
      const score = dot / ((Math.sqrt(norm) || 1) * tNorm);
      if (score > best) {
        best = score;
        bestDx = dx;
        bestDy = dy;
      }
    }
  }
  return { dx: bestDx, dy: bestDy };
}

/**
 * Coarse-to-fine alignment of one frame onto the reference. Template
 * matching rather than gradient methods: the sub-frames sit ~100–200px
 * apart at full resolution, far beyond any gradient method's basin.
 */
function alignShift(img: ImageBitmap, refRect: Rect, frmRect: Rect, pivot: Pivot): Shift {
  const levels = [200, 600, 1200];
  let dx = 0;
  let dy = 0;
  for (let li = 0; li < levels.length; li++) {
    const w = levels[li];
    const ref = grayPatch(img, refRect, w);
    const frm = grayPatch(img, frmRect, w);
    const radius = li === 0 ? Math.round(w * 0.14) : 3;
    ({ dx, dy } = nccShift(ref, frm, pivot, radius, dx, dy));
    if (li < levels.length - 1) {
      const s = levels[li + 1] / w;
      dx = Math.round(dx * s);
      dy = Math.round(dy * s);
    }
  }
  const scale = refRect.w / levels[levels.length - 1];
  return { dx: dx * scale, dy: dy * scale };
}

export function alignShifts(img: ImageBitmap, rects: Rect[], pivot: Pivot): Shift[] {
  return [
    alignShift(img, rects[1], rects[0], pivot),
    { dx: 0, dy: 0 },
    alignShift(img, rects[1], rects[2], pivot),
  ];
}

/** Region (in frame coordinates) valid in every shifted frame. */
export function commonCrop(rects: Rect[], shifts: Shift[]) {
  const fw = rects[0].w;
  const fh = rects[0].h;
  const left = Math.ceil(Math.max(0, ...shifts.map((s) => s.dx)));
  const right = fw + Math.floor(Math.min(0, ...shifts.map((s) => s.dx)));
  const top = Math.ceil(Math.max(0, ...shifts.map((s) => s.dy)));
  const bottom = fh + Math.floor(Math.min(0, ...shifts.map((s) => s.dy)));
  return { left, top, right, bottom, cw: right - left, ch: bottom - top };
}

/** Render the three aligned frames, cropped to their common area. */
export function renderFrames(
  img: ImageBitmap,
  rects: Rect[],
  shifts: Shift[],
  maxWidth: number,
  matchTone: boolean
): OffscreenCanvas[] {
  const { left, top, cw, ch } = commonCrop(rects, shifts);
  const outW = Math.min(maxWidth, cw);
  let ow = outW;
  let oh = Math.round(ch * (outW / cw));
  ow -= ow % 2; // even dimensions for video codecs
  oh -= oh % 2;

  const frames = rects.map((rect, i) => {
    const c = new OffscreenCanvas(ow, oh);
    const ctx = c.getContext('2d', { willReadFrequently: true })!;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(
      img,
      rect.x + left - shifts[i].dx,
      rect.y + top - shifts[i].dy,
      cw,
      ch,
      0,
      0,
      ow,
      oh
    );
    return c;
  });

  if (matchTone) matchFrameTone(frames);
  return frames;
}

/**
 * Match the outer frames' per-channel mean/std to the center frame
 * (the outer lenses usually render slightly darker/warmer), measured on
 * the middle 50% of the image.
 */
function matchFrameTone(frames: OffscreenCanvas[]) {
  const stats = (c: OffscreenCanvas) => {
    const w = c.width;
    const h = c.height;
    const d = c.getContext('2d')!.getImageData(w >> 2, h >> 2, w >> 1, h >> 1).data;
    const mean = [0, 0, 0];
    const sd = [0, 0, 0];
    const n = d.length / 4;
    for (let i = 0; i < d.length; i += 4) {
      mean[0] += d[i];
      mean[1] += d[i + 1];
      mean[2] += d[i + 2];
    }
    for (let ch = 0; ch < 3; ch++) mean[ch] /= n;
    for (let i = 0; i < d.length; i += 4) {
      for (let ch = 0; ch < 3; ch++) {
        const v = d[i + ch] - mean[ch];
        sd[ch] += v * v;
      }
    }
    for (let ch = 0; ch < 3; ch++) sd[ch] = Math.sqrt(sd[ch] / n) || 1;
    return { mean, sd };
  };

  const ref = stats(frames[1]);
  for (const i of [0, 2]) {
    const c = frames[i];
    const s = stats(c);
    const ctx = c.getContext('2d')!;
    const img = ctx.getImageData(0, 0, c.width, c.height);
    const d = img.data;
    const gain = [0, 1, 2].map((ch) => ref.sd[ch] / s.sd[ch]);
    const off = [0, 1, 2].map((ch) => ref.mean[ch] - s.mean[ch] * gain[ch]);
    for (let p = 0; p < d.length; p += 4) {
      d[p] = d[p] * gain[0] + off[0];
      d[p + 1] = d[p + 1] * gain[1] + off[1];
      d[p + 2] = d[p + 2] * gain[2] + off[2];
    }
    ctx.putImageData(img, 0, 0);
  }
}
