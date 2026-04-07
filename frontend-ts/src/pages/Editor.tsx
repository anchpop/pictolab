import { useEffect, useRef, useState } from 'react';
import { ArrowLeftRight, Download, Eye, ImagePlus, Loader2, Minus, Plus } from 'lucide-react';
import ImageDropZone from '@/components/ImageDropZone';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { DualSlider } from '@/components/ui/dual-slider';
import { Segmented } from '@/components/ui/segmented';
import { Collapsible } from '@/components/ui/collapsible';

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

// Scan an 8-bit RGBA buffer for any non-opaque pixel. Cheap O(n) walk.
function bufferHasAlpha(rgba: Uint8Array): boolean {
  for (let i = 3; i < rgba.length; i += 4) {
    if (rgba[i] !== 255) return true;
  }
  return false;
}

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

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const requestIdRef = useRef(0);
  // Tracks whether the persistent GPU context has been initialized against
  // the current canvas element.
  const gpuInitedRef = useRef(false);

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

  // After the source state is set and React mounts the canvas, init the
  // GPU context (if needed), upload the source, and kick off precompute.
  useEffect(() => {
    const wasm = wasmRef.current;
    const src = source;
    const c = canvasRef.current;
    if (!wasm || !src || !c) return;
    let cancelled = false;
    (async () => {
      c.width = src.w;
      c.height = src.h;
      try {
        if (!gpuInitedRef.current) {
          await wasm.gpu_init(c);
          gpuInitedRef.current = true;
        }
        if (cancelled) return;
        if (src.hdr) {
          wasm.gpu_set_source_linear_f16(src.data, src.w, src.h);
        } else {
          wasm.gpu_set_source(src.data, src.w, src.h);
        }
      } catch (err) {
        console.error('GPU init/upload failed:', err);
        return;
      }
      targetWRef.current = src.w;
      targetHRef.current = src.h;
      setTargetW(src.w);
      setTargetH(src.h);
      // Carve is the default mode and needs the seam precompute before
      // build_carve_lut works. Show the source at its native size as a
      // first paint (identity Lanczos = the unmodified image, not a
      // resize fallback) so the canvas isn't blank while we wait. All
      // resize controls are gated on isPrecomputing.
      if (resizeModeRef.current === 'carve') {
        wasm.gpu_set_squish_dims(src.w, src.h);
        setReady(true);
        requestAnimationFrame(render);
        runPrecompute(src, directionRef.current, () => {
          applyResize();
          render();
        });
      } else {
        applyResize();
        setReady(true);
        requestAnimationFrame(render);
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source]);

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

  // Issue a render with the current L/C/hue params. Cheap; safe to call
  // from any callback.
  const render = () => {
    const wasm = wasmRef.current;
    if (!wasm || !gpuInitedRef.current) return;
    const [lMin, lMax] = lRangeRef.current;
    const [cMin, cMax] = cRangeRef.current;
    const t = performance.now() / 1000;
    wasm.gpu_render(lMin, lMax, cMin, cMax, hueRef.current, showHdrRef.current ? 1 : 0, t);
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

    // Try the HDR path first. Returns null for SDR / unsupported formats
    // so we can fall through to the canvas path.
    try {
      const { decodeHdr } = await import('@/lib/hdr-decode');
      const decoded = await decodeHdr(buf);
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
    const img = new Image();
    img.onload = async () => {
      const tmp = document.createElement('canvas');
      tmp.width = img.width;
      tmp.height = img.height;
      const ctx = tmp.getContext('2d', { colorSpace: 'display-p3' })!;
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, img.width, img.height, {
        colorSpace: 'display-p3',
      });
      const sdrData = new Uint8Array(data.data);
      const src: SourceState = {
        url: imageUrl,
        data: sdrData,
        sdrData,
        w: img.width,
        h: img.height,
        hdr: false,
        hasAlpha: bufferHasAlpha(sdrData),
      };
      setSource(src);
      sourceRef.current = src;

      // Compute the L+chroma histogram once so the clipping readouts are live.
      lcHistRef.current = wasm.compute_lc_histogram(src.sdrData);
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(blob);
  };

  const handleDirectionChange = (dir: Direction) => {
    setDirection(dir);
    directionRef.current = dir;
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
    if (resizeModeRef.current === 'carve' && orderDirectionRef.current !== dir) {
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
    targetWRef.current = w;
    targetHRef.current = h;
    setTargetW(w);
    setTargetH(h);
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
    if (mode === 'carve' && orderDirectionRef.current !== directionRef.current) {
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
  const [exportFormat, setExportFormat] = useState<ExportFormat>('jpeg');
  const [downloading, setDownloading] = useState(false);

  const handleDownload = async () => {
    const wasm = wasmRef.current;
    const src = sourceRef.current;
    if (!wasm || !src) return;
    setDownloading(true);
    try {
      // Make sure the texture has the latest params (and turn off the
      // HDR-view marker pixels for the export).
      const [lMin, lMax] = lRangeRef.current;
      const [cMin, cMax] = cRangeRef.current;
      wasm.gpu_render(lMin, lMax, cMin, cMax, hueRef.current, 0, 0);

      const outW = targetWRef.current;
      const outH = targetHRef.current;

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
        });
        blob = new Blob([new Uint8Array(jpeg)], { type: 'image/jpeg' });
        filename = 'pictolab.jpg';
      } else {
        // 10-bit BT.2020 PQ readback for HDR AVIF.
        const rgba16 = await wasm.gpu_readback_hdr_pq_u16(10);
        const { encodeAvif, CP_BT2020, TC_PQ, MC_BT2020_NCL } = await import('@/lib/avif-hdr');
        const avif = await encodeAvif(rgba16, outW, outH, {
          bitDepth: 10,
          // Higher than the libavif default (~60) — for HDR master output
          // we want chroma transitions in saturated highlights to stay
          // clean, and 60 leaves visible banding/blocking on bright sky.
          quality: 80,
          cicpColorPrimaries: CP_BT2020,
          cicpTransferCharacteristics: TC_PQ,
          cicpMatrixCoefficients: MC_BT2020_NCL,
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
      // Restore the HDR view if it was on.
      if (showHdrRef.current) render();
    }
  };

  const handleNewImage = () => {
    setSource(null);
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

  return (
    <div className="flex h-full min-h-screen bg-background text-foreground">
      {/* ── Main preview area ──────────────────────────────────────── */}
      <main className="flex flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-border px-6 py-4">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">Pictolab</h1>
            <p className="text-xs text-muted-foreground">
              Lanczos / content-aware resize · OKLab color remap · WebGPU
            </p>
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

        <div
          className="flex flex-1 items-center justify-center overflow-auto p-8"
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
              browser with WebGPU enabled (recent Chrome, Edge, or Safari Technology
              Preview).
            </div>
          ) : !source ? (
            <div className="w-full max-w-2xl">
              <ImageDropZone onImageSelect={handleImageSelect} />
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              {isPrecomputing && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Precomputing seam order…
                </div>
              )}
              <div
                className="relative max-h-[75vh] rounded-lg border border-border bg-card shadow-sm"
                style={{
                  aspectRatio: `${source.w} / ${source.h}`,
                  height: 'min(75vh, 80vw)',
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
              </div>
              <p className="text-xs text-muted-foreground">
                {outW} × {outH}
              </p>
            </div>
          )}
        </div>
      </main>

      {/* ── Right sidebar ──────────────────────────────────────────── */}
      <aside className="w-80 shrink-0 overflow-y-auto border-l border-border bg-card p-4">
        {!source ? (
          <div className="text-sm text-muted-foreground">
            {gpuMissing
              ? 'WebGPU unavailable.'
              : 'Drop an image to start editing.'}
          </div>
        ) : (
          <div className="space-y-4">
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

            <Card>
              <CardHeader>
                <CardTitle>Resize</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
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

                <Collapsible
                  label="Aspect ratio"
                  meta={
                    aspect !== 'free' && (
                      <span className="font-mono text-[10px] text-foreground">
                        {ASPECT_OPTIONS.find((o) => o.value === aspect)?.label}
                      </span>
                    )
                  }
                >
                  <div className="flex flex-wrap gap-1">
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
                </Collapsible>
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
                          value={[targetW]}
                          disabled={isPrecomputing}
                          onValueChange={(v) => handleTargetWChange(v[0])}
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
                          value={[targetH]}
                          disabled={isPrecomputing}
                          onValueChange={(v) => handleTargetHChange(v[0])}
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
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Lightness</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Range</Label>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-6 px-2"
                      onClick={() => handleLRangeChange([lRangeRef.current[1], lRangeRef.current[0]])}
                      title="Invert (swap handles)"
                    >
                      <ArrowLeftRight className="h-3 w-3" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-6 px-2"
                      onClick={() => adjustExposure(-0.5)}
                      title="−½ stop"
                    >
                      <Minus className="h-3 w-3" />
                      ½ EV
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-6 px-2"
                      onClick={() => adjustExposure(0.5)}
                      title="+½ stop"
                    >
                      <Plus className="h-3 w-3" />
                      ½ EV
                    </Button>
                  </div>
                </div>
                <DualSlider
                  min={-50}
                  max={150}
                  step={1}
                  value={lRange}
                  onValueChange={handleLRangeChange}
                  safeRange={[0, 100]}
                />
                {lClipping && (lClipping.dark > 0 || lClipping.light > 0) ? (
                  <p className="text-[11px] leading-snug text-destructive">
                    Clipping: {lClipping.dark.toFixed(1)}% shadows · {lClipping.light.toFixed(1)}% highlights
                  </p>
                ) : (
                  <p className="text-[11px] leading-snug text-muted-foreground">
                    L ranges 0–100 by default. Push past either end to boost/reduce exposure. 100 → 0 inverts; collapse for equalize.
                  </p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Chroma</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Label className="text-xs">Range</Label>
                <DualSlider
                  min={-50}
                  max={150}
                  step={1}
                  value={cRange}
                  onValueChange={handleCRangeChange}
                  safeRange={[0, 100]}
                />
                {cClipping && (cClipping.boosted > 0 || cClipping.inverted > 0) ? (
                  <p className="text-[11px] leading-snug text-destructive">
                    {cClipping.boosted > 0 && <>Clipping: {cClipping.boosted.toFixed(1)}% boosted past gamut</>}
                    {cClipping.boosted > 0 && cClipping.inverted > 0 && <> · </>}
                    {cClipping.inverted > 0 && <>{cClipping.inverted.toFixed(1)}% hue-inverted</>}
                  </p>
                ) : (
                  <p className="text-[11px] leading-snug text-muted-foreground">
                    Same semantic as L. Identity is 0 → 100. Push past 100 to boost saturation, below 0 to invert hue.
                  </p>
                )}

                <div className="space-y-2 pt-2">
                  <div className="flex items-center justify-between">
                    <Label className="text-xs">Hue rotation</Label>
                    <span className="font-mono text-xs text-muted-foreground">
                      {hue > 0 ? '+' : ''}
                      {hue}°
                    </span>
                  </div>
                  <Slider
                    min={-180}
                    max={180}
                    step={1}
                    value={[hue]}
                    onValueChange={(v) => handleHueChange(v[0])}
                  />
                </div>
              </CardContent>
            </Card>

            <div className="space-y-2">
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
                <p className="text-[11px] leading-snug text-amber-500">
                  This image has transparency, which JPEG can't store —
                  transparent pixels will be flattened to a background
                  color. Use AVIF to preserve alpha.
                </p>
              )}
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
          </div>
        )}
      </aside>
    </div>
  );
}

export default Editor;
