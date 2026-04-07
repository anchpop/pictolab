import { useEffect, useRef, useState } from 'react';
import { ArrowLeftRight, Download, Eye, ImagePlus, Loader2, Minus, Plus } from 'lucide-react';
import ImageDropZone from '@/components/ImageDropZone';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { DualSlider } from '@/components/ui/dual-slider';

type Direction = 'width' | 'height';

interface SourceState {
  url: string;
  data: Uint8Array;
  w: number;
  h: number;
}

function Editor() {
  const [source, setSource] = useState<SourceState | null>(null);
  const [direction, setDirection] = useState<Direction>('width');
  const [targetSize, setTargetSize] = useState(0);
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
  const sourceRef = useRef<SourceState | null>(null);
  // Combined L (101 bins) + chroma (161 bins) histogram for the current
  // source. Used to display how many pixels would clip under the current
  // L/C remap settings.
  const lcHistRef = useRef<Uint32Array | null>(null);
  const directionRef = useRef<Direction>('width');
  const targetSizeRef = useRef(0);
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
        wasm.gpu_set_source(src.data, src.w, src.h);
      } catch (err) {
        console.error('GPU init/upload failed:', err);
        return;
      }
      runPrecompute(src, directionRef.current);
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

  // Push the current LUT (carve mapping) into the GPU context based on the
  // active seam order, source dims, target size, and direction.
  const refreshLut = () => {
    const wasm = wasmRef.current;
    const src = sourceRef.current;
    const order = orderRef.current;
    if (!wasm || !src || !order) return;
    const dir = directionRef.current;
    const dirNum = dir === 'width' ? 0 : 1;
    const target = targetSizeRef.current;
    const targetW = dir === 'width' ? target : src.w;
    const targetH = dir === 'height' ? target : src.h;
    const lut = wasm.build_carve_lut(order, src.w, src.h, target, dirNum);
    wasm.gpu_set_carve_lut(lut, targetW, targetH);
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

  // Run seam-order precompute via worker.
  const runPrecompute = (src: SourceState, dir: Direction) => {
    setIsPrecomputing(true);
    setReady(false);

    const id = ++requestIdRef.current;
    const dirNum = dir === 'width' ? 0 : 1;

    const worker = getWorker();
    worker.onmessage = (e: MessageEvent) => {
      const { order, requestId } = e.data;
      if (requestId !== requestIdRef.current) return;

      orderRef.current = order;
      const max = dir === 'width' ? src.w : src.h;
      // Default target = original size (no carving applied yet).
      targetSizeRef.current = max;
      setTargetSize(max);
      setIsPrecomputing(false);
      setReady(true);
      refreshLut();
      requestAnimationFrame(render);
    };

    worker.postMessage({
      imageData: src.data,
      width: src.w,
      height: src.h,
      direction: dirNum,
      requestId: id,
      useGPU: true,
    });
  };

  const handleImageSelect = (imageUrl: string) => {
    const wasm = wasmRef.current;
    if (!wasm) return;

    const img = new Image();
    img.onload = async () => {
      const tmp = document.createElement('canvas');
      tmp.width = img.width;
      tmp.height = img.height;
      // Decode straight into Display P3 so wide-gamut sources keep their
      // colors and our pipeline always sees pixels in the same space.
      const ctx = tmp.getContext('2d', { colorSpace: 'display-p3' })!;
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, img.width, img.height, {
        colorSpace: 'display-p3',
      });
      const src: SourceState = {
        url: imageUrl,
        data: new Uint8Array(data.data),
        w: img.width,
        h: img.height,
      };
      setSource(src);
      sourceRef.current = src;

      // Compute the L+chroma histogram once so the clipping readouts are live.
      lcHistRef.current = wasm.compute_lc_histogram(src.data);

      // Initialize GPU context against the canvas (once) and upload source.
      // The canvas element only mounts after `source` is set, so we wait a
      // frame for React to commit before grabbing it.
      // GPU init + precompute kicks off from a useEffect once the canvas
      // element is actually mounted (see below).
    };
    img.src = imageUrl;
  };

  const handleDirectionChange = (dir: Direction) => {
    setDirection(dir);
    directionRef.current = dir;
    if (sourceRef.current) {
      runPrecompute(sourceRef.current, dir);
    }
  };

  const handleTargetChange = (size: number) => {
    if (!sourceRef.current) return;
    const max = directionRef.current === 'width' ? sourceRef.current.w : sourceRef.current.h;
    const clamped = Math.max(1, Math.min(max, Math.round(size)));
    setTargetSize(clamped);
    targetSizeRef.current = clamped;
    refreshLut();
    render();
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

      const dir = directionRef.current;
      const target = targetSizeRef.current;
      const outW = dir === 'width' ? target : src.w;
      const outH = dir === 'height' ? target : src.h;

      let blob: Blob;
      let filename: string;
      if (exportFormat === 'jpeg') {
        // Linear extended Display P3 half-floats → Ultra HDR JPEG.
        // libultrahdr derives the SDR base + gain map from the HDR input.
        const f16 = await wasm.gpu_readback_linear_f16();
        const { encodeUhdr } = await import('@/lib/uhdr');
        const jpeg = await encodeUhdr(f16, outW, outH);
        blob = new Blob([new Uint8Array(jpeg)], { type: 'image/jpeg' });
        filename = 'pictolab.jpg';
      } else {
        // 10-bit BT.2020 PQ readback for HDR AVIF.
        const rgba16 = await wasm.gpu_readback_hdr_pq_u16(10);
        const { encodeAvif, CP_BT2020, TC_PQ, MC_BT2020_NCL } = await import('@/lib/avif-hdr');
        const avif = await encodeAvif(rgba16, outW, outH, {
          bitDepth: 10,
          quality: 60,
          cicpColorPrimaries: CP_BT2020,
          cicpTransferCharacteristics: TC_PQ,
          cicpMatrixCoefficients: MC_BT2020_NCL,
          fullRange: false,
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
    lcHistRef.current = null;
    setReady(false);
    setLRange([0, 100]);
    setCRange([0, 100]);
    setHue(0);
    lRangeRef.current = [0, 100];
    cRangeRef.current = [0, 100];
    hueRef.current = 0;
  };

  const outW = direction === 'width' ? targetSize : source?.w ?? 0;
  const outH = direction === 'height' ? targetSize : source?.h ?? 0;
  const maxSize = direction === 'width' ? source?.w ?? 1 : source?.h ?? 1;

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
              Content-aware resize · OKLab color remap · WebGPU
            </p>
          </div>
          {source && (
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={handleNewImage}>
                <ImagePlus className="mr-1 h-4 w-4" />
                New image
              </Button>
            </div>
          )}
        </header>

        <div className="flex flex-1 items-center justify-center overflow-auto p-8">
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
                    width: direction === 'width' ? `${(targetSize / source.w) * 100}%` : '100%',
                    height: direction === 'height' ? `${(targetSize / source.h) * 100}%` : '100%',
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
                <div className="flex gap-2">
                  <Button
                    variant={direction === 'width' ? 'default' : 'outline'}
                    size="sm"
                    className="flex-1"
                    onClick={() => handleDirectionChange('width')}
                    disabled={isPrecomputing}
                  >
                    Width
                  </Button>
                  <Button
                    variant={direction === 'height' ? 'default' : 'outline'}
                    size="sm"
                    className="flex-1"
                    onClick={() => handleDirectionChange('height')}
                    disabled={isPrecomputing}
                  >
                    Height
                  </Button>
                </div>

                {ready && (
                  <>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label>{direction === 'width' ? 'Width' : 'Height'}</Label>
                        <span className="text-xs text-muted-foreground">{targetSize}px</span>
                      </div>
                      <Slider
                        min={1}
                        max={maxSize}
                        step={1}
                        value={[targetSize]}
                        onValueChange={(v) => handleTargetChange(v[0])}
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <Label className="text-xs">W</Label>
                        <Input
                          type="number"
                          value={outW}
                          min={1}
                          max={source.w}
                          disabled={direction !== 'width'}
                          onChange={(e) => {
                            if (direction === 'width') handleTargetChange(Number(e.target.value));
                          }}
                        />
                      </div>
                      <div>
                        <Label className="text-xs">H</Label>
                        <Input
                          type="number"
                          value={outH}
                          min={1}
                          max={source.h}
                          disabled={direction !== 'height'}
                          onChange={(e) => {
                            if (direction === 'height') handleTargetChange(Number(e.target.value));
                          }}
                        />
                      </div>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Lightness Remap</CardTitle>
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
                <CardTitle>Chroma Remap</CardTitle>
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
              <div className="grid grid-cols-2 gap-2">
                <Button
                  variant={exportFormat === 'jpeg' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setExportFormat('jpeg')}
                  title="Ultra HDR JPEG (SDR base + gain map). Widely supported."
                >
                  JPEG
                </Button>
                <Button
                  variant={exportFormat === 'avif' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setExportFormat('avif')}
                  title="10-bit BT.2020 PQ AVIF. Smaller, real HDR."
                >
                  AVIF
                </Button>
              </div>
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
