import { useEffect, useRef, useState } from 'react';
import { Download, ImagePlus, Loader2, Minus, Plus, Zap } from 'lucide-react';
import ImageDropZone from '@/components/ImageDropZone';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { DualSlider } from '@/components/ui/dual-slider';
import { Switch } from '@/components/ui/switch';

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
  // null = still detecting, true/false = known result.
  const [gpuAvailable, setGpuAvailable] = useState<boolean | null>(null);
  const [useGPU, setUseGPU] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const requestIdRef = useRef(0);
  const remapReqIdRef = useRef(0);
  // Serializes GPU remap apply calls — the persistent context has a single
  // read buffer, so concurrent mapAsync calls would race on it.
  const gpuApplyChainRef = useRef<Promise<unknown>>(Promise.resolve());
  const useGPURef = useRef(false);
  // Tracks whether the persistent GPU remap context has been seeded with
  // the current source. Reset on new image or when WebGPU is toggled off.
  const gpuSourceReadyRef = useRef(false);

  // Cached pieces of the pipeline.
  const wasmRef = useRef<typeof import('frontend-rs') | null>(null);
  const orderRef = useRef<Uint32Array | null>(null);
  // Source pixels with the current L/C remap baked in. Recomputed only when
  // the L or C ranges change, so resize-drag is just a seam-carve render.
  const remappedSourceRef = useRef<Uint8Array | null>(null);
  const sourceRef = useRef<SourceState | null>(null);
  // Combined L (101 bins) + chroma (129 bins) histogram for the current
  // source, packed as one Uint32Array of length 230. Used to display how
  // many pixels would clip under the current L/C remap settings.
  const lcHistRef = useRef<Uint32Array | null>(null);
  const directionRef = useRef<Direction>('width');
  const targetSizeRef = useRef(0);
  const lRangeRef = useRef<[number, number]>([0, 100]);
  const cRangeRef = useRef<[number, number]>([0, 100]);
  const hueRef = useRef(0);

  // Detect WebGPU + load wasm.
  useEffect(() => {
    (async () => {
      const wasm = await import('frontend-rs');
      wasmRef.current = wasm;
      let available = false;
      try {
        if (wasm.is_webgpu_available()) {
          const adapter = await navigator.gpu?.requestAdapter();
          if (adapter) {
            available = true;
            setUseGPU(true);
            useGPURef.current = true;
          }
        }
      } catch {
        /* no gpu */
      }
      setGpuAvailable(available);
    })();
    return () => {
      workerRef.current?.terminate();
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

  // Ensure the persistent GPU remap context has the current source uploaded.
  // No-op when GPU is disabled or already seeded.
  const ensureGpuSource = async () => {
    const wasm = wasmRef.current;
    const src = sourceRef.current;
    if (!wasm || !src) return;
    if (!useGPURef.current || gpuSourceReadyRef.current) return;
    try {
      await wasm.remap_lab_gpu_set_source(src.data, src.w, src.h);
      gpuSourceReadyRef.current = true;
    } catch (err) {
      console.warn('GPU remap set_source failed, falling back to CPU:', err);
      gpuSourceReadyRef.current = false;
    }
  };

  // Recompute the remapped-source cache. Cheap when ranges are identity
  // (we just point at the original source data). Called when the source
  // loads or when the L/C ranges change. Uses GPU when available; otherwise
  // falls back to the CPU `remap_lab` path. Stale results are dropped via
  // a request id so out-of-order GPU promises can't clobber a newer state.
  const refreshRemappedSource = async () => {
    const wasm = wasmRef.current;
    const src = sourceRef.current;
    if (!wasm || !src) return;
    const [lMin, lMax] = lRangeRef.current;
    const [cMin, cMax] = cRangeRef.current;
    const hueDeg = hueRef.current;

    const id = ++remapReqIdRef.current;

    if (lMin === 0 && lMax === 100 && cMin === 0 && cMax === 100 && hueDeg === 0) {
      if (id !== remapReqIdRef.current) return;
      remappedSourceRef.current = src.data;
      return;
    }

    if (useGPURef.current && gpuSourceReadyRef.current) {
      // Wait for any in-flight apply to finish before starting a new one,
      // since the GPU context's read buffer can only be mapped once at a time.
      const prev = gpuApplyChainRef.current;
      const next = (async () => {
        try {
          await prev;
        } catch {
          /* don't propagate previous failures */
        }
        // A newer request may have arrived while we were waiting — bail.
        if (id !== remapReqIdRef.current) return null;
        return wasm.remap_lab_gpu_apply(lMin, lMax, cMin, cMax, hueDeg);
      })();
      gpuApplyChainRef.current = next;

      try {
        const result = await next;
        if (id !== remapReqIdRef.current) return;
        if (result) {
          remappedSourceRef.current = result;
          return;
        }
      } catch (err) {
        console.warn('GPU remap apply failed, falling back to CPU:', err);
        gpuSourceReadyRef.current = false;
      }
    }

    const result = wasm.remap_lab(src.data, src.w, src.h, lMin, lMax, cMin, cMax, hueDeg);
    if (id !== remapReqIdRef.current) return;
    remappedSourceRef.current = result;
  };

  // Re-render the carved image (using the cached remapped source) to
  // the canvas. Safe to call from any callback — pulls everything from refs.
  const renderPreview = () => {
    const wasm = wasmRef.current;
    const src = sourceRef.current;
    const order = orderRef.current;
    const remapped = remappedSourceRef.current;
    const canvas = canvasRef.current;
    if (!wasm || !src || !order || !remapped || !canvas) return;

    const dir = directionRef.current;
    const target = targetSizeRef.current;
    const dirNum = dir === 'width' ? 0 : 1;

    const carved = wasm.render_seam_carved(
      remapped,
      order,
      src.w,
      src.h,
      target,
      dirNum
    );
    const outW = dir === 'width' ? target : src.w;
    const outH = dir === 'height' ? target : src.h;

    canvas.width = outW;
    canvas.height = outH;
    // The canvas itself must be tagged as Display P3 so the bytes we
    // putImageData below are interpreted in the wide gamut.
    const ctx = canvas.getContext('2d', { colorSpace: 'display-p3' });
    if (!ctx) return;
    ctx.putImageData(
      new ImageData(new Uint8ClampedArray(carved), outW, outH, {
        colorSpace: 'display-p3',
      }),
      0,
      0
    );
  };

  // Run seam-order precompute via worker.
  const runPrecompute = (src: SourceState, dir: Direction) => {
    setIsPrecomputing(true);
    setReady(false);

    const id = ++requestIdRef.current;
    const dirNum = dir === 'width' ? 0 : 1;

    const worker = getWorker();
    worker.onmessage = async (e: MessageEvent) => {
      const { order, requestId } = e.data;
      if (requestId !== requestIdRef.current) return;

      orderRef.current = order;
      const max = dir === 'width' ? src.w : src.h;
      // Default target = original size (no carving applied yet).
      targetSizeRef.current = max;
      setTargetSize(max);
      setIsPrecomputing(false);
      setReady(true);
      await ensureGpuSource();
      await refreshRemappedSource();
      requestAnimationFrame(renderPreview);
    };

    worker.postMessage({
      imageData: src.data,
      width: src.w,
      height: src.h,
      direction: dirNum,
      requestId: id,
      useGPU: useGPURef.current,
    });
  };

  const handleImageSelect = (imageUrl: string) => {
    const img = new Image();
    img.onload = () => {
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
      // New image → GPU context needs to re-upload source.
      gpuSourceReadyRef.current = false;
      // Compute the L+chroma histogram once so the clipping readouts are live.
      if (wasmRef.current) {
        lcHistRef.current = wasmRef.current.compute_lc_histogram(src.data);
      }
      runPrecompute(src, directionRef.current);
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
    renderPreview();
  };

  const handleLRangeChange = (next: [number, number]) => {
    setLRange(next);
    lRangeRef.current = next;
    if (ready) {
      refreshRemappedSource().then(renderPreview);
    }
  };

  const handleCRangeChange = (next: [number, number]) => {
    setCRange(next);
    cRangeRef.current = next;
    if (ready) {
      refreshRemappedSource().then(renderPreview);
    }
  };

  const handleHueChange = (next: number) => {
    setHue(next);
    hueRef.current = next;
    if (ready) {
      refreshRemappedSource().then(renderPreview);
    }
  };

  // Bump exposure by N stops. Composes with the current L transform:
  // a stop is a 2× factor in *linear* light, which corresponds to a 2^(1/3)
  // factor in L*. The remap is affine in L*, so we just multiply both
  // endpoints by k = 2^(stops/3) to apply the same scale to L_out.
  const adjustExposure = (stops: number) => {
    const k = Math.pow(2, stops / 3);
    const next: [number, number] = [lRangeRef.current[0] * k, lRangeRef.current[1] * k];
    // Snap to integer values to keep the slider readout tidy.
    const snapped: [number, number] = [Math.round(next[0]), Math.round(next[1])];
    handleLRangeChange(snapped);
  };

  const handleGpuToggle = async (next: boolean) => {
    setUseGPU(next);
    useGPURef.current = next;
    if (next) {
      // Newly enabled: seed source then refresh.
      gpuSourceReadyRef.current = false;
      await ensureGpuSource();
    } else {
      // Disable: drop the GPU context entirely so resources are freed.
      gpuSourceReadyRef.current = false;
      try {
        wasmRef.current?.remap_lab_gpu_dispose();
      } catch {
        /* ignore */
      }
    }
    if (ready) {
      await refreshRemappedSource();
      renderPreview();
    }
  };

  const handleDownload = () => {
    const wasm = wasmRef.current;
    const canvas = canvasRef.current;
    if (!wasm || !canvas) return;
    // Pull pixels back out of the canvas in P3 and hand them to the wasm
    // PNG encoder, which embeds the Display P3 ICC profile so other
    // viewers render the same wide-gamut colors.
    const ctx = canvas.getContext('2d', { colorSpace: 'display-p3' });
    if (!ctx) return;
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height, {
      colorSpace: 'display-p3',
    });
    const png = wasm.encode_png(
      new Uint8Array(imageData.data),
      canvas.width,
      canvas.height
    );
    // Copy into a fresh ArrayBuffer so the Blob owns its bytes (avoids
    // referencing wasm linear memory directly).
    const blob = new Blob([new Uint8Array(png)], { type: 'image/png' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'pictolab.png';
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleNewImage = () => {
    setSource(null);
    sourceRef.current = null;
    orderRef.current = null;
    remappedSourceRef.current = null;
    lcHistRef.current = null;
    gpuSourceReadyRef.current = false;
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

  // Live clipping percentages for the current L remap. Iterates the 101-bin
  // L histogram and tallies bins whose remapped value falls outside [0, 100].
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

  // Approximate chroma clipping. Iterates the 161-bin chroma histogram,
  // applies the same per-pixel mapping as `remap_lab` (input chroma 0..160
  // → output c_min..c_max as a percentage of the 160-unit reference) and
  // counts pixels whose absolute output chroma exceeds 160 — a rough
  // Display P3 practical ceiling. The exact gamut boundary varies with
  // L and hue, so this under-counts in dark/light regions but is a good
  // directional indicator. Also tracks pixels whose output chroma goes
  // negative (hue inversion), which is a perceptually significant
  // boundary even though it isn't strictly "clipping".
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

  return (
    <div className="flex h-full min-h-screen bg-background text-foreground">
      {/* ── Main preview area ──────────────────────────────────────── */}
      <main className="flex flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-border px-6 py-4">
          <div>
            <h1 className="text-lg font-semibold tracking-tight">Pictolab</h1>
            <p className="text-xs text-muted-foreground">
              Content-aware resize · LAB color remap
            </p>
          </div>
          {source && (
            <Button variant="outline" size="sm" onClick={handleNewImage}>
              <ImagePlus className="mr-1 h-4 w-4" />
              New image
            </Button>
          )}
        </header>

        <div className="flex flex-1 items-center justify-center overflow-auto p-8">
          {!source ? (
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
                  // Cap by viewport height while preserving original aspect.
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
        <div
          className={`mb-4 flex items-center justify-between rounded-md border px-3 py-2 ${
            gpuAvailable === false ? 'bg-muted opacity-60' : 'bg-background'
          }`}
        >
          <div className="flex items-center gap-2">
            <Zap
              className={`h-4 w-4 ${
                gpuAvailable && useGPU ? 'text-primary' : 'text-muted-foreground'
              }`}
            />
            <Label htmlFor="gpu-toggle" className="text-sm font-medium">
              WebGPU acceleration
            </Label>
          </div>
          <Switch
            id="gpu-toggle"
            checked={useGPU}
            onCheckedChange={handleGpuToggle}
            disabled={gpuAvailable !== true || isPrecomputing}
          />
        </div>

        {!source ? (
          <div className="text-sm text-muted-foreground">
            {gpuAvailable === false
              ? 'WebGPU is unavailable on this device — falling back to CPU. Drop an image to start.'
              : 'Drop an image to start editing.'}
          </div>
        ) : (
          <div className="space-y-4">
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
                    <span className="font-mono text-xs text-muted-foreground">
                      {lRange[0]} → {lRange[1]}
                    </span>
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
                    L ranges 0–100 by default. Push past either end to boost/reduce exposure (clips at the sRGB gamut). 100 → 0 inverts; collapse for equalize.
                  </p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Chroma Remap</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Range</Label>
                  <span className="font-mono text-xs text-muted-foreground">
                    {cRange[0]} → {cRange[1]}
                  </span>
                </div>
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

            <Card>
              <CardHeader>
                <CardTitle>Export</CardTitle>
              </CardHeader>
              <CardContent>
                <Button className="w-full" onClick={handleDownload} disabled={!ready}>
                  <Download className="mr-1 h-4 w-4" />
                  Download PNG
                </Button>
              </CardContent>
            </Card>
          </div>
        )}
      </aside>
    </div>
  );
}

export default Editor;
