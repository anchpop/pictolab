import { useCallback, useEffect, useRef, useState } from 'react';
import { GIFEncoder, quantize, applyPalette } from 'gifenc';
import { Clapperboard, Download, ImagePlus, Loader2 } from 'lucide-react';
import ImageDropZone from '@/components/ImageDropZone';
import { Button } from '@/components/ui/button';
import { CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import {
  alignShifts,
  commonCrop,
  frameRects,
  renderFrames,
  type Pivot,
  type Shift,
} from '@/lib/wigglegram';

const SEQ = [0, 1, 2, 1]; // bounce loop

function Wiggle() {
  const [source, setSource] = useState<ImageBitmap | null>(null);
  const [frames, setFrames] = useState<OffscreenCanvas[] | null>(null);
  const [shifts, setShifts] = useState<Shift[] | null>(null);
  const [pivot, setPivot] = useState<Pivot>({ x: 0.5, y: 0.45 });
  const [fps, setFps] = useState(11);
  const [inset, setInset] = useState(0);
  const [tone, setTone] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Recompute alignment (expensive-ish, ~0.5s) and re-render the frames.
  // Deferred through setTimeout so the busy indicator paints first.
  const reprocess = useCallback(
    (img: ImageBitmap, piv: Pivot, insetPct: number, toneOn: boolean, realign: boolean, prevShifts: Shift[] | null) => {
      setBusy(realign ? 'Aligning frames…' : 'Rendering…');
      setTimeout(() => {
        try {
          const rects = frameRects(img, insetPct / 100);
          const s = realign || !prevShifts ? alignShifts(img, rects, piv) : prevShifts;
          setShifts(s);
          setFrames(renderFrames(img, rects, s, toneOn));
          setStatus(null);
        } catch {
          setFrames(null);
          setShifts(null);
          setStatus("Couldn't process that photo — is it a 3-frame wigglegram shot?");
        } finally {
          setBusy(null);
        }
      }, 30);
    },
    []
  );

  const handleImageSelect = async (imageUrl: string) => {
    setBusy('Loading photo…');
    try {
      const blob = await (await fetch(imageUrl)).blob();
      const img = await createImageBitmap(blob, { imageOrientation: 'from-image' });
      const piv = { x: 0.5, y: 0.45 };
      setSource(img);
      setPivot(piv);
      setStatus(null);
      reprocess(img, piv, inset, tone, true, null);
    } catch {
      setBusy(null);
      setStatus("Couldn't read that image.");
    }
  };

  // Animate the preview.
  useEffect(() => {
    if (!frames || !canvasRef.current) return;
    const canvas = canvasRef.current;
    canvas.width = frames[0].width;
    canvas.height = frames[0].height;
    const ctx = canvas.getContext('2d')!;
    let i = 0;
    ctx.drawImage(frames[SEQ[0]], 0, 0);
    const timer = setInterval(() => {
      ctx.drawImage(frames[SEQ[++i % SEQ.length]], 0, 0);
    }, 1000 / fps);
    return () => clearInterval(timer);
  }, [frames, fps]);

  const handlePreviewClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!source || !shifts || busy) return;
    const rects = frameRects(source, inset / 100);
    const { left, top, cw, ch } = commonCrop(rects, shifts);
    const box = e.currentTarget.getBoundingClientRect();
    const cx = (e.clientX - box.left) / box.width;
    const cy = (e.clientY - box.top) / box.height;
    const piv = {
      x: (left + cx * cw) / rects[0].w,
      y: (top + cy * ch) / rects[0].h,
    };
    setPivot(piv);
    reprocess(source, piv, inset, tone, true, null);
  };

  // Pivot dot position within the cropped preview, as fractions.
  const pivotFrac = (() => {
    if (!source || !shifts) return null;
    const rects = frameRects(source, inset / 100);
    const { left, top, cw, ch } = commonCrop(rects, shifts);
    const fx = (pivot.x * rects[0].w - left) / cw;
    const fy = (pivot.y * rects[0].h - top) / ch;
    if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return null;
    return { fx, fy };
  })();

  const download = (blob: Blob, name: string) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 30_000);
  };

  const handleDownloadGif = () => {
    if (!frames || busy) return;
    setBusy('Encoding GIF…');
    setTimeout(() => {
      try {
        const delay = Math.round(1000 / fps);
        const gif = GIFEncoder();
        const frameData = frames.map((c) =>
          c.getContext('2d')!.getImageData(0, 0, c.width, c.height).data
        );
        // One palette from the center frame for all frames — avoids color flicker.
        const palette = quantize(frameData[1], 256);
        const w = frames[0].width;
        const h = frames[0].height;
        SEQ.forEach((fi, i) => {
          const index = applyPalette(frameData[fi], palette);
          gif.writeFrame(index, w, h, { palette: i === 0 ? palette : undefined, delay });
        });
        gif.finish();
        download(new Blob([gif.bytes().slice().buffer], { type: 'image/gif' }), 'wigglegram.gif');
        setStatus('GIF saved.');
      } catch {
        setStatus('GIF encoding failed — try trimming the seams or a smaller photo.');
      } finally {
        setBusy(null);
      }
    }, 30);
  };

  const handleDownloadVideo = () => {
    if (!frames || busy) return;
    // Ask for H.264 explicitly: Chrome's bare 'video/mp4' picks its default
    // codec (VP9/AV1 in an MP4 container), which QuickTime can't play.
    // avc1.640033 = High profile level 5.1, needed for full-res frames.
    const mime = [
      'video/mp4;codecs=avc1.640033',
      'video/mp4;codecs=avc1',
      'video/mp4',
      'video/webm;codecs=vp9',
      'video/webm',
    ].find((m) => 'MediaRecorder' in window && MediaRecorder.isTypeSupported(m));
    if (!mime) {
      setStatus("Video recording isn't supported in this browser — use the GIF.");
      return;
    }
    setBusy('Recording video…');

    const c = document.createElement('canvas');
    c.width = frames[0].width;
    c.height = frames[0].height;
    const ctx = c.getContext('2d')!;

    const timerRef = { id: undefined as ReturnType<typeof setInterval> | undefined };
    let failed = false;
    const fail = () => {
      failed = true;
      if (timerRef.id) clearInterval(timerRef.id);
      setStatus('Video encoding failed at this resolution — use the GIF.');
      setBusy(null);
    };

    let rec: MediaRecorder;
    try {
      rec = new MediaRecorder(c.captureStream(), {
        mimeType: mime,
        // ~0.2 bits/px/frame, capped — full-res frames need far more than a
        // fixed 12 Mbps to avoid smearing.
        videoBitsPerSecond: Math.min(40_000_000, Math.round(c.width * c.height * fps * 0.2)),
      });
    } catch {
      fail();
      return;
    }
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => chunks.push(e.data);
    rec.onerror = fail;
    rec.onstop = () => {
      if (failed) return;
      if (chunks.length === 0) {
        fail();
        return;
      }
      const ext = mime.startsWith('video/mp4') ? 'mp4' : 'webm';
      download(new Blob(chunks, { type: mime }), `wigglegram.${ext}`);
      setStatus('Video saved.');
      setBusy(null);
    };

    let i = 0;
    ctx.drawImage(frames[SEQ[0]], 0, 0);
    try {
      rec.start();
    } catch {
      fail();
      return;
    }
    timerRef.id = setInterval(() => {
      ctx.drawImage(frames[SEQ[++i % SEQ.length]], 0, 0);
      if (i >= fps * 4) {
        clearInterval(timerRef.id);
        if (rec.state !== 'inactive') rec.stop();
      }
    }, 1000 / fps);
  };

  const handleNewImage = () => {
    setSource(null);
    setFrames(null);
    setShifts(null);
    setStatus(null);
  };

  return (
    <div className="flex h-full min-h-screen flex-col bg-background text-foreground lg:h-screen lg:overflow-hidden">
      <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-3 sm:px-6 sm:py-4">
        <div className="flex items-baseline gap-2">
          <h1 className="text-lg font-semibold tracking-tight">
            <a href="/" className="hover:underline">pictolab.io</a>
            <span className="text-muted-foreground"> / wiggle</span>
          </h1>
        </div>
        <div className="flex items-center gap-2">
          {source && (
            <Button variant="outline" size="sm" onClick={handleNewImage}>
              <ImagePlus className="mr-1 h-4 w-4" />
              New image
            </Button>
          )}
        </div>
      </header>

      <div className="flex flex-1 flex-col lg:min-h-0 lg:flex-row">
        <main className="flex flex-1 flex-col bg-background lg:min-h-0">
          <div
            className="flex flex-1 items-center justify-center overflow-auto p-4 sm:p-8"
            style={{
              backgroundColor: '#1a1a1f',
              backgroundImage:
                'radial-gradient(circle, rgba(255,255,255,0.12) 1px, transparent 1px)',
              backgroundSize: '18px 18px',
            }}
          >
            {!source ? (
              <div className="w-full max-w-2xl">
                <div className="mb-4 rounded-lg bg-black/30 px-4 py-3 text-center text-sm text-neutral-300">
                  Got a three-lens wigglegram lens? Drop a photo straight off the camera —
                  all three sub-frames side by side — and get a looping 3D wiggle.
                  Everything runs in your browser; photos are never uploaded.
                </div>
                <ImageDropZone onImageSelect={handleImageSelect} showExamples={false} />
              </div>
            ) : (
              <div className="relative inline-block">
                {/* w/h auto keeps the element box at the bitmap's aspect ratio,
                    so click→pivot mapping needs no letterbox correction */}
                <canvas
                  ref={canvasRef}
                  onClick={handlePreviewClick}
                  className="block h-auto max-h-[76vh] w-auto max-w-full cursor-crosshair rounded-md"
                />
                {pivotFrac && !busy && (
                  <div
                    className="pointer-events-none absolute h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-primary shadow-[0_0_0_2px_rgba(0,0,0,0.45)]"
                    style={{ left: `${pivotFrac.fx * 100}%`, top: `${pivotFrac.fy * 100}%` }}
                  />
                )}
                {busy && (
                  <div className="absolute inset-0 flex items-center justify-center rounded-md bg-black/40 text-sm text-white">
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    {busy}
                  </div>
                )}
              </div>
            )}
          </div>
          {source && (
            <div className="border-t border-border px-4 py-2 text-center text-xs text-muted-foreground">
              Click the photo on whatever should stay still — usually a face. Everything
              else wiggles around it.
            </div>
          )}
        </main>

        <aside className="w-full shrink-0 overflow-y-auto border-t border-border bg-card p-4 lg:h-full lg:min-h-0 lg:w-80 lg:border-t-0 lg:border-l">
          <div className="space-y-6 pb-24">
            <section className="space-y-4">
              <CardTitle>Wiggle</CardTitle>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs font-normal text-muted-foreground">Speed</Label>
                  <span className="text-xs tabular-nums text-muted-foreground">{fps} fps</span>
                </div>
                <Slider
                  value={[fps]}
                  min={6}
                  max={18}
                  step={1}
                  onValueChange={([v]) => setFps(v)}
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs font-normal text-muted-foreground">Trim seams</Label>
                  <span className="text-xs tabular-nums text-muted-foreground">{inset}%</span>
                </div>
                <Slider
                  value={[inset]}
                  min={0}
                  max={8}
                  step={1}
                  onValueChange={([v]) => setInset(v)}
                  onValueCommit={([v]) => {
                    if (source) reprocess(source, pivot, v, tone, true, null);
                  }}
                />
              </div>
              <div className="flex items-center justify-between">
                <Label htmlFor="tone-match" className="text-xs font-normal text-muted-foreground">
                  Match frame brightness
                </Label>
                <Switch
                  id="tone-match"
                  checked={tone}
                  onCheckedChange={(v) => {
                    setTone(v);
                    if (source) reprocess(source, pivot, inset, v, false, shifts);
                  }}
                />
              </div>
            </section>

            <section className="space-y-4">
              <CardTitle>Export</CardTitle>
              <Button className="w-full" disabled={!frames || !!busy} onClick={handleDownloadVideo}>
                <Clapperboard className="mr-1 h-4 w-4" />
                Download video
              </Button>
              <Button
                variant="outline"
                className="w-full"
                disabled={!frames || !!busy}
                onClick={handleDownloadGif}
              >
                <Download className="mr-1 h-4 w-4" />
                Download GIF
              </Button>
              {status && <p className="text-xs text-muted-foreground">{status}</p>}
            </section>

            <section className="space-y-2 text-xs leading-relaxed text-muted-foreground">
              <p className="font-medium text-foreground">How it works</p>
              <p>
                A wigglegram lens exposes three slightly-offset views onto one photo.
                This splits them apart, matches their brightness, aligns them on the
                pivot point, and loops them 1‑2‑3‑2 — the background parallax reads
                as depth.
              </p>
            </section>
          </div>
        </aside>
      </div>
    </div>
  );
}

export default Wiggle;
