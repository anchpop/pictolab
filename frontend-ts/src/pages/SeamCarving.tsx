import { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';
import ImageDropZone from '../components/ImageDropZone';
import './SeamCarving.css';

function SeamCarving() {
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [isPrecomputing, setIsPrecomputing] = useState(false);
  const [ready, setReady] = useState(false);
  const [direction, setDirection] = useState<'width' | 'height'>('width');
  const [targetSize, setTargetSize] = useState(0);
  const [origDims, setOrigDims] = useState({ w: 0, h: 0 });
  const [aspectText, setAspectText] = useState('');

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const requestIdRef = useRef(0);
  const stateRef = useRef({
    imageData: null as Uint8Array | null,
    orderMap: null as Uint32Array | null,
    wasm: null as any,
    origW: 0,
    origH: 0,
    direction: 'width' as 'width' | 'height',
  });

  // Clean up worker on unmount
  useEffect(() => {
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

  const renderAtSize = (size: number) => {
    const { wasm, imageData, orderMap, origW, origH, direction: dir } = stateRef.current;
    if (!wasm || !imageData || !orderMap) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const dirNum = dir === 'width' ? 0 : 1;
    const outputArray = wasm.render_seam_carved(imageData, orderMap, origW, origH, size, dirNum);

    const outW = dir === 'width' ? size : origW;
    const outH = dir === 'height' ? size : origH;
    canvas.width = outW;
    canvas.height = outH;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.putImageData(new ImageData(new Uint8ClampedArray(outputArray), outW, outH), 0, 0);
  };

  const runPrecompute = async (w: number, h: number, dir: 'width' | 'height') => {
    const imageData = stateRef.current.imageData;
    if (!imageData) return;

    setIsPrecomputing(true);
    setReady(false);

    // Load WASM on main thread for render_seam_carved
    const wasm = await import('frontend-rs');
    stateRef.current.wasm = wasm;
    stateRef.current.direction = dir;

    const currentId = ++requestIdRef.current;
    const dirNum = dir === 'width' ? 0 : 1;

    const worker = getWorker();
    worker.onmessage = (e: MessageEvent) => {
      const { order, requestId } = e.data;

      // Ignore stale results from previous precomputations
      if (requestId !== requestIdRef.current) return;

      stateRef.current.orderMap = order;

      const max = dir === 'width' ? w : h;
      const defaultTarget = Math.round(max * 0.8);

      setTargetSize(defaultTarget);
      setIsPrecomputing(false);
      setReady(true);
      updateAspectText(dir, defaultTarget, w, h);

      // Defer render to next frame so React has mounted the canvas
      requestAnimationFrame(() => renderAtSize(defaultTarget));
    };

    // Send image data to worker (structured clone copies it)
    worker.postMessage({
      imageData: imageData,
      width: w,
      height: h,
      direction: dirNum,
      requestId: currentId,
    });
  };

  const handleImageSelect = (imageUrl: string) => {
    setOriginalImage(imageUrl);

    const img = new Image();
    img.onload = () => {
      setOrigDims({ w: img.width, h: img.height });
      stateRef.current.origW = img.width;
      stateRef.current.origH = img.height;

      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = img.width;
      tempCanvas.height = img.height;
      const ctx = tempCanvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, img.width, img.height);
      stateRef.current.imageData = new Uint8Array(data.data);

      runPrecompute(img.width, img.height, direction);
    };
    img.src = imageUrl;
  };

  const handleDirectionChange = (dir: 'width' | 'height') => {
    setDirection(dir);
    if (stateRef.current.imageData) {
      runPrecompute(stateRef.current.origW, stateRef.current.origH, dir);
    }
  };

  const handleTargetChange = (size: number) => {
    const dir = stateRef.current.direction;
    const max = dir === 'width' ? stateRef.current.origW : stateRef.current.origH;
    const clamped = Math.max(1, Math.min(max, Math.round(size)));
    setTargetSize(clamped);
    updateAspectText(dir, clamped, stateRef.current.origW, stateRef.current.origH);
    renderAtSize(clamped);
  };

  const updateAspectText = (dir: string, target: number, w: number, h: number) => {
    const outW = dir === 'width' ? target : w;
    const outH = dir === 'height' ? target : h;
    setAspectText(outH > 0 ? (outW / outH).toFixed(2) : '');
  };

  const handleAspectCommit = () => {
    const ratio = parseFloat(aspectText);
    if (!ratio || ratio <= 0 || !isFinite(ratio)) return;
    const dir = stateRef.current.direction;
    if (dir === 'width') {
      handleTargetChange(ratio * stateRef.current.origH);
    } else {
      handleTargetChange(stateRef.current.origW / ratio);
    }
  };

  const handleDownload = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.href = canvas.toDataURL();
    link.download = 'seam-carved.png';
    link.click();
  };

  const dir = stateRef.current.direction;
  const outputWidth = dir === 'width' ? targetSize : origDims.w;
  const outputHeight = dir === 'height' ? targetSize : origDims.h;
  const maxSize = dir === 'width' ? origDims.w : origDims.h;

  return (
    <div className="lab-inversion">
      <header className="demo-header">
        <Link to="/" className="back-link">← Back to Home</Link>
        <h1>Seam Carving</h1>
        <p className="main-description">Content-aware image resizing with forward energy</p>
        <p className="technical-description">Uses <a href="https://avikdas.com/2019/07/29/improved-seam-carving-with-forward-energy.html" target="_blank" rel="noopener noreferrer">seam carving with forward energy</a> in LAB color space. All seams are precomputed so any target size renders in real-time.</p>
      </header>

      <main className="demo-main">
        <div className="upload-section">
          <ImageDropZone onImageSelect={handleImageSelect} disabled={isPrecomputing} />
        </div>

        {originalImage && (
          <div className="controls">
            <div className="direction-toggle">
              <button
                className={direction === 'width' ? 'active' : ''}
                onClick={() => handleDirectionChange('width')}
                disabled={isPrecomputing}
              >
                Reduce Width
              </button>
              <button
                className={direction === 'height' ? 'active' : ''}
                onClick={() => handleDirectionChange('height')}
                disabled={isPrecomputing}
              >
                Reduce Height
              </button>
            </div>

            {ready && (
              <>
                <label htmlFor="size-slider">
                  {direction === 'width' ? 'Width' : 'Height'}: {targetSize}px
                </label>
                <input
                  id="size-slider"
                  type="range"
                  min="1"
                  max={maxSize}
                  value={targetSize}
                  onChange={(e) => handleTargetChange(Number(e.target.value))}
                />

                <div className="size-inputs">
                  <label>
                    W:
                    <input
                      type="number"
                      value={outputWidth}
                      min={1}
                      max={origDims.w}
                      onChange={(e) => {
                        if (direction === 'width') handleTargetChange(Number(e.target.value));
                      }}
                      disabled={direction !== 'width'}
                    />
                  </label>
                  <span className="times">&times;</span>
                  <label>
                    H:
                    <input
                      type="number"
                      value={outputHeight}
                      min={1}
                      max={origDims.h}
                      onChange={(e) => {
                        if (direction === 'height') handleTargetChange(Number(e.target.value));
                      }}
                      disabled={direction !== 'height'}
                    />
                  </label>
                </div>

                <div className="aspect-input">
                  <label>
                    Aspect Ratio:
                    <input
                      type="text"
                      value={aspectText}
                      onChange={(e) => setAspectText(e.target.value)}
                      onBlur={handleAspectCommit}
                      onKeyDown={(e) => { if (e.key === 'Enter') handleAspectCommit(); }}
                    />
                  </label>
                </div>
              </>
            )}
          </div>
        )}

        {isPrecomputing && (
          <div className="processing">
            Precomputing seam order...
          </div>
        )}

        {(originalImage || ready) && (
          <div className="images-container">
            {originalImage && (
              <div className="image-box">
                <h3>Original ({origDims.w} &times; {origDims.h})</h3>
                <img src={originalImage} alt="Original" />
              </div>
            )}

            {ready && (
              <div className="image-box">
                <h3>Carved ({outputWidth} &times; {outputHeight})</h3>
                <div
                  className="canvas-container"
                  style={{
                    aspectRatio: `${origDims.w} / ${origDims.h}`,
                  }}
                >
                  <canvas
                    ref={canvasRef}
                    className="result-canvas"
                    style={{
                      width: direction === 'width' ? `${(targetSize / origDims.w) * 100}%` : '100%',
                      height: direction === 'height' ? `${(targetSize / origDims.h) * 100}%` : '100%',
                    }}
                  />
                </div>
                <button className="download-button" onClick={handleDownload}>
                  Download
                </button>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default SeamCarving;
