import { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import ImageDropZone from '../components/ImageDropZone';
import './Laberation.css';

function Laberation() {
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [offset, setOffset] = useState(10);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  const handleImageSelect = (imageUrl: string) => {
    setOriginalImage(imageUrl);
    processImage(imageUrl, offset);
  };

  const handleOffsetChange = (newOffset: number) => {
    setOffset(newOffset);
    if (originalImage) {
      processImage(originalImage, newOffset);
    }
  };

  const processImage = async (imageUrl: string, offsetValue: number) => {
    setIsProcessing(true);

    try {
      // Load the WASM module
      const wasm = await import('frontend-rs');

      const img = new Image();
      img.onload = () => {
        imageRef.current = img;
        const canvas = canvasRef.current;
        if (!canvas) return;

        // Set canvas dimensions to match image
        canvas.width = img.width;
        canvas.height = img.height;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Draw original image to canvas
        ctx.drawImage(img, 0, 0);

        // Get image data
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        // Convert to Uint8Array and process with WASM
        const inputArray = new Uint8Array(data);
        const outputArray = wasm.laberation(inputArray, canvas.width, canvas.height, offsetValue);

        // Update canvas with processed data
        const outputImageData = new ImageData(
          new Uint8ClampedArray(outputArray),
          canvas.width,
          canvas.height
        );
        ctx.putImageData(outputImageData, 0, 0);

        // Convert canvas to data URL
        setProcessedImage(canvas.toDataURL());
        setIsProcessing(false);
      };

      img.src = imageUrl;
    } catch (error) {
      console.error('Error processing image:', error);
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (!processedImage) return;

    const link = document.createElement('a');
    link.href = processedImage;
    link.download = 'laberation.png';
    link.click();
  };

  return (
    <div className="lab-inversion">
      <header className="demo-header">
        <Link to="/" className="back-link">← Back to Home</Link>
        <h1>Laberation</h1>
        <p className="main-description">Chromatic aberration, but in LAB color space</p>
        <p className="technical-description">Splits and offsets the L (lightness), A (green-red), and B (blue-yellow) channels to create glitchy color fringing</p>
      </header>

      <main className="demo-main">
        <div className="upload-section">
          <ImageDropZone onImageSelect={handleImageSelect} disabled={isProcessing} />
        </div>

        {originalImage && (
          <div className="controls">
            <label htmlFor="offset-slider">
              Offset: {offset}px
            </label>
            <input
              id="offset-slider"
              type="range"
              min="0"
              max="50"
              value={offset}
              onChange={(e) => handleOffsetChange(Number(e.target.value))}
              disabled={isProcessing}
            />
          </div>
        )}

        {isProcessing && (
          <div className="processing">
            Processing image...
          </div>
        )}

        {(originalImage || processedImage) && (
          <div className="images-container">
            {originalImage && (
              <div className="image-box">
                <h3>Original</h3>
                <img src={originalImage} alt="Original" />
              </div>
            )}

            {processedImage && (
              <div className="image-box">
                <h3>Laberated</h3>
                <img src={processedImage} alt="Processed" />
                <button className="download-button" onClick={handleDownload}>
                  Download
                </button>
              </div>
            )}
          </div>
        )}

        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </main>
    </div>
  );
}

export default Laberation;
