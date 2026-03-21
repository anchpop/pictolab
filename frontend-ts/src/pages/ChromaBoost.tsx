import { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import ImageDropZone from '../components/ImageDropZone';
import './Laberation.css';

function ChromaBoost() {
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [factor, setFactor] = useState(150);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  const handleImageSelect = (imageUrl: string) => {
    setOriginalImage(imageUrl);
    processImage(imageUrl, factor);
  };

  const handleFactorChange = (newFactor: number) => {
    setFactor(newFactor);
    if (originalImage) {
      processImage(originalImage, newFactor);
    }
  };

  const processImage = async (imageUrl: string, factorValue: number) => {
    setIsProcessing(true);

    try {
      const wasm = await import('frontend-rs');

      const img = new Image();
      img.onload = () => {
        imageRef.current = img;
        const canvas = canvasRef.current;
        if (!canvas) return;

        canvas.width = img.width;
        canvas.height = img.height;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.drawImage(img, 0, 0);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        const inputArray = new Uint8Array(data);
        const outputArray = wasm.boost_chroma_lab(inputArray, canvas.width, canvas.height, factorValue / 100);

        const outputImageData = new ImageData(
          new Uint8ClampedArray(outputArray),
          canvas.width,
          canvas.height
        );
        ctx.putImageData(outputImageData, 0, 0);

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
    link.download = 'chroma-boost.png';
    link.click();
  };

  return (
    <div className="lab-inversion">
      <header className="demo-header">
        <Link to="/" className="back-link">← Back to Home</Link>
        <h1>Chroma Boost</h1>
        <p className="main-description">Increase or decrease color intensity in LAB space</p>
        <p className="technical-description">Scales the A (green-red) and B (blue-yellow) channels to amplify or reduce chroma while preserving lightness</p>
      </header>

      <main className="demo-main">
        <div className="upload-section">
          <ImageDropZone onImageSelect={handleImageSelect} disabled={isProcessing} />
        </div>

        {originalImage && (
          <div className="controls">
            <label htmlFor="factor-slider">
              Chroma: {factor}%
            </label>
            <input
              id="factor-slider"
              type="range"
              min="0"
              max="300"
              value={factor}
              onChange={(e) => handleFactorChange(Number(e.target.value))}
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
                <h3>Boosted</h3>
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

export default ChromaBoost;
