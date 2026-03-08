import { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import ImageDropZone from '../components/ImageDropZone';
import './SeamCarving.css';

function SeamCarving() {
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [seamsPercent, setSeamsPercent] = useState(20);
  const [imageWidth, setImageWidth] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleImageSelect = (imageUrl: string) => {
    setOriginalImage(imageUrl);
    // Need to load image to get width before processing
    const img = new Image();
    img.onload = () => {
      setImageWidth(img.width);
      processImage(imageUrl, seamsPercent, img.width);
    };
    img.src = imageUrl;
  };

  const handleSeamsChange = (newPercent: number) => {
    setSeamsPercent(newPercent);
    if (originalImage && imageWidth > 0) {
      processImage(originalImage, newPercent, imageWidth);
    }
  };

  const processImage = async (imageUrl: string, percent: number, imgWidth: number) => {
    setIsProcessing(true);

    try {
      const wasm = await import('frontend-rs');

      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        canvas.width = img.width;
        canvas.height = img.height;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.drawImage(img, 0, 0);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const inputArray = new Uint8Array(imageData.data);
        const seamsToRemove = Math.floor(imgWidth * (percent / 100));
        const outputArray = wasm.seam_carve_lab(inputArray, canvas.width, canvas.height, seamsToRemove);

        const newWidth = canvas.width - seamsToRemove;
        canvas.width = newWidth;

        const outputImageData = new ImageData(
          new Uint8ClampedArray(outputArray),
          newWidth,
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
    link.download = 'seam-carved.png';
    link.click();
  };

  const seamsToRemove = Math.floor(imageWidth * (seamsPercent / 100));

  return (
    <div className="lab-inversion">
      <header className="demo-header">
        <Link to="/" className="back-link">← Back to Home</Link>
        <h1>Seam Carving</h1>
        <p className="main-description">Content-aware image resizing with forward energy</p>
        <p className="technical-description">Computes energy in LAB color space for perceptually accurate seam detection, using forward energy to minimize artifacts</p>
      </header>

      <main className="demo-main">
        <div className="upload-section">
          <ImageDropZone onImageSelect={handleImageSelect} disabled={isProcessing} />
        </div>

        {originalImage && (
          <div className="controls">
            <label htmlFor="seams-slider">
              Remove: {seamsPercent}% width ({seamsToRemove}px)
            </label>
            <input
              id="seams-slider"
              type="range"
              min="1"
              max="50"
              value={seamsPercent}
              onChange={(e) => handleSeamsChange(Number(e.target.value))}
              disabled={isProcessing}
            />
          </div>
        )}

        {isProcessing && (
          <div className="processing">
            Carving seams...
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
                <h3>Carved</h3>
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

export default SeamCarving;
