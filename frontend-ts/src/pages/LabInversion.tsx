import { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import ImageDropZone from '../components/ImageDropZone';
import './LabInversion.css';

function LabInversion() {
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleImageSelect = (imageUrl: string) => {
    setOriginalImage(imageUrl);
    processImage(imageUrl);
  };

  const processImage = async (imageUrl: string) => {
    setIsProcessing(true);

    try {
      // Load the WASM module
      const wasm = await import('frontend-rs');

      const img = new Image();
      img.onload = () => {
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
        const outputArray = wasm.invert_lightness_lab(inputArray, canvas.width, canvas.height);

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
    link.download = 'lab-inverted.png';
    link.click();
  };

  return (
    <div className="lab-inversion">
      <header className="demo-header">
        <Link to="/" className="back-link">← Back to Home</Link>
        <h1>Smart Invert</h1>
        <p className="main-description">Invert the brightness without messing up the colors</p>
        <p className="technical-description">Uses LAB color space to invert only the lightness channel, preserving hue and saturation for perceptually uniform results</p>
      </header>

      <main className="demo-main">
        <div className="upload-section">
          <ImageDropZone onImageSelect={handleImageSelect} disabled={isProcessing} />
        </div>

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
                <h3>LAB Inverted</h3>
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

export default LabInversion;
