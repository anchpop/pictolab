import { useRef, useState, useEffect } from 'react';
import './ImageDropZone.css';

const EXAMPLE_IMAGES = [
  { src: '/examples/artwork_01.webp', label: 'Artwork 1' },
  { src: '/examples/artwork_02.webp', label: 'Artwork 2' },
  { src: '/examples/artwork_03.webp', label: 'Artwork 3' },
  { src: '/examples/artwork_04.webp', label: 'Artwork 4' },
  { src: '/examples/artwork_05.webp', label: 'Artwork 5' },
];

const IS_IOS =
  typeof navigator !== 'undefined' &&
  (/iPad|iPhone|iPod/.test(navigator.userAgent) ||
    (navigator.platform === 'MacIntel' && (navigator as any).maxTouchPoints > 1));

const DB_NAME = 'pictolab-images';
const STORE_NAME = 'uploads';
const MAX_SAVED_IMAGES = 10;

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function getSavedImages(): Promise<{ id: number; dataUrl: string; thumbnail: string }[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function saveImage(dataUrl: string, thumbnail: string) {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, 'readwrite');
  const store = tx.objectStore(STORE_NAME);

  // Check count and evict oldest if needed
  const countReq = store.count();
  countReq.onsuccess = () => {
    if (countReq.result >= MAX_SAVED_IMAGES) {
      const cursorReq = store.openCursor();
      cursorReq.onsuccess = () => {
        const cursor = cursorReq.result;
        if (cursor) cursor.delete();
      };
    }
  };

  store.add({ dataUrl, thumbnail });
}

async function removeSavedImage(id: number) {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, 'readwrite');
  tx.objectStore(STORE_NAME).delete(id);
}

// Decode-via-canvas thumbnail. Works for any format the browser can
// natively decode (PNG/JPEG/WebP/AVIF). Rejects on error so the caller
// can fall through to the wasm-decode path for HEIC etc.
function makeThumbnailViaImage(dataUrl: string, maxSize: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const scale = Math.min(maxSize / img.width, maxSize / img.height, 1);
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL('image/jpeg', 0.7));
    };
    img.onerror = () => reject(new Error('image decode failed'));
    img.src = dataUrl;
  });
}

// Fallback that decodes through our HDR dispatcher (handles HEIC and
// any other format the browser can't natively render). Operates on the
// 8-bit Display P3 sdrPixels the dispatcher already produces.
async function makeThumbnailViaWasm(dataUrl: string, maxSize: number): Promise<string> {
  const buf = new Uint8Array(await (await fetch(dataUrl)).arrayBuffer());
  const { decodeHdr } = await import('@/lib/hdr-decode');
  const decoded = await decodeHdr(buf);
  if (!decoded) throw new Error('wasm decode failed');
  const scale = Math.min(maxSize / decoded.width, maxSize / decoded.height, 1);
  const w = Math.max(1, Math.round(decoded.width * scale));
  const h = Math.max(1, Math.round(decoded.height * scale));
  // Drop the source pixels into a full-size canvas first, then scale.
  const src = document.createElement('canvas');
  src.width = decoded.width;
  src.height = decoded.height;
  const sctx = src.getContext('2d')!;
  const id = sctx.createImageData(decoded.width, decoded.height);
  id.data.set(decoded.sdrPixels);
  sctx.putImageData(id, 0, 0);
  const dst = document.createElement('canvas');
  dst.width = w;
  dst.height = h;
  const dctx = dst.getContext('2d')!;
  dctx.drawImage(src, 0, 0, w, h);
  return dst.toDataURL('image/jpeg', 0.7);
}

async function makeThumbnail(dataUrl: string, maxSize = 150): Promise<string> {
  try {
    return await makeThumbnailViaImage(dataUrl, maxSize);
  } catch {
    return await makeThumbnailViaWasm(dataUrl, maxSize);
  }
}

interface ImageDropZoneProps {
  onImageSelect: (imageUrl: string) => void;
  disabled?: boolean;
}

function ImageDropZone({ onImageSelect, disabled = false }: ImageDropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [savedImages, setSavedImages] = useState<{ id: number; dataUrl: string; thumbnail: string }[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getSavedImages().then(setSavedImages).catch(() => {});
  }, []);

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file');
      return;
    }

    const reader = new FileReader();
    reader.onload = async (e) => {
      const imageUrl = e.target?.result as string;
      onImageSelect(imageUrl);

      // Save to IndexedDB for future use
      try {
        const thumbnail = await makeThumbnail(imageUrl);
        await saveImage(imageUrl, thumbnail);
        const updated = await getSavedImages();
        setSavedImages(updated);
      } catch {
        // Storage failure is non-critical
      }
    };
    reader.readAsDataURL(file);
  };

  const handlePaste = (e: ClipboardEvent) => {
    if (disabled) return;

    const items = e.clipboardData?.items;
    if (!items) return;

    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) {
          handleFile(file);
          break;
        }
      }
    }
  };

  useEffect(() => {
    window.addEventListener('paste', handlePaste);
    return () => {
      window.removeEventListener('paste', handlePaste);
    };
  }, [disabled, onImageSelect]);

  const handleFileInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (disabled) return;

    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const handleClick = () => {
    if (!disabled) {
      fileInputRef.current?.click();
    }
  };

  const handleExampleClick = (src: string) => {
    if (disabled) return;
    // Fetch the example image and convert to data URL so it works the same as uploads
    fetch(src)
      .then((res) => res.blob())
      .then((blob) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          const dataUrl = e.target?.result as string;
          onImageSelect(dataUrl);
        };
        reader.readAsDataURL(blob);
      });
  };

  const handleSavedClick = (img: { id: number; dataUrl: string }) => {
    if (disabled) return;
    onImageSelect(img.dataUrl);
  };

  const handleRemoveSaved = async (e: React.MouseEvent, id: number) => {
    e.stopPropagation();
    await removeSavedImage(id);
    setSavedImages((prev) => prev.filter((img) => img.id !== id));
  };

  return (
    <div className="image-picker">
      {IS_IOS && (
        <div className="ios-hdr-banner">
          Heads up: iOS Safari converts photos to standard JPEG when uploading,
          so HDR images can't be edited at full fidelity here on iPhone or iPad.
        </div>
      )}
      <div
        className={`image-drop-zone ${isDragging ? 'dragging' : ''} ${disabled ? 'disabled' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileInput}
          style={{ display: 'none' }}
        />
        <div className="drop-zone-content">
          <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <p className="drop-zone-title">Drop an image here</p>
          <p className="drop-zone-subtitle">or click to browse, or paste from clipboard</p>
        </div>
      </div>

      <div className="image-gallery">
        <div className="gallery-section">
          <p className="gallery-label">
            Examples — art by{' '}
            <a href="https://x.com/aletiune" target="_blank" rel="noopener noreferrer" className="gallery-attribution">
              aletiune
            </a>
            , used with permission
          </p>
          <div className="gallery-thumbs">
            {EXAMPLE_IMAGES.map((img) => (
              <button
                key={img.src}
                className="gallery-thumb"
                onClick={() => handleExampleClick(img.src)}
                disabled={disabled}
                title={img.label}
              >
                <img src={img.src} alt={img.label} loading="lazy" />
              </button>
            ))}
          </div>
        </div>

        {savedImages.length > 0 && (
          <div className="gallery-section">
            <p className="gallery-label">Your uploads</p>
            <div className="gallery-thumbs">
              {savedImages.map((img) => (
                <button
                  key={img.id}
                  className="gallery-thumb saved-thumb"
                  onClick={() => handleSavedClick(img)}
                  disabled={disabled}
                >
                  <img src={img.thumbnail} alt="Saved upload" />
                  <span
                    className="remove-thumb"
                    onClick={(e) => handleRemoveSaved(e, img.id)}
                    title="Remove"
                  >
                    &times;
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ImageDropZone;
