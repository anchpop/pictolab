const ctx = self as unknown as Worker;

let wasm: any = null;

ctx.onmessage = async (e: MessageEvent) => {
  const { imageData, width, height, direction, requestId, useGPU } = e.data;

  if (!wasm) {
    wasm = await import('frontend-rs');
  }

  const input = new Uint8Array(imageData);
  let order: Uint32Array;
  let usedGPU = false;

  if (useGPU) {
    try {
      order = await wasm.precompute_seam_order_gpu(input, width, height, direction);
      usedGPU = true;
    } catch (err) {
      console.warn('GPU seam carving failed, falling back to WASM:', err);
      order = wasm.precompute_seam_order(input, width, height, direction);
    }
  } else {
    order = wasm.precompute_seam_order(input, width, height, direction);
  }

  ctx.postMessage({ order, requestId, usedGPU }, [order.buffer]);
};

export {};
