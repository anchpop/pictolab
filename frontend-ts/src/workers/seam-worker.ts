const ctx = self as unknown as Worker;

let wasm: any = null;

ctx.onmessage = async (e: MessageEvent) => {
  const { imageData, width, height, direction, requestId } = e.data;

  if (!wasm) {
    wasm = await import('frontend-rs');
  }

  const order = wasm.precompute_seam_order(
    new Uint8Array(imageData),
    width,
    height,
    direction
  );

  // Transfer the order buffer back to avoid copying
  ctx.postMessage({ order, requestId }, [order.buffer]);
};

export {};
