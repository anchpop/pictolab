// Mirrors the matrices in frontend-rs/src/lib.rs so the TS color picker
// produces identical bytes to the Rust pipeline. OKLab is in canonical
// scale (L ∈ [0,1], a/b ∈ ~[-0.4, 0.4]).

const M_LMS_TO_P3: readonly [readonly number[], readonly number[], readonly number[]] = [
  [3.1280366, -2.2571161, 0.1292834],
  [-1.0911312, 2.4133403, -0.3221775],
  [-0.0260093, -0.5080364, 1.5332833],
];

function linearToSrgbGamma(c: number): number {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

// OKLCh → linear extended Display P3 (no gamma encode, no clamp). Returned
// values are in the same color space the GPU pipeline carries internally,
// so they can be handed directly to the gpu_render bg uniforms.
export function lchToLinearP3(L: number, C: number, hDeg: number): [number, number, number] {
  const h = (hDeg * Math.PI) / 180;
  const a = C * Math.cos(h);
  const b = C * Math.sin(h);
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.2914855480 * b;
  const ll = l_ * l_ * l_;
  const mm = m_ * m_ * m_;
  const ss = s_ * s_ * s_;
  return [
    M_LMS_TO_P3[0][0] * ll + M_LMS_TO_P3[0][1] * mm + M_LMS_TO_P3[0][2] * ss,
    M_LMS_TO_P3[1][0] * ll + M_LMS_TO_P3[1][1] * mm + M_LMS_TO_P3[1][2] * ss,
    M_LMS_TO_P3[2][0] * ll + M_LMS_TO_P3[2][1] * mm + M_LMS_TO_P3[2][2] * ss,
  ];
}

export function oklabToP3Bytes(L: number, a: number, b: number): [number, number, number] {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.2914855480 * b;
  const ll = l_ * l_ * l_;
  const mm = m_ * m_ * m_;
  const ss = s_ * s_ * s_;
  const lr = M_LMS_TO_P3[0][0] * ll + M_LMS_TO_P3[0][1] * mm + M_LMS_TO_P3[0][2] * ss;
  const lg = M_LMS_TO_P3[1][0] * ll + M_LMS_TO_P3[1][1] * mm + M_LMS_TO_P3[1][2] * ss;
  const lb = M_LMS_TO_P3[2][0] * ll + M_LMS_TO_P3[2][1] * mm + M_LMS_TO_P3[2][2] * ss;
  return [
    Math.round(linearToSrgbGamma(Math.max(0, Math.min(1, lr))) * 255),
    Math.round(linearToSrgbGamma(Math.max(0, Math.min(1, lg))) * 255),
    Math.round(linearToSrgbGamma(Math.max(0, Math.min(1, lb))) * 255),
  ];
}

// Polar OKLCh → Cartesian OKLab. h is in degrees so the slider reads
// natural units; cos/sin ingest radians.
export function lchToP3Bytes(L: number, C: number, hDeg: number): [number, number, number] {
  const h = (hDeg * Math.PI) / 180;
  return oklabToP3Bytes(L, C * Math.cos(h), C * Math.sin(h));
}

// CSS `color(display-p3 ...)` so the swatch matches what the GPU pipeline
// would produce on a P3 display, instead of rounding through sRGB.
export function lchToCssColor(L: number, C: number, hDeg: number): string {
  const [r, g, b] = lchToP3Bytes(L, C, hDeg);
  return `color(display-p3 ${(r / 255).toFixed(4)} ${(g / 255).toFixed(4)} ${(b / 255).toFixed(4)})`;
}
