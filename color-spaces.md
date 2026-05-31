Note: these are just my working notes, not guaranteed to be accurate and partially summarized by an LLM, read at your own risk


It helps to think of two independent axes:

- **Gamut** — which primaries (which red, green, blue) you're mixing.
  sRGB, Display P3, and BT.2020 are gamuts; P3 is wider than sRGB and
  BT.2020 is wider still.
- **Transfer function** — how a stored value maps to physical light.
  Linear is mathematically clean but perceptually nonuniform. The sRGB
  curve is roughly perceptually uniform across SDR. **PQ** (SMPTE ST
  2084) is designed by Dolby to stay perceptually uniform up to 10,000
  nits. **HLG** (BT.2100) is similar but backward-compatible with SDR.

Pictures arrive with some combination of those two and have to be
normalized to a single internal format before edits can touch them.

### The internal working format: linear extended Display P3

Everything inside the pipeline is **P3 gamut + linear transfer + values
allowed above 1.0**. "Extended" is doing double duty: >1.0 for HDR
highlights, and <0.0 for the BT.2020 colors that fall outside P3 (the
decoder doesn't clamp after the BT.2020→P3 matrix, so a wide-gamut color
just becomes linear-P3 with a negative channel and survives through
`f32` on the CPU and `rgba16float` on the GPU).

Why P3 and not BT.2020 as the internal gamut? Both are defensible — the
OKLab values come out the same either way, since the LMS matrix can be
derived through XYZ for any RGB basis. The deciding factor is that the
browser canvas API gives us P3 natively (`getImageData({ colorSpace:
'display-p3' })`) but doesn't support BT.2020, so SDR loads would need
an extra matrix conversion. P3 also matches the gamut of the display
we're previewing on, so the buffer and the screen don't drift apart by
a matrix.

### On input: normalize to linear extended P3

- SDR images come through a `getImageData({ colorSpace: 'display-p3' })`
  canvas, which puts them on P3 primaries; we then invert the sRGB
  transfer curve to get linear values.
- HDR AVIF / Ultra HDR JPEG / 10-bit HEIC arrive as **BT.2020** primaries
  with a **PQ** or **HLG** transfer. The decoder inverts PQ/HLG to get
  scene-linear BT.2020, then applies a 3×3 matrix to land in linear P3.

### Inside the editor: OKLab for the actual edits

Linear P3 is the right space for *transport* (mixing, compositing,
extended-range arithmetic) but it's not perceptually uniform — equal
numerical steps don't look like equal color steps. So slider edits don't
happen in linear P3; they happen in **OKLab** (Björn Ottosson, 2020).

The conversion is linear-P3 → **LMS** (cone responses) → **cube root**
(this is what makes the space perceptually uniform — analogous in spirit
to what PQ does for a transport encoding, but applied per-cone in the
edit layer) → OKLab. Picked over CIELAB for its better hue uniformity,
especially across blues where CIELAB drifts toward purple as L changes.

OKLab values are scaled internally (L·100, a·400, b·400) so slider
numbers stay in the familiar 0..100 / ~0..160 ranges. **OKLCh** (the
polar form) drives the color picker.

### On output: re-encode for the target container

Linear extended P3 gets the inverse of whatever the destination expects:
sRGB transfer + 8-bit P3 for SDR, or a BT.2020 matrix + PQ encode for
HDR AVIF.

### Implementation note

The P3↔LMS matrices are precomputed and mirrored byte-for-byte between
`frontend-rs/src/lib.rs`, `frontend-rs/src/gpu.rs` (WGSL), and
`frontend-ts/src/lib/oklab.ts` so the JS color picker produces the same
output as the Rust/GPU pipeline.
