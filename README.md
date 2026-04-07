Random image experiments, mostly involving the OKLab color space, content-aware
resize via seam carving, and an HDR-aware WebGPU pipeline. Live at
[pictolab.io](https://pictolab.io).

Decodes HDR AVIF (10/12-bit BT.2020 PQ/HLG) and Ultra HDR JPEGs natively
through vendored builds of libavif and libultrahdr — wide-gamut and
above-1.0 values survive end-to-end into the editor and back out as
HDR AVIF or Ultra HDR JPEG.
