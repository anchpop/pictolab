Random image experiments, mostly involving the OKLab color space, content-aware
resize via seam carving, and an HDR-aware WebGPU pipeline. Live at
[pictolab.io](https://pictolab.io).

As far as I can tell, this is one of the only online image editors with
real HDR support. It decodes HDR AVIF (10/12-bit BT.2020 PQ/HLG), Ultra
HDR JPEGs, and iPhone gain-mapped HDR HEICs natively through vendored
builds of libavif, libultrahdr, and libheif — wide-gamut and above-1.0
values survive end-to-end into the editor and back out as HDR AVIF or
Ultra HDR JPEG.

The Apple HDR HEIC composition (gain-map application + the headroom
calculation from `MakerNotes` tags 33/48) is ported from the Python
project [`johncf/apple-hdr-heic`](https://github.com/johncf/apple-hdr-heic),
which itself follows Apple's "Applying Apple HDR effect to your photos"
developer documentation.
