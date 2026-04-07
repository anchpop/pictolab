// Minimal Embind binding around libultrahdr's encoder. Takes a single HDR
// raw image (rgba16float, linear, Display P3 by default) and emits an
// Ultra HDR JPEG (a regular SDR JPEG with a gain-map JPEG embedded via
// MPF + the Adobe gain-map XMP). libultrahdr derives the SDR base image,
// the gain map, and the metadata internally.

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "ultrahdr_api.h"

#include <memory>
#include <string>
#include <cstring>

using namespace emscripten;

struct UhdrOptions {
  // [0 - 100] base SDR JPEG quality (libultrahdr default 95)
  int baseQuality;
  // [0 - 100] gain map JPEG quality (libultrahdr default 95)
  int gainMapQuality;
  // UHDR_CG_DISPLAY_P3 (1) | UHDR_CG_BT_709 (0) | UHDR_CG_BT_2100 (2)
  int colorGamut;
  // UHDR_CT_LINEAR (0) | UHDR_CT_HLG (1) | UHDR_CT_PQ (2) | UHDR_CT_SRGB (3)
  int colorTransfer;
  // 1 = use a 3-channel gain map (better fidelity), 0 = monochrome
  int multiChannelGainmap;
  // Target display peak brightness in nits, written into the gain map
  // metadata (hdr_capacity_max). Determines the weight by which decoders
  // scale the gain map. Pass 0 to leave the libultrahdr default
  // (10000 nits for CT_LINEAR/CT_PQ, 1000 nits for CT_HLG). Should be
  // set to the actual content peak for this image — anything else makes
  // decoders under- or over-apply the gain map.
  float targetDisplayPeakNits;
};

thread_local const val Uint8Array = val::global("Uint8Array");

// Encode a half-float RGBA image (linear by default) as Ultra HDR JPEG.
// `buffer` is interpreted as raw bytes of `width * height * 8` (4 channels
// of f16 each). Returns a Uint8Array view of the encoded JPEG, or null on
// failure.
val encode(std::string buffer, int width, int height, UhdrOptions options) {
  if ((size_t)buffer.size() < (size_t)width * (size_t)height * 8) {
    return val::null();
  }

  uhdr_raw_image_t img{};
  img.fmt = UHDR_IMG_FMT_64bppRGBAHalfFloat;
  img.cg = (uhdr_color_gamut_t)options.colorGamut;
  img.ct = (uhdr_color_transfer_t)options.colorTransfer;
  img.range = UHDR_CR_FULL_RANGE;
  img.w = (unsigned int)width;
  img.h = (unsigned int)height;
  img.planes[UHDR_PLANE_PACKED] =
      reinterpret_cast<void*>(const_cast<char*>(buffer.data()));
  img.stride[UHDR_PLANE_PACKED] = (unsigned int)width; // stride is in *pixels*
  img.planes[1] = nullptr;
  img.planes[2] = nullptr;
  img.stride[1] = 0;
  img.stride[2] = 0;

  uhdr_codec_private_t* enc = uhdr_create_encoder();
  if (!enc) return val::null();

  uhdr_error_info_t err;

  err = uhdr_enc_set_raw_image(enc, &img, UHDR_HDR_IMG);
  if (err.error_code != UHDR_CODEC_OK) {
    uhdr_release_encoder(enc);
    return val::null();
  }

  err = uhdr_enc_set_quality(enc, options.baseQuality, UHDR_BASE_IMG);
  if (err.error_code != UHDR_CODEC_OK) {
    uhdr_release_encoder(enc);
    return val::null();
  }
  err = uhdr_enc_set_quality(enc, options.gainMapQuality, UHDR_GAIN_MAP_IMG);
  if (err.error_code != UHDR_CODEC_OK) {
    uhdr_release_encoder(enc);
    return val::null();
  }

  err = uhdr_enc_set_using_multi_channel_gainmap(enc, options.multiChannelGainmap);
  if (err.error_code != UHDR_CODEC_OK) {
    uhdr_release_encoder(enc);
    return val::null();
  }

  if (options.targetDisplayPeakNits > 0.0f) {
    err = uhdr_enc_set_target_display_peak_brightness(enc, options.targetDisplayPeakNits);
    if (err.error_code != UHDR_CODEC_OK) {
      uhdr_release_encoder(enc);
      return val::null();
    }
  }

  err = uhdr_encode(enc);
  if (err.error_code != UHDR_CODEC_OK) {
    uhdr_release_encoder(enc);
    return val::null();
  }

  uhdr_compressed_image_t* out = uhdr_get_encoded_stream(enc);
  if (!out || !out->data || out->data_sz == 0) {
    uhdr_release_encoder(enc);
    return val::null();
  }

  // Copy out before releasing the encoder.
  val js_result = Uint8Array.new_(typed_memory_view(out->data_sz, (uint8_t*)out->data));
  // typed_memory_view aliases the encoder's heap memory, so make a real
  // owned copy on the JS side before we release.
  val owned = Uint8Array.new_(js_result);

  uhdr_release_encoder(enc);
  return owned;
}

// Quick magic-byte check: "is this an Ultra HDR JPEG?". Mirrors libuhdr's
// own probe so JS callers can sniff a buffer before paying for a decode.
bool is_ultra_hdr(std::string buffer) {
  return is_uhdr_image(const_cast<char*>(buffer.data()), (int)buffer.size()) != 0;
}

// Decode an Ultra HDR JPEG into linear half-float RGBA pixels (8 bytes per
// pixel). Color is BT.2100 / Display P3 primaries — the gamut field is
// returned alongside so the caller can convert if needed. Returns null on
// failure.
val decode(std::string buffer) {
  uhdr_codec_private_t* dec = uhdr_create_decoder();
  if (!dec) return val::null();

  uhdr_compressed_image_t in{};
  in.data = const_cast<char*>(buffer.data());
  in.data_sz = buffer.size();
  in.capacity = buffer.size();
  in.cg = UHDR_CG_UNSPECIFIED;
  in.ct = UHDR_CT_UNSPECIFIED;
  in.range = UHDR_CR_UNSPECIFIED;

  uhdr_error_info_t err;
  err = uhdr_dec_set_image(dec, &in);
  if (err.error_code != UHDR_CODEC_OK) {
    uhdr_release_decoder(dec);
    return val::null();
  }

  // Ask libultrahdr for linear half-float output. Per the API docs,
  // UHDR_CT_LINEAR must pair with UHDR_IMG_FMT_64bppRGBAHalfFloat.
  err = uhdr_dec_set_out_img_format(dec, UHDR_IMG_FMT_64bppRGBAHalfFloat);
  if (err.error_code != UHDR_CODEC_OK) {
    uhdr_release_decoder(dec);
    return val::null();
  }
  err = uhdr_dec_set_out_color_transfer(dec, UHDR_CT_LINEAR);
  if (err.error_code != UHDR_CODEC_OK) {
    uhdr_release_decoder(dec);
    return val::null();
  }

  err = uhdr_decode(dec);
  if (err.error_code != UHDR_CODEC_OK) {
    uhdr_release_decoder(dec);
    return val::null();
  }

  uhdr_raw_image_t* out = uhdr_get_decoded_image(dec);
  if (!out || !out->planes[UHDR_PLANE_PACKED]) {
    uhdr_release_decoder(dec);
    return val::null();
  }

  const size_t pixel_count = (size_t)out->w * (size_t)out->h;
  const size_t byte_count = pixel_count * 8; // f16 RGBA
  uint8_t* src = reinterpret_cast<uint8_t*>(out->planes[UHDR_PLANE_PACKED]);
  // stride is in *pixels*; row size in bytes is stride * 8.
  const size_t row_bytes_dst = (size_t)out->w * 8;
  const size_t row_bytes_src = (size_t)out->stride[UHDR_PLANE_PACKED] * 8;

  // Pack tightly for JS. Allocate via Uint8Array on the JS side.
  val pixels = Uint8Array.new_(byte_count);
  if (row_bytes_dst == row_bytes_src) {
    pixels.call<void>("set", val(typed_memory_view(byte_count, src)));
  } else {
    for (size_t y = 0; y < (size_t)out->h; ++y) {
      val row = val(typed_memory_view(row_bytes_dst, src + y * row_bytes_src));
      pixels.call<void>("set", row, (unsigned)(y * row_bytes_dst));
    }
  }

  val result = val::object();
  result.set("data", pixels);
  result.set("width", (int)out->w);
  result.set("height", (int)out->h);
  result.set("colorGamut", (int)out->cg);
  result.set("colorTransfer", (int)out->ct);

  uhdr_release_decoder(dec);
  return result;
}

EMSCRIPTEN_BINDINGS(uhdr_module) {
  value_object<UhdrOptions>("UhdrOptions")
      .field("baseQuality", &UhdrOptions::baseQuality)
      .field("gainMapQuality", &UhdrOptions::gainMapQuality)
      .field("colorGamut", &UhdrOptions::colorGamut)
      .field("colorTransfer", &UhdrOptions::colorTransfer)
      .field("multiChannelGainmap", &UhdrOptions::multiChannelGainmap)
      .field("targetDisplayPeakNits", &UhdrOptions::targetDisplayPeakNits);

  function("encode", &encode);
  function("decode", &decode);
  function("isUltraHdr", &is_ultra_hdr);
}
