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

EMSCRIPTEN_BINDINGS(uhdr_module) {
  value_object<UhdrOptions>("UhdrOptions")
      .field("baseQuality", &UhdrOptions::baseQuality)
      .field("gainMapQuality", &UhdrOptions::gainMapQuality)
      .field("colorGamut", &UhdrOptions::colorGamut)
      .field("colorTransfer", &UhdrOptions::colorTransfer)
      .field("multiChannelGainmap", &UhdrOptions::multiChannelGainmap);

  function("encode", &encode);
}
