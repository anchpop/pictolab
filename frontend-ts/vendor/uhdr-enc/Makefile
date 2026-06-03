# Build the libultrahdr emscripten encoder + our embind binding.
#
# Stage 1: emcmake configures the upstream libultrahdr CMake project to
#          build a static libuhdr.a, using jpegli for the underlying JPEG
#          encoder/decoder.
# Stage 2: em++ compiles uhdr_enc.cpp and links it against libuhdr.a +
#          jpegli, producing uhdr_enc.{js,wasm}.

BUILD_DIR := build
LIBUHDR_BUILD := $(BUILD_DIR)/libuhdr
LIBUHDR_OUT := $(LIBUHDR_BUILD)/libuhdr.a

JPEGLI_COMMIT := 031a0077f5799a6041004267fc12b956c1f52a20
JPEGLI_SOURCE_DIR := $(BUILD_DIR)/jpegli-src
JPEGLI_BUILD := $(BUILD_DIR)/jpegli
JPEGLI_BUILD_STAMP := $(JPEGLI_BUILD)/.built
JPEGLI_INCLUDE_DIR := $(JPEGLI_BUILD)/lib/include/jpegli
JPEGLI_LIB := $(JPEGLI_BUILD)/lib/libjpegli-static.a
JPEGLI_HWY_LIB := $(JPEGLI_BUILD)/third_party/highway/libhwy.a
JPEGLI_COMPAT_OBJ := $(JPEGLI_BUILD)/libjpeg_wrapper.o
JPEGLI_COMPAT_LIB := $(JPEGLI_BUILD)/libjpegli-compat.a

OUT_JS := uhdr_enc.js
OUT_WASM := uhdr_enc.wasm
OUT_CPP := uhdr_enc.cpp

CFLAGS ?= -Oz -flto
CXXFLAGS ?= $(CFLAGS) -std=c++17
LDFLAGS ?= $(CFLAGS)

EM_LDFLAGS := \
	--bind \
	-s ERROR_ON_UNDEFINED_SYMBOLS=1 \
	-s ENVIRONMENT=web,worker \
	-s EXPORT_ES6=1 \
	-s DYNAMIC_EXECUTION=0 \
	-s MODULARIZE=1 \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s STACK_SIZE=5242880 \
	-s INITIAL_MEMORY=16777216 \
	-s FILESYSTEM=0

.PHONY: all clean

all: $(OUT_JS)

$(JPEGLI_SOURCE_DIR)/.git:
	mkdir -p $(BUILD_DIR)
	git clone https://github.com/google/jpegli.git $(JPEGLI_SOURCE_DIR)
	git -C $(JPEGLI_SOURCE_DIR) checkout $(JPEGLI_COMMIT)
	git -C $(JPEGLI_SOURCE_DIR) submodule update --init --recursive

$(JPEGLI_BUILD_STAMP): $(JPEGLI_SOURCE_DIR)/.git
	mkdir -p $(JPEGLI_BUILD)
	emcmake cmake \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=OFF \
		-DBUILD_TESTING=OFF \
		-DJPEGLI_ENABLE_TOOLS=OFF \
		-DJPEGLI_ENABLE_DEVTOOLS=OFF \
		-DJPEGLI_ENABLE_BENCHMARK=OFF \
		-DJPEGLI_ENABLE_DOXYGEN=OFF \
		-DJPEGLI_ENABLE_MANPAGES=OFF \
		-DJPEGLI_ENABLE_JNI=OFF \
		-DJPEGLI_ENABLE_OPENEXR=OFF \
		-DJPEGLI_ENABLE_SJPEG=OFF \
		-DJPEGLI_ENABLE_WASM_THREADS=OFF \
		-DJPEGLI_ENABLE_LTO=OFF \
		-DJPEGLI_BUNDLE_LIBPNG=OFF \
		-S $(JPEGLI_SOURCE_DIR) \
		-B $(JPEGLI_BUILD)
	$(MAKE) -C $(JPEGLI_BUILD) jpegli-static
	touch $(JPEGLI_BUILD_STAMP)

$(JPEGLI_COMPAT_LIB): $(JPEGLI_BUILD_STAMP)
	em++ \
		$(CXXFLAGS) \
		-I $(JPEGLI_SOURCE_DIR) \
		-I $(JPEGLI_INCLUDE_DIR) \
		-c $(JPEGLI_SOURCE_DIR)/lib/jpegli/libjpeg_wrapper.cc \
		-o $(JPEGLI_COMPAT_OBJ)
	emar rcs $(JPEGLI_COMPAT_LIB) $(JPEGLI_COMPAT_OBJ)

$(LIBUHDR_OUT): $(JPEGLI_COMPAT_LIB)
	mkdir -p $(LIBUHDR_BUILD)
	emcmake cmake \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=0 \
		-DUHDR_BUILD_EXAMPLES=0 \
		-DUHDR_BUILD_TESTS=0 \
		-DUHDR_BUILD_BENCHMARK=0 \
		-DUHDR_BUILD_FUZZERS=0 \
		-DUHDR_BUILD_JAVA=0 \
		-DUHDR_BUILD_DEPS=0 \
		-DUHDR_JPEG_INCLUDE_DIRS="$(JPEGLI_INCLUDE_DIR)" \
		-DUHDR_JPEG_LIBRARIES="$(JPEGLI_COMPAT_LIB);$(JPEGLI_LIB);$(JPEGLI_HWY_LIB)" \
		-DCMAKE_C_FLAGS="$(CFLAGS)" \
		-DCMAKE_CXX_FLAGS="$(CXXFLAGS)" \
		-S . \
		-B $(LIBUHDR_BUILD)
	$(MAKE) -C $(LIBUHDR_BUILD) uhdr

$(OUT_JS): $(OUT_CPP) $(LIBUHDR_OUT) $(JPEGLI_COMPAT_LIB)
	em++ \
		-I . \
		$(CXXFLAGS) \
		$(LDFLAGS) \
		$(EM_LDFLAGS) \
		-o $(OUT_JS) \
		$(OUT_CPP) $(LIBUHDR_OUT) $(JPEGLI_COMPAT_LIB) $(JPEGLI_LIB) $(JPEGLI_HWY_LIB)

clean:
	rm -rf $(BUILD_DIR) $(OUT_JS) $(OUT_WASM)
