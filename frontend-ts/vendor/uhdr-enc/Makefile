# Build the libultrahdr emscripten encoder + our embind binding.
#
# Stage 1: emcmake configures the upstream libultrahdr CMake project to
#          build a static libuhdr.a, using emscripten's libjpeg port for
#          the underlying JPEG encoder.
# Stage 2: em++ compiles uhdr_enc.cpp and links it against libuhdr.a +
#          --use-port=libjpeg, producing uhdr_enc.{js,wasm}.

BUILD_DIR := build
LIBUHDR_BUILD := $(BUILD_DIR)/libuhdr
LIBUHDR_OUT := $(LIBUHDR_BUILD)/libuhdr.a

OUT_JS := uhdr_enc.js
OUT_WASM := uhdr_enc.wasm
OUT_CPP := uhdr_enc.cpp

CFLAGS ?= -Oz -flto
CXXFLAGS ?= $(CFLAGS) -std=c++17
LDFLAGS ?= $(CFLAGS)

EM_LDFLAGS := \
	--use-port=libjpeg \
	--bind \
	-s ERROR_ON_UNDEFINED_SYMBOLS=0 \
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

$(LIBUHDR_OUT):
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
		-DCMAKE_C_FLAGS="$(CFLAGS) --use-port=libjpeg" \
		-DCMAKE_CXX_FLAGS="$(CXXFLAGS) --use-port=libjpeg" \
		-S . \
		-B $(LIBUHDR_BUILD)
	$(MAKE) -C $(LIBUHDR_BUILD) uhdr

$(OUT_JS): $(OUT_CPP) $(LIBUHDR_OUT)
	em++ \
		-I . \
		$(CXXFLAGS) \
		$(LDFLAGS) \
		$(EM_LDFLAGS) \
		-o $(OUT_JS) \
		$(OUT_CPP) $(LIBUHDR_OUT)

clean:
	rm -rf $(BUILD_DIR) $(OUT_JS) $(OUT_WASM)
