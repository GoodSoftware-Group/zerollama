//go:build darwin && arm64

package metal

// Embedded Metal libraries for the ggml Metal backend (GGML_METAL_EMBED_LIBRARY).
//
// b10615 split shaders into kernels/<kind>.metal. ggml-metal-device.m JIT-compiles
// each kind from UTF-8 bytes between _ggml_metallib_<kind>_{start,end}.
// scripts/build/gen_ggml_metal_embed.sh inlines headers (CMake-equivalent) for
// kernels/*.metal and copies eliza-shipped/*.metal as their own embed kinds
// (do not concat onto misc — those files redefine QJL/Polar types). Do not
// embed compiled MTLB here.
//
//	GOFLAGS=-mod=readonly go generate ./ml/backend/ggml/ggml/src/ggml-metal/
//
//go:generate ../../../../../../scripts/build/gen_ggml_metal_embed.sh

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -DGGML_METAL_NDEBUG -DGGML_METAL_EMBED_LIBRARY -DGGML_METAL_HAS_BF16 -I.. -I../../include
// #cgo LDFLAGS: -framework Metal -framework MetalKit -framework IOSurface
import "C"
