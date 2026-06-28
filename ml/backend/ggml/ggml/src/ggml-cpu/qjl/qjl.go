// Package qjl links elizaOS QJL CPU kernels into the ggml-cpu CGO build.
//
// Why: vendor ggml references qjl1_256 in type traits; CMake lists these TUs under
// ggml-cpu/qjl/ but CGO only compiles C sources beside a .go file in the package.
package qjl

// #cgo CFLAGS: -O3
// #cgo CPPFLAGS: -I${SRCDIR}/.. -I${SRCDIR}/../.. -I${SRCDIR}/../../../include
// #cgo CPPFLAGS: -I${SRCDIR}/include -I${SRCDIR}
import "C"
