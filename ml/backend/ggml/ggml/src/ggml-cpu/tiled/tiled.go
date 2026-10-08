package tiled

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR} -I${SRCDIR}/.. -I${SRCDIR}/../.. -I${SRCDIR}/../../../include
// #cgo linux CPPFLAGS: -D_GNU_SOURCE
import "C"

// WHY: tip ggml-cpu calls tiled matmul; CGO only compiles .c/.cpp in this package dir.
