package parsers

// #cgo CXXFLAGS: -std=c++17 -Wno-deprecated-declarations
// #cgo CPPFLAGS: -I${SRCDIR} -I${SRCDIR}/.. -I${SRCDIR}/../../include -I${SRCDIR}/../../vendor
// #cgo CPPFLAGS: -I${SRCDIR}/../../../../ml/backend/ggml/ggml/include
import "C"

// WHY: tip chat.cpp specialized parsers live under common/parsers/ (subdir of common).
