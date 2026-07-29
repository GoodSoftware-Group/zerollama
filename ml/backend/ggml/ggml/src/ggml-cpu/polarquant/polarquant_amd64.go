//go:build amd64

package polarquant

// #cgo CFLAGS: -O3 -march=x86-64-v3
// #cgo CPPFLAGS: -I${SRCDIR}/.. -I${SRCDIR}/../.. -I${SRCDIR}/../../../include -I${SRCDIR}
// #cgo CPPFLAGS: -DPOLARQUANT_HAVE_AVX2=1
import "C"

// WHY -march=x86-64-v3 (not -mavx2 -mfma): Go 1.25+ cgo rejects -mfma; v3 implies AVX2+FMA.
