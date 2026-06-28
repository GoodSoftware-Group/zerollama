//go:build amd64

package polarquant

// #cgo CFLAGS: -O3 -mavx2 -mfma
// #cgo CPPFLAGS: -I${SRCDIR}/.. -I${SRCDIR}/../.. -I${SRCDIR}/../../../include -I${SRCDIR}
// #cgo CPPFLAGS: -DPOLARQUANT_HAVE_AVX2=1
import "C"
