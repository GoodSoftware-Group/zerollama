//go:build arm64

package polarquant

// #cgo CFLAGS: -O3
// #cgo CPPFLAGS: -I${SRCDIR}/.. -I${SRCDIR}/../.. -I${SRCDIR}/../../../include -I${SRCDIR}
// #cgo CPPFLAGS: -DPOLARQUANT_HAVE_NEON=1
import "C"
