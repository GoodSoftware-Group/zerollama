//go:build edge

package envconfig

// GgmlRunnerLinked is false for -tags edge binaries (Phase 16 v1 subprocess stub + v2 CGO drop).
// WHY false at compile time: paired with server.go //go:build !edge so edge artifacts cannot
// accidentally load ggml even when env is misconfigured.
func GgmlRunnerLinked() bool {
	return false
}
