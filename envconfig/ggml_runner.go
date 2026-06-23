//go:build !edge

package envconfig

// GgmlRunnerLinked reports whether in-process / subprocess ggml inference is compiled in.
// WHY compile-time not env: edge operators need a smaller binary without llama.cpp CGO even if
// someone forgets ZEROLLAMA_LLAMA_SERVER=1 — link-time exclusion is stronger than runtime checks.
func GgmlRunnerLinked() bool {
	return true
}
