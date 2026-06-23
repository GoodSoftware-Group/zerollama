//go:build edge

package runner

import "fmt"

// Execute is stubbed in edge builds — GGUF inference uses Go → llama-server only.
// WHY fail at subprocess entry: the main binary still exposes `zerollama runner` for API compat;
// edge builds must not spawn a hidden ggml child when llama-server routing is required.
func Execute(args []string) error {
	return fmt.Errorf("ggml runner is not included in edge builds; set ZEROLLAMA_LLAMA_SERVER=1/auto or use zerollama serve --edge with llama-server on disk")
}
