package comfyui

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/ollama/ollama/envconfig"
)

// TestE2ESmoke exercises Generate against a real, already-running ComfyUI server.
// Skipped unless RUN_E2E_COMFY=1; point OLLAMA_COMFYUI_URL at your instance and set
// COMFY_E2E_WORKFLOW_DIR to a workflow directory with a small/fast model already
// installed (e.g. a Comfy-side Z-Image or SD-Turbo checkpoint) before running:
//
//	RUN_E2E_COMFY=1 OLLAMA_COMFYUI_URL=http://127.0.0.1:8188 \
//	  COMFY_E2E_WORKFLOW_DIR=/path/to/workflow/dir \
//	  go test ./server/modality/comfyui/... -run TestE2ESmoke -v
func TestE2ESmoke(t *testing.T) {
	if os.Getenv("RUN_E2E_COMFY") != "1" {
		t.Skip("set RUN_E2E_COMFY=1 to run against a real ComfyUI server")
	}
	dir := os.Getenv("COMFY_E2E_WORKFLOW_DIR")
	if dir == "" {
		t.Skip("set COMFY_E2E_WORKFLOW_DIR to a workflow template directory")
	}

	t.Logf("using ComfyUI at %s", envconfig.ComfyUIURL())

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	result, err := Generate(ctx, Request{
		WorkflowDir: dir,
		Prompt:      "a small red cube on a white background",
		Width:       512,
		Height:      512,
		Steps:       4,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(result.PNG) == 0 {
		t.Fatal("expected non-empty PNG output")
	}
}
