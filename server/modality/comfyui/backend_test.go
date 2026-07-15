package comfyui

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveWorkflowDirAbsolute(t *testing.T) {
	if got, want := resolveWorkflowDir("/abs/path/x"), "/abs/path/x"; got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestResolveWorkflowDirHome(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip(err)
	}
	got := resolveWorkflowDir("~/comfy/qwen-image")
	want := filepath.Join(home, "comfy/qwen-image")
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestResolveWorkflowDirRelativeWithoutRoot(t *testing.T) {
	t.Setenv("OLLAMA_COMFYUI_WORKFLOWS_ROOT", "")
	if got, want := resolveWorkflowDir("scripts/comfyui/qwen-image"), "scripts/comfyui/qwen-image"; got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestResolveWorkflowDirRelativeWithRoot(t *testing.T) {
	t.Setenv("OLLAMA_COMFYUI_WORKFLOWS_ROOT", "/srv/zerollama")
	got := resolveWorkflowDir("scripts/comfyui/qwen-image")
	want := filepath.Join("/srv/zerollama", "scripts/comfyui/qwen-image")
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}
