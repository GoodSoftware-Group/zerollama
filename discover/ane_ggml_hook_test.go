package discover

import (
	"runtime"
	"testing"
)

func TestProbeGGMLIOSurfaceHookStatusDarwin(t *testing.T) {
	st := ProbeGGMLIOSurfaceHookStatus()
	if st.CFunction == "" || st.BackendFunction == "" {
		t.Fatal("expected hook function names")
	}
	if runtime.GOOS == "darwin" && !ggmlIOSurfaceHookInTree() {
		t.Fatal("expected in-tree ggml IOSurface API on darwin")
	}
	if runtime.GOOS != "darwin" && st.APIAvailable {
		t.Fatal("API should not be available off darwin")
	}
}
