package discover

import (
	"testing"
)

// TestParseLlamaServerDevices_FreeMemoryFromListDevices locks that Go discovery
// copies llama-server --list-devices "MiB free" into DeviceInfo.FreeMemory
// (minefield trap 96 awareness). See docs/model-serving-minefield.md.
func TestParseLlamaServerDevices_FreeMemoryFromListDevices(t *testing.T) {
	output := `Available devices:
  CUDA0: NVIDIA GeForce RTX 4090 (24564 MiB, 12345 MiB free)
  Metal: Apple M3 Max (98304 MiB, 50000 MiB free)
`
	devices := parseLlamaServerDevices(output, []string{"/lib/ollama", "/lib/ollama/cuda_v12"})
	if len(devices) != 2 {
		t.Fatalf("got %d devices, want 2", len(devices))
	}

	wantFree := []uint64{
		12345 * 1024 * 1024,
		50000 * 1024 * 1024,
	}
	for i, d := range devices {
		if d.FreeMemory != wantFree[i] {
			t.Errorf("device %d (%s) FreeMemory = %d, want %d",
				i, d.Name, d.FreeMemory, wantFree[i])
		}
	}
}

// TestParseLlamaServerDevices_ClampsFreeExceedingTotal locks trap 96 mitigation:
// free > total is clamped to total (cannot be a valid per-device free figure).
func TestParseLlamaServerDevices_ClampsFreeExceedingTotal(t *testing.T) {
	output := `Available devices:
  CUDA0: NVIDIA GeForce RTX 4090 (24564 MiB, 65536 MiB free)
`
	devices := parseLlamaServerDevices(output, []string{"/lib/ollama", "/lib/ollama/cuda_v12"})
	if len(devices) != 1 {
		t.Fatalf("got %d devices", len(devices))
	}
	d := devices[0]
	want := uint64(24564) * 1024 * 1024
	if d.FreeMemory != want {
		t.Fatalf("FreeMemory = %d, want clamped to total %d", d.FreeMemory, want)
	}
	if d.FreeMemory > d.TotalMemory {
		t.Fatal("free must not exceed total after clamp")
	}
}

func TestPreferNativeDeviceFree(t *testing.T) {
	got, ok := preferNativeDeviceFree("CUDA", 20<<30, 24<<30, 10<<30)
	if !ok || got != 10<<30 {
		t.Fatalf("cuda prefer native: got=%d ok=%v", got, ok)
	}
	got, ok = preferNativeDeviceFree("ROCm", 20<<30, 24<<30, 30<<30) // native > total → clamp
	if !ok || got != 24<<30 {
		t.Fatalf("rocm clamp native: got=%d ok=%v", got, ok)
	}
	if _, ok := preferNativeDeviceFree("Metal", 20<<30, 24<<30, 10<<30); ok {
		t.Fatal("Metal must keep list-devices/unified path")
	}
	if _, ok := preferNativeDeviceFree("CUDA", 20<<30, 24<<30, 0); ok {
		t.Fatal("zero native free is not a preference")
	}
}
