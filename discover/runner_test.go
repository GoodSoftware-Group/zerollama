package discover

import (
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/ollama/ollama/ml"
)

func init() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))
	slog.SetDefault(logger)
}

func TestFilterOverlapByLibrary(t *testing.T) {
	type testcase struct {
		name string
		inp  map[string]map[string]map[string]int
		exp  []bool
	}
	for _, tc := range []testcase{
		{
			name: "empty",
			inp:  map[string]map[string]map[string]int{},
			exp:  []bool{}, // needs deletion
		},
		{
			name: "single no overlap",
			inp: map[string]map[string]map[string]int{
				"CUDA": {
					"cuda_v12": {
						"GPU-d7b00605-c0c8-152d-529d-e03726d5dc52": 0,
					},
				},
			},
			exp: []bool{false},
		},
		{
			name: "100% overlap pick 2nd",
			inp: map[string]map[string]map[string]int{
				"CUDA": {
					"cuda_v12": {
						"GPU-d7b00605-c0c8-152d-529d-e03726d5dc52": 0,
						"GPU-cd6c3216-03d2-a8eb-8235-2ffbf571712e": 1,
					},
					"cuda_v13": {
						"GPU-d7b00605-c0c8-152d-529d-e03726d5dc52": 2,
						"GPU-cd6c3216-03d2-a8eb-8235-2ffbf571712e": 3,
					},
				},
			},
			exp: []bool{true, true, false, false},
		},
		{
			name: "100% overlap pick 1st",
			inp: map[string]map[string]map[string]int{
				"CUDA": {
					"cuda_v13": {
						"GPU-d7b00605-c0c8-152d-529d-e03726d5dc52": 0,
						"GPU-cd6c3216-03d2-a8eb-8235-2ffbf571712e": 1,
					},
					"cuda_v12": {
						"GPU-d7b00605-c0c8-152d-529d-e03726d5dc52": 2,
						"GPU-cd6c3216-03d2-a8eb-8235-2ffbf571712e": 3,
					},
				},
			},
			exp: []bool{false, false, true, true},
		},
		{
			name: "partial overlap pick older",
			inp: map[string]map[string]map[string]int{
				"CUDA": {
					"cuda_v13": {
						"GPU-d7b00605-c0c8-152d-529d-e03726d5dc52": 0,
					},
					"cuda_v12": {
						"GPU-d7b00605-c0c8-152d-529d-e03726d5dc52": 1,
						"GPU-cd6c3216-03d2-a8eb-8235-2ffbf571712e": 2,
					},
				},
			},
			exp: []bool{true, false, false},
		},
		{
			name: "no overlap",
			inp: map[string]map[string]map[string]int{
				"CUDA": {
					"cuda_v13": {
						"GPU-d7b00605-c0c8-152d-529d-e03726d5dc52": 0,
					},
					"cuda_v12": {
						"GPU-cd6c3216-03d2-a8eb-8235-2ffbf571712e": 1,
					},
				},
			},
			exp: []bool{false, false},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			needsDelete := make([]bool, len(tc.exp))
			filterOverlapByLibrary(tc.inp, needsDelete)
			for i, exp := range tc.exp {
				if needsDelete[i] != exp {
					t.Fatalf("expected: %v\ngot: %v", tc.exp, needsDelete)
				}
			}
		})
	}
}

func TestRecordPersistentRunnerEnv(t *testing.T) {
	devices := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{Library: "Metal", ID: "0"}},
		{DeviceID: ml.DeviceID{Library: "CUDA", ID: "1"}},
	}

	recordPersistentRunnerEnv(devices, map[string]string{
		"GGML_METAL_TENSOR_DISABLE": "1",
		"CUDA_VISIBLE_DEVICES":      "1",
	})

	if got := devices[0].RunnerEnvOverrides["GGML_METAL_TENSOR_DISABLE"]; got != "1" {
		t.Fatalf("Metal RunnerEnvOverrides = %q, want %q", got, "1")
	}

	if _, ok := devices[0].RunnerEnvOverrides["CUDA_VISIBLE_DEVICES"]; ok {
		t.Fatal("unexpected CUDA_VISIBLE_DEVICES in Metal RunnerEnvOverrides")
	}

	if devices[1].RunnerEnvOverrides != nil {
		t.Fatalf("unexpected RunnerEnvOverrides recorded for non-Metal device: %#v", devices[1].RunnerEnvOverrides)
	}
}

func TestFilterIntegratedGPUs(t *testing.T) {
	devices := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{Library: "CUDA", ID: "0"}, Description: "NVIDIA integrated", Integrated: true},
		{DeviceID: ml.DeviceID{Library: "Metal", ID: "0"}, Description: "Apple GPU", Integrated: true},
		{DeviceID: ml.DeviceID{Library: "Vulkan", ID: "0"}, Description: "AMD Radeon(TM) Graphics", Integrated: true},
		{DeviceID: ml.DeviceID{Library: "ROCm", ID: "0"}, Description: "AMD Radeon 780M", Integrated: true, GFXTarget: "gfx1103"},
		{DeviceID: ml.DeviceID{Library: "ROCm", ID: "1"}, Description: "AMD Radeon 8060S Graphics", Integrated: true, GFXTarget: "gfx1151"},
		{DeviceID: ml.DeviceID{Library: "Vulkan", ID: "1"}, Description: "AMD Radeon RX 6800"},
	}

	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		t.Setenv("OLLAMA_IGPU_ENABLE", "false")
		got := filterIntegratedGPUs(append([]ml.DeviceInfo{}, devices...))
		want := []ml.DeviceID{
			{Library: "CUDA", ID: "0"},
			{Library: "Metal", ID: "0"},
			{Library: "Vulkan", ID: "0"},
			{Library: "ROCm", ID: "0"},
			{Library: "ROCm", ID: "1"},
			{Library: "Vulkan", ID: "1"},
		}
		assertDeviceIDs(t, got, want)
		return
	}

	t.Run("auto admits only allowlisted integrated GPUs", func(t *testing.T) {
		got := filterIntegratedGPUs(append([]ml.DeviceInfo{}, devices...))
		want := []ml.DeviceID{
			{Library: "CUDA", ID: "0"},
			{Library: "ROCm", ID: "1"},
			{Library: "Vulkan", ID: "1"},
		}
		assertDeviceIDs(t, got, want)
	})

	t.Run("explicit true admits all integrated GPUs", func(t *testing.T) {
		t.Setenv("OLLAMA_IGPU_ENABLE", "true")
		got := filterIntegratedGPUs(append([]ml.DeviceInfo{}, devices...))
		want := []ml.DeviceID{
			{Library: "CUDA", ID: "0"},
			{Library: "Metal", ID: "0"},
			{Library: "Vulkan", ID: "0"},
			{Library: "ROCm", ID: "0"},
			{Library: "ROCm", ID: "1"},
			{Library: "Vulkan", ID: "1"},
		}
		assertDeviceIDs(t, got, want)
	})

	t.Run("explicit false drops integrated GPUs", func(t *testing.T) {
		t.Setenv("OLLAMA_IGPU_ENABLE", "false")
		got := filterIntegratedGPUs(append([]ml.DeviceInfo{}, devices...))
		want := []ml.DeviceID{{Library: "Vulkan", ID: "1"}}
		assertDeviceIDs(t, got, want)
	})
}

func assertDeviceIDs(t *testing.T, got []ml.DeviceInfo, want []ml.DeviceID) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d devices, want %d: %#v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].DeviceID != want[i] {
			t.Fatalf("device %d = %#v, want %#v", i, got[i].DeviceID, want[i])
		}
	}
}

func TestIsDiscoverableRunnerLib(t *testing.T) {
	cases := []struct {
		path string
		want bool
	}{
		{"tools/ane-ggml-map/ane-ggml-map-smoke", false},
		{"build/arm64-cpu/lib/ollama/libggml-base.dylib", true},
		{"lib/ollama/cuda_v12/libggml-cuda.so", true},
		{"lib/ollama/cuda_v12/libggml-cuda.dll", true},
		{"notes/ggml-b9509-migration.md", false},
	}
	for _, tc := range cases {
		if got := isDiscoverableRunnerLib(tc.path); got != tc.want {
			t.Fatalf("isDiscoverableRunnerLib(%q) = %v, want %v", tc.path, got, tc.want)
		}
	}
}

func TestCollectRunnerLibDirs(t *testing.T) {
	root := t.TempDir()
	cuda := filepath.Join(root, "cuda_v13")
	if err := os.MkdirAll(cuda, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cuda, "libggml-cuda.so"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	dirs := collectRunnerLibDirs([]string{root})
	if _, ok := dirs[cuda]; !ok {
		t.Fatalf("expected plugin dir %q in %v", cuda, dirs)
	}

	pluginRoot := t.TempDir()
	if err := os.WriteFile(filepath.Join(pluginRoot, "libggml-cuda.so"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	dirs = collectRunnerLibDirs([]string{pluginRoot})
	if _, ok := dirs[pluginRoot]; !ok {
		t.Fatalf("expected env plugin root %q in %v", pluginRoot, dirs)
	}
}

func TestLLMLibraryInSearch(t *testing.T) {
	root := t.TempDir()
	cuda := filepath.Join(root, "cuda_v13")
	if err := os.MkdirAll(cuda, 0o755); err != nil {
		t.Fatal(err)
	}

	if !llmLibraryInSearch("cuda_v13", map[string]struct{}{cuda: {}}, []string{root}) {
		t.Fatal("expected cuda_v13 in libDirs to count as searched")
	}
	if !llmLibraryInSearch("cuda_v13", map[string]struct{}{}, []string{root}) {
		t.Fatal("expected cuda_v13 subdir of search root to count as searched")
	}
	if llmLibraryInSearch("cuda_v13", map[string]struct{}{"": {}}, []string{t.TempDir()}) {
		t.Fatal("missing cuda_v13 should not count as searched")
	}
}

func TestBootstrapLibraryBase(t *testing.T) {
	dir := "/usr/lib/ollama/cuda_v13"
	if got := bootstrapLibraryBase(dir); got != "/usr/lib/ollama" && got != filepath.Dir(dir) {
		t.Fatalf("plugin parent: got %q", got)
	}
	if got := bootstrapLibraryBase(""); got != ml.LibOllamaPath {
		t.Fatalf("empty dir: got %q want LibOllamaPath", got)
	}
}

func TestBootstrapLibraryDirs(t *testing.T) {
	base := "/Users/me/zerollama"
	got := bootstrapLibraryDirs(base, "")
	if len(got) != 1 || got[0] != base {
		t.Fatalf("empty dir: %v", got)
	}
	got = bootstrapLibraryDirs(base, base)
	if len(got) != 1 || got[0] != base {
		t.Fatalf("same dir: %v", got)
	}
	got = bootstrapLibraryDirs(base, base+"/cuda_v12")
	if len(got) != 2 || got[1] != base+"/cuda_v12" {
		t.Fatalf("plugin dir: %v", got)
	}
}

func TestSkipSubprocessVRAMRefresh(t *testing.T) {
	metal := []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "Metal"}}}
	cuda := []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}}}
	if runtime.GOOS == "darwin" {
		if !skipSubprocessVRAMRefresh(metal) {
			t.Fatal("darwin Metal should skip subprocess VRAM refresh")
		}
		if skipSubprocessVRAMRefresh(cuda) {
			t.Fatal("darwin CUDA should not skip subprocess VRAM refresh")
		}
		return
	}
	if skipSubprocessVRAMRefresh(metal) || skipSubprocessVRAMRefresh(cuda) {
		t.Fatal("non-darwin should not skip subprocess VRAM refresh")
	}
}
