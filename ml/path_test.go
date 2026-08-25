package ml

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFindLibOllamaPath(t *testing.T) {
	root := t.TempDir()
	none := []string{}

	tests := []struct {
		name   string
		search libOllamaPathSearch
		dirs   []string
		want   string
	}{
		{
			name: "linux standard install layout",
			search: libOllamaPathSearch{
				executable:  filepath.Join(root, "linux-install", "bin", "ollama"),
				goos:        "linux",
				goarch:      "amd64",
				systemPaths: &none,
			},
			dirs: []string{filepath.Join(root, "linux-install", "lib", "ollama")},
			want: filepath.Join(root, "linux-install", "lib", "ollama"),
		},
		{
			name: "run/zerollama honors OLLAMA_LIBRARY_PATH when ../lib/ollama is missing",
			search: libOllamaPathSearch{
				executable:     filepath.Join(root, "repo", "run", "zerollama"),
				workingDir:     filepath.Join(root, "repo"),
				goos:           "linux",
				goarch:         "amd64",
				libraryPathEnv: filepath.Join(root, "usr", "lib", "ollama") + string(filepath.ListSeparator) + filepath.Join(root, "usr", "lib", "ollama", "cuda_v13"),
				systemPaths:    &none,
			},
			dirs: []string{
				filepath.Join(root, "usr", "lib", "ollama"),
				filepath.Join(root, "usr", "lib", "ollama", "cuda_v13"),
			},
			want: filepath.Join(root, "usr", "lib", "ollama"),
		},
		{
			name: "run/zerollama falls back to packaged /usr/lib/ollama",
			search: libOllamaPathSearch{
				executable:  filepath.Join(root, "repo", "run", "zerollama"),
				workingDir:  filepath.Join(root, "repo"),
				goos:        "linux",
				goarch:      "amd64",
				systemPaths: &[]string{filepath.Join(root, "usr", "lib", "ollama")},
			},
			dirs: []string{
				filepath.Join(root, "usr", "lib", "ollama"),
				filepath.Join(root, "usr", "lib", "ollama", "cuda_v13"),
			},
			want: filepath.Join(root, "usr", "lib", "ollama"),
		},
		{
			name: "empty build/lib/ollama loses to system CUDA plugins",
			search: libOllamaPathSearch{
				executable:  filepath.Join(root, "mixed", "run", "zerollama"),
				workingDir:  filepath.Join(root, "mixed"),
				goos:        "linux",
				goarch:      "amd64",
				systemPaths: &[]string{filepath.Join(root, "mixed", "usr", "lib", "ollama")},
			},
			dirs: []string{
				filepath.Join(root, "mixed", "build", "lib", "ollama"),
				filepath.Join(root, "mixed", "usr", "lib", "ollama", "cuda_v13"),
			},
			want: filepath.Join(root, "mixed", "usr", "lib", "ollama"),
		},
		{
			name: "OLLAMA_LIBRARY_PATH overrides sibling lib/ollama",
			search: libOllamaPathSearch{
				executable:     filepath.Join(root, "override", "bin", "zerollama"),
				goos:           "linux",
				goarch:         "amd64",
				libraryPathEnv: filepath.Join(root, "override", "opt", "lib", "ollama"),
				systemPaths:    &none,
			},
			dirs: []string{
				filepath.Join(root, "override", "lib", "ollama", "cuda_v12"),
				filepath.Join(root, "override", "opt", "lib", "ollama", "cuda_v13"),
			},
			want: filepath.Join(root, "override", "opt", "lib", "ollama"),
		},
		{
			name: "darwin local build layout before executable directory fallback",
			search: libOllamaPathSearch{
				executable:  filepath.Join(root, "darwin-dev", "ollama"),
				workingDir:  filepath.Join(root, "darwin-dev"),
				goos:        "darwin",
				goarch:      "arm64",
				systemPaths: &none,
			},
			dirs: []string{filepath.Join(root, "darwin-dev", "build", "lib", "ollama")},
			want: filepath.Join(root, "darwin-dev", "build", "lib", "ollama"),
		},
		{
			name: "windows release layout",
			search: libOllamaPathSearch{
				executable:  filepath.Join(root, "windows-release", "ollama.exe"),
				goos:        "windows",
				goarch:      "amd64",
				systemPaths: &none,
			},
			dirs: []string{filepath.Join(root, "windows-release", "lib", "ollama")},
			want: filepath.Join(root, "windows-release", "lib", "ollama"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, dir := range tt.dirs {
				if err := os.MkdirAll(dir, 0o755); err != nil {
					t.Fatal(err)
				}
			}

			got := findLibOllamaPath(tt.search)
			if got != tt.want {
				t.Fatalf("findLibOllamaPath() = %q, want %q; candidates: %v", got, tt.want, libOllamaPathCandidates(tt.search))
			}
		})
	}
}

func TestFindLibOllamaPathFallsBackToExeDir(t *testing.T) {
	root := t.TempDir()
	none := []string{}
	exe := filepath.Join(root, "run", "zerollama")
	if err := os.MkdirAll(filepath.Dir(exe), 0o755); err != nil {
		t.Fatal(err)
	}
	got := findLibOllamaPath(libOllamaPathSearch{
		executable:  exe,
		goos:        "linux",
		goarch:      "amd64",
		systemPaths: &none,
	})
	if got != filepath.Dir(exe) {
		t.Fatalf("got %q, want exe dir %q", got, filepath.Dir(exe))
	}
}

func TestIsGPUPluginDir(t *testing.T) {
	for name, want := range map[string]bool{
		"cuda_v13":     true,
		"cuda_v12":     true,
		"cuda_jetpack": true,
		"vulkan":       true,
		"rocm":         true,
		"rocm_v7_2":    true,
		"mlx_cuda_v12": true,
		"ollama":       false,
		"bin":          false,
	} {
		if got := IsGPUPluginDir(name); got != want {
			t.Fatalf("IsGPUPluginDir(%q) = %v, want %v", name, got, want)
		}
	}
}

func TestLibOllamaRootFromPath(t *testing.T) {
	got := libOllamaRootFromPath("/usr/lib/ollama/cuda_v13")
	if got != "/usr/lib/ollama" && got != filepath.Clean("/usr/lib/ollama") {
		// Windows would differ; linux/darwin keep slash form after Dir.
		want := filepath.Dir("/usr/lib/ollama/cuda_v13")
		if got != want {
			t.Fatalf("got %q, want %q", got, want)
		}
	}
}
