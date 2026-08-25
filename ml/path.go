package ml

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

type libOllamaPathSearch struct {
	executable     string
	workingDir     string
	goos           string
	goarch         string
	libraryPathEnv string
	// systemPaths, if non-nil, replaces the default Linux package locations.
	// Tests pass an empty slice to disable /usr/lib/ollama.
	systemPaths *[]string
}

// LibOllamaPath is a path to lookup dynamic libraries.
// In development it's usually 'build/lib/ollama'.
// In distribution builds it's 'lib/ollama' on Windows,
// '../lib/ollama' on Linux, and the executable's directory on macOS.
// GPU-specific libraries live in subdirectories such as cuda_v12, cuda_v13, rocm, vulkan.
//
// Linux extras (zerollama): OLLAMA_LIBRARY_PATH roots and package installs
// (/usr/lib/ollama, /usr/local/lib/ollama) so a binary in run/zerollama still
// finds packaged CUDA plugins.
var LibOllamaPath string = func() string {
	return findLibOllamaPath(liveLibOllamaPathSearch())
}()

func liveLibOllamaPathSearch() libOllamaPathSearch {
	exe, err := os.Executable()
	if err != nil {
		exe = ""
	} else if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}
	cwd, err := os.Getwd()
	if err != nil {
		cwd = ""
	}
	return libOllamaPathSearch{
		executable:     exe,
		workingDir:     cwd,
		goos:           runtime.GOOS,
		goarch:         runtime.GOARCH,
		libraryPathEnv: os.Getenv("OLLAMA_LIBRARY_PATH"),
	}
}

// LibOllamaSearchRoots returns existing directories to scan for ggml GPU plugins.
// Includes LibOllamaPath, OLLAMA_LIBRARY_PATH entries, sibling layouts, and
// Linux package paths. Re-reads the environment so discovery matches the serve wrapper.
func LibOllamaSearchRoots() []string {
	roots := existingLibOllamaRoots(liveLibOllamaPathSearch())
	if LibOllamaPath != "" {
		found := false
		for _, r := range roots {
			if r == LibOllamaPath {
				found = true
				break
			}
		}
		if !found {
			roots = append([]string{LibOllamaPath}, roots...)
		}
	}
	return roots
}

func findLibOllamaPath(search libOllamaPathSearch) string {
	existing := existingLibOllamaRoots(search)
	for _, path := range existing {
		if libOllamaHasGPUPluginDir(path) {
			return path
		}
	}
	if len(existing) > 0 {
		return existing[0]
	}
	if search.executable != "" {
		return filepath.Dir(search.executable)
	}
	return ""
}

func existingLibOllamaRoots(search libOllamaPathSearch) []string {
	var existing []string
	for _, path := range libOllamaPathCandidates(search) {
		if libOllamaPathExists(path) {
			existing = append(existing, path)
		}
	}
	return existing
}

func libOllamaPathCandidates(search libOllamaPathSearch) []string {
	goos := search.goos
	if goos == "" {
		goos = runtime.GOOS
	}
	goarch := search.goarch
	if goarch == "" {
		goarch = runtime.GOARCH
	}

	seen := map[string]bool{}
	var candidates []string
	add := func(path string) {
		if path == "" {
			return
		}
		path = filepath.Clean(path)
		if !seen[path] {
			seen[path] = true
			candidates = append(candidates, path)
		}
	}

	// Explicit override first — serve wrappers already set OLLAMA_LIBRARY_PATH
	// to /usr/lib/ollama:/usr/lib/ollama/cuda_v13 even when the binary lives in run/.
	for _, entry := range filepath.SplitList(search.libraryPathEnv) {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		add(libOllamaRootFromPath(entry))
		add(entry)
	}

	if search.executable != "" {
		exeDir := filepath.Dir(search.executable)
		switch goos {
		case "darwin":
			add(filepath.Join(exeDir, "lib", "ollama"))
			add(filepath.Join(exeDir, "..", "lib", "ollama"))
		case "linux":
			add(filepath.Join(exeDir, "..", "lib", "ollama"))
			add(filepath.Join(exeDir, "lib", "ollama"))
		case "windows":
			add(filepath.Join(exeDir, "lib", "ollama"))
			add(filepath.Join(exeDir, "..", "lib", "ollama"))
		default:
			add(filepath.Join(exeDir, "lib", "ollama"))
			add(filepath.Join(exeDir, "..", "lib", "ollama"))
		}
		addLocalLibOllamaPaths(add, exeDir, goos, goarch)
		if goos == "darwin" {
			add(exeDir)
		}
	}
	addLocalLibOllamaPaths(add, search.workingDir, goos, goarch)

	if goos == "linux" {
		for _, path := range linuxSystemLibOllamaPaths(search) {
			add(path)
		}
	}

	return candidates
}

func addLocalLibOllamaPaths(add func(string), base, goos, goarch string) {
	if base == "" {
		return
	}
	add(filepath.Join(base, "build", "lib", "ollama"))
	add(filepath.Join(base, "dist", goos+"-"+goarch, "lib", "ollama"))
	if goos+"_"+goarch != goos+"-"+goarch {
		add(filepath.Join(base, "dist", goos+"_"+goarch, "lib", "ollama"))
	}
	if goos == "darwin" {
		add(filepath.Join(base, "dist", "darwin"))
	}
}

func linuxSystemLibOllamaPaths(search libOllamaPathSearch) []string {
	if search.systemPaths != nil {
		return *search.systemPaths
	}
	return []string{"/usr/lib/ollama", "/usr/local/lib/ollama"}
}

func libOllamaRootFromPath(path string) string {
	if IsGPUPluginDir(filepath.Base(path)) {
		return filepath.Dir(path)
	}
	return path
}

// IsGPUPluginDir reports whether name is a ggml/MLX backend subdirectory
// (cuda_v13, vulkan, rocm, mlx_cuda_v12, …).
func IsGPUPluginDir(name string) bool {
	n := strings.ToLower(name)
	switch {
	case strings.HasPrefix(n, "cuda_"):
		return true
	case strings.HasPrefix(n, "rocm"):
		return true
	case n == "vulkan" || n == "metal":
		return true
	case strings.HasPrefix(n, "mlx_"):
		return true
	case strings.HasPrefix(n, "hip"):
		return true
	default:
		return false
	}
}

func libOllamaHasGPUPluginDir(root string) bool {
	if IsGPUPluginDir(filepath.Base(root)) {
		return true
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if e.IsDir() && IsGPUPluginDir(e.Name()) {
			return true
		}
	}
	return false
}

func libOllamaPathExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}
