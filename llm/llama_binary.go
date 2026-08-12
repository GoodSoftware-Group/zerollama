package llm

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/ollama/ollama/ml"
)

type llamaCppBinarySearch struct {
	libOllamaPath string
	executable    string
	workingDir    string
	goos          string
	goarch        string
}

func defaultLlamaCppBinarySearch() llamaCppBinarySearch {
	executable, _ := os.Executable()
	if executable != "" {
		if eval, err := filepath.EvalSymlinks(executable); err == nil {
			executable = eval
		}
	}

	workingDir, _ := os.Getwd()
	return llamaCppBinarySearch{
		libOllamaPath: ml.LibOllamaPath,
		executable:    executable,
		workingDir:    workingDir,
		goos:          runtime.GOOS,
		goarch:        runtime.GOARCH,
	}
}

// FindLlamaCppBinary locates a llama.cpp helper binary in installed and local
// development layouts.
func FindLlamaCppBinary(name string) (string, error) {
	path, candidates, err := findLlamaCppBinary(name, defaultLlamaCppBinarySearch())
	if err != nil {
		return "", fmt.Errorf("%s binary not found (checked: %s)", name, strings.Join(candidates, ", "))
	}
	return path, nil
}

func findLlamaCppBinary(name string, search llamaCppBinarySearch) (string, []string, error) {
	candidates := llamaCppBinaryCandidates(name, search)
	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path, candidates, nil
		}
	}

	return "", candidates, os.ErrNotExist
}

func llamaCppBinaryCandidates(name string, search llamaCppBinarySearch) []string {
	goos := search.goos
	if goos == "" {
		goos = runtime.GOOS
	}
	goarch := search.goarch
	if goarch == "" {
		goarch = runtime.GOARCH
	}

	suffix := llamaCppBinaryName(name, goos)
	seen := map[string]bool{}
	var candidates []string
	add := func(dir string) {
		if dir == "" {
			return
		}
		path := filepath.Clean(filepath.Join(dir, suffix))
		if !seen[path] {
			seen[path] = true
			candidates = append(candidates, path)
		}
	}

	add(search.libOllamaPath)

	addPackagedLayoutDirs := func(base string) {
		if base == "" {
			return
		}
		switch goos {
		case "darwin":
			// macOS tarballs and apps colocate llama.cpp helpers with ollama.
			add(base)
			// Per-architecture local dist output keeps helpers under lib/ollama.
			add(filepath.Join(base, "lib", "ollama"))
			// Standard CMake installs put ollama in bin/ and helpers in ../lib/ollama/.
			add(filepath.Join(base, "..", "lib", "ollama"))
		case "linux":
			// Linux packages install ollama in bin/ and helpers in ../lib/ollama/.
			add(filepath.Join(base, "..", "lib", "ollama"))
		case "windows":
			// Windows packages keep ollama.exe at top level with lib/ as a peer.
			add(filepath.Join(base, "lib", "ollama"))
			// Standard CMake installs put ollama.exe in bin/ and helpers in ../lib/ollama/.
			add(filepath.Join(base, "..", "lib", "ollama"))
		default:
			add(filepath.Join(base, "lib", "ollama"))
			add(filepath.Join(base, "..", "lib", "ollama"))
		}
	}

	addLocalLayoutDirs := func(base string) {
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

	if search.executable != "" {
		exeDir := filepath.Dir(search.executable)
		addPackagedLayoutDirs(exeDir)
		addLocalLayoutDirs(exeDir)
	}
	if search.workingDir != "" {
		addLocalLayoutDirs(search.workingDir)
	}

	addBuildOutputDirs := func(base string) {
		if base == "" {
			return
		}
		matches, _ := filepath.Glob(filepath.Join(base, "build", "llama-server-*", "bin"))
		// Include flash-moe build dirs always; llamaCppBuildOutputRank prefers them when
		// ZEROLLAMA_FLASH_MOE=1 — why: FindFlashMoELlamaServer and FindLlamaServer share candidates.
		if flash, _ := filepath.Glob(filepath.Join(base, "build", "flash-moe-llama-server-*", "bin")); len(flash) > 0 {
			matches = append(matches, flash...)
		}
		slices.SortFunc(matches, func(a, b string) int {
			if rank := llamaCppBuildOutputRank(a) - llamaCppBuildOutputRank(b); rank != 0 {
				return rank
			}
			return strings.Compare(a, b)
		})
		for _, m := range matches {
			add(m)
		}
	}
	if search.executable != "" {
		addBuildOutputDirs(filepath.Dir(search.executable))
	}
	addBuildOutputDirs(search.workingDir)

	return candidates
}

func llamaCppBinaryName(name, goos string) string {
	if goos == "windows" && filepath.Ext(name) == "" {
		return name + ".exe"
	}
	return name
}

// minUsableLlamaServerBytes rejects tiny shell/Mach-O stubs that pass the
// executable bit but cannot actually run (broken installs + test placeholders).
const minUsableLlamaServerBytes = 1 << 20 // 1 MiB

// minSplitLlamaServerBytes is the floor for cmake split builds: a small
// llama-server PIE plus libllama-server-impl.so in the same directory (~18 KiB
// on Linux). Below this we still treat the file as a placeholder stub.
const minSplitLlamaServerBytes = 8 << 10 // 8 KiB

// isUsableLlamaServerBin reports whether path is a runnable llama-server.
// Accepts either a monolithic binary (≥1 MiB) or a split build with
// libllama-server-impl.so beside the thin launcher (vendor/sibling CUDA pins).
func isUsableLlamaServerBin(path string) bool {
	fi, err := os.Stat(path)
	if err != nil || !fi.Mode().IsRegular() || (fi.Mode()&0o111) == 0 {
		return false
	}
	if fi.Size() >= minUsableLlamaServerBytes {
		return true
	}
	if fi.Size() < minSplitLlamaServerBytes {
		return false
	}
	impl := filepath.Join(filepath.Dir(path), "libllama-server-impl.so")
	ifi, err := os.Stat(impl)
	return err == nil && ifi.Mode().IsRegular() && ifi.Size() > 0
}

func llamaCppBuildOutputRank(path string) int {
	if strings.Contains(path, "flash-moe-llama-server") {
		if preferFlashMoELlamaServer() {
			return -1
		}
		return 3
	}
	if strings.Contains(path, "llama-server-darwin") ||
		strings.Contains(path, "llama-server-cuda") ||
		strings.Contains(path, "llama-server-rocm") {
		return 0
	}
	return 1
}
