package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/llm"
)

const llamaPatchSeqCopyMarker = `"/kv/seq-copy"`

func doctorCheckLlamaPatches(repo string) doctorCheck {
	name := "llama.cpp patches"
	fix := "./scripts/vendor/llama_patch_doctor.sh"

	inTree := filepath.Join(repo, "llama", "llama.cpp", "tools", "server", "server.cpp")
	data, err := os.ReadFile(inTree)
	if err != nil {
		return doctorCheck{
			Name:    name,
			Status:  "fail",
			Detail:  "in-tree server.cpp missing — sync vendor: ./scripts/vendor/rebase_vendor_unified.sh --sync",
			FixHint: fix,
		}
	}
	if !strings.Contains(string(data), llamaPatchSeqCopyMarker) {
		return doctorCheck{
			Name:    name,
			Status:  "fail",
			Detail:  "in-tree llama.cpp lacks POST /kv/seq-copy (kv-seq-copy patch not synced)",
			FixHint: "./scripts/vendor/rebase_vendor_unified.sh --apply --sync",
		}
	}

	patchDir := filepath.Join(repo, "llama", "patches")
	// Match by content stem (numbers shift as patches are inserted).
	required := []string{
		"ollama-llama-kv-ext",
		"ollama-kv-seq-copy-endpoint",
	}
	var missing []string
	for _, sub := range required {
		found := false
		entries, _ := os.ReadDir(patchDir)
		for _, e := range entries {
			if strings.Contains(e.Name(), sub) {
				found = true
				break
			}
		}
		if !found {
			missing = append(missing, sub)
		}
	}
	if len(missing) > 0 {
		return doctorCheck{
			Name:    name,
			Status:  "fail",
			Detail:  "missing patch files: " + strings.Join(missing, ", "),
			FixHint: fix,
		}
	}

	detail := "in-tree /kv/seq-copy OK"
	status := "ok"

	if head, expected, ok := doctorVendorHeadMatches(repo); ok {
		if expected != "" && head != expected {
			status = "warn"
			detail += fmt.Sprintf("; vendor HEAD %s != expected %s", head[:12], expected[:12])
		} else if head != "" {
			detail += fmt.Sprintf("; vendor +patches @ %s", head[:12])
		}
	}

	if bin, err := llm.FindLlamaServer(); err == nil && bin != "" {
		if doctorBinaryEmbedsSeqCopy(bin) {
			detail += "; llama-server embeds /kv/seq-copy"
		} else {
			status = "warn"
			detail += "; llama-server binary missing /kv/seq-copy — rebuild: ./scripts/build/build_llama_server.sh"
		}
	}

	return doctorCheck{
		Name:    name,
		Status:  status,
		Detail:  detail,
		FixHint: fix,
	}
}

func doctorExpectedVendorHead(repo string) string {
	data, err := os.ReadFile(filepath.Join(repo, "LLAMA_CPP_VENDOR_HEAD"))
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if len(line) >= 40 {
			return line
		}
	}
	return ""
}

func doctorVendorHeadMatches(repo string) (head, expected string, vendorPresent bool) {
	fetchHead := doctorMakefileSyncFetchHead(repo)
	vendor := filepath.Join(repo, "vendor", "llama-cpp-"+fetchHead)
	if st, err := os.Stat(filepath.Join(vendor, ".git")); err != nil || !st.IsDir() {
		return "", doctorExpectedVendorHead(repo), false
	}
	out, err := exec.Command("git", "-C", vendor, "rev-parse", "HEAD").Output()
	if err != nil {
		return "", doctorExpectedVendorHead(repo), true
	}
	return strings.TrimSpace(string(out)), doctorExpectedVendorHead(repo), true
}

func doctorBinaryEmbedsSeqCopy(bin string) bool {
	candidates := []string{bin}
	// Thin llama-server launchers keep route strings in libllama-server-impl.
	dir := filepath.Dir(bin)
	for _, name := range []string{
		"libllama-server-impl.dylib",
		"libllama-server-impl.so",
		"llama-server.exe",
	} {
		candidates = append(candidates, filepath.Join(dir, name))
	}
	for _, path := range candidates {
		if doctorFileContainsSeqCopy(path) {
			return true
		}
	}
	return false
}

func doctorFileContainsSeqCopy(path string) bool {
	st, err := os.Stat(path)
	if err != nil || st.IsDir() {
		return false
	}
	out, err := exec.Command("strings", path).Output()
	if err != nil {
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return false
		}
		return strings.Contains(string(data), "/kv/seq-copy") || strings.Contains(string(data), "kv/seq-copy")
	}
	return strings.Contains(string(out), "/kv/seq-copy") || strings.Contains(string(out), "kv/seq-copy")
}
