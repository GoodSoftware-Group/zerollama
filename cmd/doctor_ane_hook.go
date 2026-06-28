package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ollama/ollama/llm"
)

const aneHookMarkerB7 = "B7 shadow step="
const aneHookMarkerTryDrive = "common_ane_draft_try_drive_token"

// doctorLlamaCppRootForANE returns sibling or vendor llama.cpp used for runtime builds.
func doctorLlamaCppRootForANE(repo string) string {
	if v := strings.TrimSpace(os.Getenv("LLAMA_CPP_ROOT")); v != "" {
		return filepath.Clean(v)
	}
	fetchHead := doctorMakefileSyncFetchHead(repo)
	vendor := filepath.Join(repo, "vendor", "llama-cpp-"+fetchHead)
	if st, err := os.Stat(filepath.Join(vendor, "CMakeLists.txt")); err == nil && !st.IsDir() {
		return vendor
	}
	return filepath.Clean(filepath.Join(repo, "..", "llama.cpp"))
}

func doctorMakefileSyncFetchHead(repo string) string {
	data, err := os.ReadFile(filepath.Join(repo, "Makefile.sync"))
	if err != nil {
		return "c84b3020"
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "FETCH_HEAD=") {
			return strings.TrimSpace(strings.TrimPrefix(line, "FETCH_HEAD="))
		}
	}
	return "c84b3020"
}

func doctorANESourceMarkers(root string) (specOK, filesOK bool, detail string) {
	specPath := filepath.Join(root, "common", "speculative.cpp")
	specData, err := os.ReadFile(specPath)
	if err != nil {
		return false, false, fmt.Sprintf("missing %s", specPath)
	}
	specText := string(specData)
	specOK = strings.Contains(specText, aneHookMarkerB7) &&
		strings.Contains(specText, aneHookMarkerTryDrive) &&
		strings.Contains(specText, "common_ane_draft_handoff_after_decode")

	required := []string{
		"ane_draft_hook.cpp",
		"ane_draft_hook.h",
		"ane_draft_session.mm",
	}
	var missing []string
	for _, f := range required {
		if st, err := os.Stat(filepath.Join(root, "common", f)); err != nil || st.IsDir() {
			missing = append(missing, f)
		}
	}
	filesOK = len(missing) == 0
	if !specOK {
		detail = "speculative.cpp missing ANE B7/handoff markers"
	} else if !filesOK {
		detail = "missing common/: " + strings.Join(missing, ", ")
	} else {
		detail = "sources synced"
	}
	return specOK, filesOK, detail
}

func doctorANEBinaryHasHook(repo string) (ok bool, detail string) {
	candidates := []string{}
	if p, err := llm.FindLlamaServer(); err == nil && p != "" {
		candidates = append(candidates, p)
	}
	root := doctorLlamaCppRootForANE(repo)
	candidates = append(candidates,
		filepath.Join(root, "build", "bin", "llama-server"),
		filepath.Join(root, "build", "bin", "libllama-common.0.0.1.dylib"),
		filepath.Join(root, "build", "bin", "libllama-common.dylib"),
	)
	for _, bin := range candidates {
		if st, err := os.Stat(bin); err != nil || st.IsDir() {
			continue
		}
		out, err := exec.Command("strings", bin).Output()
		if err != nil {
			continue
		}
		text := string(out)
		if strings.Contains(text, "B7 drive mode") || strings.Contains(text, aneHookMarkerB7) {
			return true, filepath.Base(bin)
		}
	}
	return false, "built llama-server/libllama-common lacks ANE B7 markers (stale build?)"
}

func doctorCheckANEDraftHook(repo string) doctorCheck {
	name := "ane draft hook (llama.cpp)"
	if runtime.GOOS != "darwin" {
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: "darwin-only lab hook; skipped",
		}
	}

	root := doctorLlamaCppRootForANE(repo)
	if st, err := os.Stat(filepath.Join(root, "CMakeLists.txt")); err != nil || st.IsDir() {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("llama.cpp not found at %s", root),
			FixHint: "./scripts/ensure_llama_cpp_sibling.sh && ./scripts/build_llama_server.sh",
		}
	}

	specOK, filesOK, srcDetail := doctorANESourceMarkers(root)
	binOK, binDetail := doctorANEBinaryHasHook(repo)

	switch {
	case specOK && filesOK && binOK:
		return doctorCheck{
			Name:   name,
			Status: "ok",
			Detail: fmt.Sprintf("%s @ %s; binary %s", srcDetail, filepath.Base(root), binDetail),
		}
	case specOK && filesOK && !binOK:
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("%s @ %s; %s", srcDetail, filepath.Base(root), binDetail),
			FixHint: "./scripts/build_llama_server.sh (auto-syncs ANE hook on Darwin)",
		}
	default:
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  fmt.Sprintf("%s @ %s", srcDetail, root),
			FixHint: "./scripts/sync_ane_hook_to_llama_cpp.sh && ./scripts/build_llama_server.sh",
		}
	}
}

func doctorSyncANEDraftHook(repo string) error {
	script := filepath.Join(repo, "scripts", "sync_ane_hook_to_llama_cpp.sh")
	if st, err := os.Stat(script); err != nil || st.IsDir() {
		return fmt.Errorf("missing %s", script)
	}
	fmt.Println("== doctor --fix: sync ANE draft hook to llama.cpp ==")
	cmd := exec.Command(script)
	cmd.Env = append(os.Environ(), "LLAMA_CPP_ROOT="+doctorLlamaCppRootForANE(repo))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
