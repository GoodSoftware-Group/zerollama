package llm

import (
	"bufio"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ollama/ollama/internal/reporoots"
)

const (
	defaultUnifiedLlamaCppCommit = "c84b30200c8d512c00c9d61c96bed078f1c0024d"
	unifiedLlamaCppRepo          = "https://github.com/elizaOS/llama.cpp.git"
)

// UnifiedLlamaCppRoot is the single runtime llama.cpp checkout.
//
// Order: LLAMA_CPP_ROOT → vendor/llama-cpp-<pin> (patched) → ../llama.cpp sibling.
// WHY vendor default: build_llama_server.sh and in-process ggml share Ollama patches;
// bare eliza sibling alone misses kv-ext / compat hooks.
func UnifiedLlamaCppRoot() string {
	if raw := strings.TrimSpace(os.Getenv("LLAMA_CPP_ROOT")); raw != "" {
		return filepath.Clean(raw)
	}
	if vendor := vendorLlamaCppRoot(); vendor != "" {
		return vendor
	}
	if repo := zerollamaRepoRoot(); repo != "" {
		return filepath.Clean(filepath.Join(repo, "..", "llama.cpp"))
	}
	return ""
}

func vendorLlamaCppRoot() string {
	repo := zerollamaRepoRoot()
	if repo == "" {
		return ""
	}
	fetchHead := readVendorLlamaCppPin()
	if fetchHead == "" {
		return ""
	}
	vendor := filepath.Join(repo, "vendor", "llama-cpp-"+fetchHead)
	if _, err := os.Stat(filepath.Join(vendor, "CMakeLists.txt")); err != nil {
		return ""
	}
	return vendor
}

func zerollamaRepoRoot() string {
	for _, root := range reporoots.SearchRoots() {
		if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
			continue
		}
		if _, err := os.Stat(filepath.Join(root, "LLAMA_CPP_COMMIT")); err == nil {
			return root
		}
	}
	return ""
}

// PinnedLlamaCppCommit reads repo-root LLAMA_CPP_COMMIT (unified runtime pin).
func PinnedLlamaCppCommit() string {
	repo := zerollamaRepoRoot()
	if repo == "" {
		return defaultUnifiedLlamaCppCommit
	}
	path := filepath.Join(repo, "LLAMA_CPP_COMMIT")
	data, err := os.ReadFile(path)
	if err != nil {
		return defaultUnifiedLlamaCppCommit
	}
	ref := strings.TrimSpace(string(data))
	if ref == "" {
		return defaultUnifiedLlamaCppCommit
	}
	return ref
}

func siblingLlamaCppRoot() string {
	repo := zerollamaRepoRoot()
	if repo == "" {
		return ""
	}
	return filepath.Clean(filepath.Join(repo, "..", "llama.cpp"))
}

func isBareSiblingLlamaRoot(path string) bool {
	if path == "" {
		return false
	}
	sibling := siblingLlamaCppRoot()
	return sibling != "" && filepath.Clean(path) == sibling
}

func llamaServerBinUnderRoot(bin, root string) bool {
	if bin == "" || root == "" {
		return false
	}
	rel, err := filepath.Rel(filepath.Clean(root), filepath.Clean(bin))
	return err == nil && !strings.HasPrefix(rel, "..")
}

func unifiedLlamaServerBinForRoot(root string) (string, bool) {
	if root == "" {
		return "", false
	}
	path := filepath.Join(root, "build", "bin", llamaCppBinaryName("llama-server", runtime.GOOS))
	if st, err := os.Stat(path); err == nil && !st.IsDir() {
		return path, true
	}
	return path, false
}

// UnifiedLlamaServerBin returns $LLAMA_CPP_ROOT/build/bin/llama-server when present.
func UnifiedLlamaServerBin() string {
	root := UnifiedLlamaCppRoot()
	if root == "" {
		return ""
	}
	path, _ := unifiedLlamaServerBinForRoot(root)
	return path
}

func unifiedLlamaServerBinExists() (string, bool) {
	if vendor := vendorLlamaCppRoot(); vendor != "" {
		if path, ok := unifiedLlamaServerBinForRoot(vendor); ok {
			return path, true
		}
	}
	return unifiedLlamaServerBinForRoot(UnifiedLlamaCppRoot())
}

// IsLegacyLlamaCppCheckout reports deprecated sibling names (eliza-llama.cpp, etc.).
func IsLegacyLlamaCppCheckout(path string) bool {
	if path == "" {
		return false
	}
	switch filepath.Base(filepath.Clean(path)) {
	case "eliza-llama.cpp", "eliza_llama.cpp", "stock-llama.cpp", "ollama-upstream":
		return true
	default:
		return false
	}
}

// LlamaCppUnificationStatus summarizes pin + binary layout for doctor / status.
type LlamaCppUnificationStatus struct {
	UnifiedRoot       string
	PinnedCommit      string
	SiblingHEAD       string
	EffectiveServer   string
	ServerFromEnv     bool
	ServerUnderRoot   bool
	LegacyCheckout    bool
	VendorPin         string
	RuntimeReady      bool
	Detail            string
	Warn              bool
	FixHint           string
}

func LlamaCppUnificationReport() LlamaCppUnificationStatus {
	root := UnifiedLlamaCppRoot()
	pin := PinnedLlamaCppCommit()
	report := LlamaCppUnificationStatus{
		UnifiedRoot:  root,
		PinnedCommit: pin,
		VendorPin:    readVendorLlamaCppPin(),
	}

	if root != "" {
		report.LegacyCheckout = IsLegacyLlamaCppCheckout(root)
		report.SiblingHEAD = gitRevParseHEAD(root)
		if path, ok := unifiedLlamaServerBinExists(); ok {
			report.RuntimeReady = true
			report.EffectiveServer = path
		}
	}

	if override := strings.TrimSpace(os.Getenv("LLAMA_SERVER_BIN")); override != "" {
		report.ServerFromEnv = true
		report.EffectiveServer = filepath.Clean(override)
	}

	if report.EffectiveServer != "" && root != "" {
		rel, err := filepath.Rel(root, report.EffectiveServer)
		report.ServerUnderRoot = err == nil && !strings.HasPrefix(rel, "..")
	}

	var parts []string
	parts = append(parts, "unified "+root+" @ pin "+shortRef(pin))
	if report.SiblingHEAD != "" {
		parts = append(parts, "HEAD "+shortRef(report.SiblingHEAD))
	}
	if report.EffectiveServer != "" {
		parts = append(parts, "llama-server "+report.EffectiveServer)
	}
	if report.VendorPin != "" && !pinRefsMatch(report.VendorPin, pin) {
		parts = append(parts, "in-process vendor "+shortRef(report.VendorPin)+" (rebase pending)")
		report.Warn = true
	}
	if report.LegacyCheckout {
		parts = append(parts, "legacy checkout name — migrate to vendor tree")
		report.Warn = true
		report.FixHint = "./scripts/rebase_vendor_unified.sh --sync && ./scripts/build_llama_server.sh"
	}
	if vendor := vendorLlamaCppRoot(); vendor != "" && isBareSiblingLlamaRoot(root) {
		parts = append(parts, "LLAMA_CPP_ROOT bare sibling — prefer "+vendor)
		report.Warn = true
		if report.FixHint == "" {
			report.FixHint = "unset LLAMA_CPP_ROOT or export LLAMA_CPP_ROOT=" + vendor
		}
	}
	if report.ServerFromEnv && report.EffectiveServer != "" {
		if vendor := vendorLlamaCppRoot(); vendor != "" && !llamaServerBinUnderRoot(report.EffectiveServer, vendor) {
			if vPath, ok := unifiedLlamaServerBinForRoot(vendor); ok {
				parts = append(parts, "patched vendor llama-server available at "+vPath)
				report.Warn = true
				if report.FixHint == "" {
					report.FixHint = "unset LLAMA_SERVER_BIN (zerollama prefers vendor build)"
				}
			}
		}
	}
	if report.ServerFromEnv && !report.ServerUnderRoot && report.EffectiveServer != "" && !report.Warn {
		parts = append(parts, "LLAMA_SERVER_BIN outside unified tree")
		report.Warn = true
		if report.FixHint == "" {
			report.FixHint = "unset LLAMA_SERVER_BIN or point at " + filepath.Join(root, "build", "bin", "llama-server")
		}
	}
	if !report.RuntimeReady && root != "" {
		parts = append(parts, "llama-server not built")
		report.Warn = true
		if report.FixHint == "" {
			report.FixHint = "./scripts/build_llama_server.sh"
		}
	}
	report.Detail = strings.Join(parts, "; ")
	return report
}

func readVendorLlamaCppPin() string {
	repo := zerollamaRepoRoot()
	if repo == "" {
		return ""
	}
	if data, err := os.ReadFile(filepath.Join(repo, "LLAMA_CPP_VERSION")); err == nil {
		if v := strings.TrimSpace(string(data)); v != "" {
			return v
		}
	}
	f, err := os.Open(filepath.Join(repo, "Makefile.sync"))
	if err != nil {
		return ""
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if strings.HasPrefix(line, "FETCH_HEAD=") {
			return strings.TrimPrefix(line, "FETCH_HEAD=")
		}
	}
	return ""
}

func pinRefsMatch(shortPin, fullPin string) bool {
	shortPin = strings.TrimSpace(shortPin)
	fullPin = strings.TrimSpace(fullPin)
	if shortPin == "" || fullPin == "" {
		return false
	}
	if shortPin == fullPin {
		return true
	}
	return strings.HasPrefix(fullPin, shortPin)
}

func LlamaCppPathUsesLegacyCheckout(path string) bool {
	if path == "" {
		return false
	}
	clean := filepath.Clean(path)
	for _, part := range strings.Split(clean, string(os.PathSeparator)) {
		if IsLegacyLlamaCppCheckout(part) {
			return true
		}
	}
	return false
}

// ApplyUnifiedLlamaCppEnv redirects legacy LLAMA_CPP_ROOT / LLAMA_SERVER_BIN to the
// patched vendor tree when available.
//
// WHY: operators often keep shell exports pointing at ../eliza-llama.cpp or bare
// ../llama.cpp after U3 — MTP/eagle3 then miss Ollama patches and argv probes fail.
func ApplyUnifiedLlamaCppEnv() []string {
	var msgs []string
	vendor := vendorLlamaCppRoot()
	unified := UnifiedLlamaCppRoot()

	if vendor != "" {
		root := strings.TrimSpace(os.Getenv("LLAMA_CPP_ROOT"))
		switch {
		case root == "":
			_ = os.Setenv("LLAMA_CPP_ROOT", vendor)
			unified = vendor
		case LlamaCppPathUsesLegacyCheckout(root) || isBareSiblingLlamaRoot(root):
			_ = os.Setenv("LLAMA_CPP_ROOT", vendor)
			msgs = append(msgs, "set LLAMA_CPP_ROOT="+vendor+" (was "+root+")")
			unified = vendor
		}
	}

	serverPath, serverOK := unifiedLlamaServerBinForRoot(unified)
	if !serverOK && vendor != "" && unified != vendor {
		if path, ok := unifiedLlamaServerBinForRoot(vendor); ok {
			serverPath, serverOK = path, true
		}
	}

	if root := strings.TrimSpace(os.Getenv("LLAMA_CPP_ROOT")); root != "" && LlamaCppPathUsesLegacyCheckout(root) && vendor == "" {
		msgs = append(msgs, "LLAMA_CPP_ROOT uses legacy checkout "+root)
		if unified != "" {
			if _, err := os.Stat(filepath.Join(unified, "CMakeLists.txt")); err == nil {
				_ = os.Setenv("LLAMA_CPP_ROOT", unified)
				msgs = append(msgs, "set LLAMA_CPP_ROOT="+unified)
			}
		}
	}

	if bin := strings.TrimSpace(os.Getenv("LLAMA_SERVER_BIN")); bin != "" {
		redirect := LlamaCppPathUsesLegacyCheckout(bin)
		if !redirect && vendor != "" && serverOK && !llamaServerBinUnderRoot(bin, vendor) {
			redirect = isBareSiblingLlamaRoot(rootPathForLlamaServerBin(bin)) || LlamaCppPathUsesLegacyCheckout(bin)
		}
		if redirect && serverOK {
			msgs = append(msgs, "LLAMA_SERVER_BIN was "+bin)
			_ = os.Setenv("LLAMA_SERVER_BIN", serverPath)
			msgs = append(msgs, "set LLAMA_SERVER_BIN="+serverPath)
		} else if LlamaCppPathUsesLegacyCheckout(bin) && !serverOK && unified != "" {
			msgs = append(msgs, "run ./scripts/build_llama_server.sh in "+unified)
		}
	}

	if strings.TrimSpace(os.Getenv("LLAMA_CPP_ROOT")) == "" && unified != "" {
		if _, err := os.Stat(filepath.Join(unified, "CMakeLists.txt")); err == nil {
			_ = os.Setenv("LLAMA_CPP_ROOT", unified)
		}
	}
	if strings.TrimSpace(os.Getenv("LLAMA_CPP_LIB")) == "" && unified != "" {
		libName := "libllama.so"
		if runtime.GOOS == "darwin" {
			libName = "libllama.dylib"
		} else if runtime.GOOS == "windows" {
			libName = "llama.dll"
		}
		lib := filepath.Join(unified, "build", "bin", libName)
		if st, err := os.Stat(lib); err == nil && !st.IsDir() {
			_ = os.Setenv("LLAMA_CPP_LIB", lib)
		}
	}
	if strings.TrimSpace(os.Getenv("LLAMA_SERVER_BIN")) == "" && serverOK {
		_ = os.Setenv("LLAMA_SERVER_BIN", serverPath)
	}

	return msgs
}

func rootPathForLlamaServerBin(bin string) string {
	// .../build/bin/llama-server → checkout root
	return filepath.Clean(filepath.Join(filepath.Dir(bin), "..", ".."))
}

func gitRevParseHEAD(root string) string {
	cmd := exec.Command("git", "-C", root, "rev-parse", "HEAD")
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func shortRef(ref string) string {
	ref = strings.TrimSpace(ref)
	if len(ref) > 12 {
		return ref[:12]
	}
	return ref
}

// UnifiedLlamaCppRepo is the canonical clone URL for ensure_llama_cpp_sibling.sh parity.
func UnifiedLlamaCppRepo() string {
	if raw := strings.TrimSpace(os.Getenv("LLAMA_CPP_REPO")); raw != "" {
		return raw
	}
	return unifiedLlamaCppRepo
}
