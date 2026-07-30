package cmd

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/internal/blobaudit"
	"github.com/ollama/ollama/internal/modelhealth"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/version"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/trainingworker"
)

type doctorCheck struct {
	Name    string `json:"name"`
	Status  string `json:"status"` // ok, warn, fail
	Detail  string `json:"detail,omitempty"`
	FixHint string `json:"fix_hint,omitempty"`
}

type doctorReport struct {
	Checks   []doctorCheck `json:"checks"`
	Failures int           `json:"failures"`
	Warnings int           `json:"warnings"`
	OK       bool          `json:"ok"`
}

func NewDoctorCommand() *cobra.Command {
	var jsonOut bool
	var fix bool
	var modelsOnly bool
	var auditStorage bool
	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "Check local zerollama / Apple Silicon runtime readiness",
		Long:  "Validate uv venv, Metal libllama, sidecar health, and autoconfig on Darwin.",
		RunE: func(_ *cobra.Command, _ []string) error {
			if auditStorage {
				report, err := blobaudit.Audit()
				if err != nil {
					return err
				}
				if jsonOut {
					enc := json.NewEncoder(os.Stdout)
					enc.SetIndent("", "  ")
					return enc.Encode(report)
				}
				fmt.Fprint(os.Stdout, blobaudit.FormatHuman(report))
				return nil
			}
			repo := doctorRepoRoot()
			if fix {
				if err := runDoctorFix(repo); err != nil {
					return err
				}
				if modelsOnly {
					if err := runDoctorModelsFix(); err != nil {
						return err
					}
				}
			}
			var report doctorReport
			if modelsOnly {
				report = buildDoctorModelsReport()
			} else {
				report = buildDoctorReport(repo)
				if !fix {
					report = mergeDoctorReports(report, buildDoctorModelsReport())
				}
			}
			if jsonOut {
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				if err := enc.Encode(report); err != nil {
					return err
				}
				if !report.OK {
					return fmt.Errorf("doctor: %d check(s) failed, %d warning(s)", report.Failures, report.Warnings)
				}
				return nil
			}
			return printDoctorHuman(report)
		},
	}
	cmd.Flags().BoolVar(&jsonOut, "json", false, "Print results as JSON")
	cmd.Flags().BoolVar(&fix, "fix", false, "Run safe auto-fixes (uv venv; on Darwin build Metal llama.cpp when missing)")
	cmd.Flags().BoolVar(&modelsOnly, "models", false, "Check local model blob integrity (missing/orphaned registrations)")
	cmd.Flags().BoolVar(&auditStorage, "audit", false, "Blob storage rollup (use with --models or alone; see also zerollama blobs audit)")
	return cmd
}

func buildDoctorReport(repo string) doctorReport {
	checks := runDoctorChecks(repo)
	report := doctorReport{Checks: checks}
	for _, c := range checks {
		switch c.Status {
		case "fail":
			report.Failures++
		case "warn":
			report.Warnings++
		}
	}
	report.OK = report.Failures == 0
	return report
}

func mergeDoctorReports(base, extra doctorReport) doctorReport {
	base.Checks = append(base.Checks, extra.Checks...)
	base.Failures += extra.Failures
	base.Warnings += extra.Warnings
	base.OK = base.Failures == 0
	return base
}

func buildDoctorModelsReport() doctorReport {
	reports, err := modelhealth.CheckAll()
	if err != nil {
		return doctorReport{
			Checks: []doctorCheck{{
				Name:   "model health",
				Status: "fail",
				Detail: err.Error(),
			}},
			Failures: 1,
		}
	}

	var checks []doctorCheck
	for _, r := range reports {
		status := "ok"
		switch r.Status {
		case modelhealth.StatusRepairable:
			status = "warn"
		case modelhealth.StatusOrphaned, modelhealth.StatusBroken:
			status = "fail"
		}
		checks = append(checks, doctorCheck{
			Name:    "model " + r.Name,
			Status:  status,
			Detail:  modelhealth.FormatSummary(r),
			FixHint: r.FixHint,
		})
	}

	report := doctorReport{Checks: checks}
	for _, c := range checks {
		switch c.Status {
		case "fail":
			report.Failures++
		case "warn":
			report.Warnings++
		}
	}
	report.OK = report.Failures == 0
	return report
}

func runDoctorModelsFix() error {
	reports, err := modelhealth.CheckAll()
	if err != nil {
		return err
	}
	var removed int
	for _, r := range reports {
		if r.Status != modelhealth.StatusOrphaned {
			continue
		}
		fmt.Printf("== removing orphaned model %s ==\n", r.Name)
		if err := modelhealth.RemoveManifest(r.Name); err != nil {
			return fmt.Errorf("remove %s: %w", r.Name, err)
		}
		removed++
	}
	if removed > 0 {
		fmt.Printf("removed %d orphaned model registration(s)\n", removed)
	}
	return nil
}

func printDoctorHuman(report doctorReport) error {
	for _, c := range report.Checks {
		fmt.Printf("[%s] %s\n", c.Status, c.Name)
		if c.Detail != "" {
			fmt.Printf("      %s\n", c.Detail)
		}
		if c.FixHint != "" && c.Status != "ok" {
			fmt.Printf("      → %s\n", c.FixHint)
		}
	}
	fmt.Println()
	if report.Failures > 0 {
		return fmt.Errorf("doctor: %d check(s) failed, %d warning(s)", report.Failures, report.Warnings)
	}
	if report.Warnings > 0 {
		fmt.Printf("doctor: ok with %d warning(s)\n", report.Warnings)
		return nil
	}
	fmt.Println("doctor: all checks passed")
	return nil
}

func runDoctorFix(repo string) error {
	fmt.Println("== doctor --fix: runtime venv ==")
	script := filepath.Join(repo, "scripts", "runtime", "runtime_uv_venv.sh")
	if _, err := os.Stat(script); err != nil {
		return fmt.Errorf("missing %s", script)
	}
	cmd := exec.Command("bash", script)
	cmd.Dir = repo
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("runtime_uv_venv: %w", err)
	}
	if runtime.GOOS == "darwin" {
		bin := filepath.Join(repo, "zerollama")
		if _, err := os.Stat(bin); err != nil {
			build := filepath.Join(repo, "scripts", "build", "build_zerollama_mac.sh")
			if _, err := os.Stat(build); err == nil {
				fmt.Println("== doctor --fix: build zerollama (mac CGO) ==")
				bcmd := exec.Command("bash", build)
				bcmd.Dir = repo
				bcmd.Stdout = os.Stdout
				bcmd.Stderr = os.Stderr
				if err := bcmd.Run(); err != nil {
					return fmt.Errorf("build_zerollama_mac: %w", err)
				}
			}
		}
	}
	if runtime.GOOS == "darwin" && os.Getenv("DOCTOR_FIX_BUILD") != "0" {
		specOK, filesOK, _ := doctorANESourceMarkers(doctorLlamaCppRootForANE(repo))
		if !specOK || !filesOK {
			if err := doctorSyncANEDraftHook(repo); err != nil {
				return err
			}
		}
		if doctorFindLibLlama(repo) == "" {
			if err := doctorEnsureLlamaCppSibling(repo); err != nil {
				return err
			}
			build := filepath.Join(repo, "scripts", "build", "build_llama_server.sh")
			if _, err := os.Stat(build); err != nil {
				// Pre-reorg flat path still present in some checkouts.
				build = filepath.Join(repo, "scripts", "build_llama_server.sh")
			}
			if _, err := os.Stat(build); err == nil {
				fmt.Println("== doctor --fix: build Metal llama.cpp ==")
				bcmd := exec.Command("bash", build)
				bcmd.Dir = repo
				bcmd.Stdout = os.Stdout
				bcmd.Stderr = os.Stderr
				if err := bcmd.Run(); err != nil {
					return fmt.Errorf("build_llama_server: %w", err)
				}
			}
		}
	}
	return nil
}

// doctorEnsureLlamaCppSibling clones ../llama.cpp when missing (M14).
// Why before build_llama_server: that script assumes LLAMA_CPP_ROOT exists; failing at
// CMake time was opaque on fresh checkouts. mac_setup already runs ensure_llama_cpp_sibling
// first — doctor --fix should match so tier-0 bootstrap is one self-service command.
func doctorEnsureLlamaCppSibling(repo string) error {
	script := filepath.Join(repo, "scripts", "vendor", "ensure_llama_cpp_sibling.sh")
	if _, err := os.Stat(script); err != nil {
		return fmt.Errorf("missing %s", script)
	}
	fmt.Println("== doctor --fix: ensure llama.cpp sibling ==")
	cmd := exec.Command("bash", script)
	cmd.Dir = repo
	cmd.Env = append(os.Environ(), "ZEROLLAMA_REPO="+repo)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ensure_llama_cpp_sibling: %w", err)
	}
	return nil
}

func runDoctorChecks(repo string) []doctorCheck {
	var out []doctorCheck

	out = append(out, doctorCheckGo())
	out = append(out, doctorCheckEdgeBuild())
	out = append(out, doctorCheckGgmlRunner())
	out = append(out, doctorCheckLlamaCppUnified(repo))
	out = append(out, doctorCheckLlamaPatches(repo))
	out = append(out, doctorCheckLlamaServer(repo))
	if runtime.GOOS == "darwin" {
		out = append(out, doctorCheckMacCGO(repo))
	}
	out = append(out, doctorCheckZerollamaBinary(repo))
	out = append(out, doctorCheckUV())
	out = append(out, doctorCheckRuntimeVenv(repo))
	out = append(out, doctorCheckLibLlama(repo))

	if runtime.GOOS == "darwin" {
		out = append(out, doctorCheckMLX(repo))
		out = append(out, doctorCheckDarwinSidecarBootstrap())
		out = append(out, doctorCheckServeModes())
		out = append(out, doctorCheckTrainingVenv(repo))
		out = append(out, doctorCheckSidecarHealth())
		out = append(out, doctorCheckTextGGUF(repo))
		out = append(out, doctorCheckANE(repo))
		out = append(out, doctorCheckANEDraftHook(repo))
		out = append(out, doctorCheckFlashMoE(repo))
	} else {
		out = append(out, doctorCheck{
			Name:   "darwin runtime smoke",
			Status: "warn",
			Detail: "full sidecar checks run on darwin only",
		})
	}
	return out
}

func doctorRepoRoot() string {
	for _, key := range []string{"ZEROLLAMA_REPO", "OLLAMA_TRAINING_PYTHONPATH"} {
		if p := strings.TrimSpace(os.Getenv(key)); p != "" {
			return filepath.Clean(p)
		}
	}
	if root, err := trainingworker.RepoRoot(); err == nil && root != "" {
		return root
	}
	if wd, err := os.Getwd(); err == nil {
		for dir := wd; ; dir = filepath.Dir(dir) {
			if st, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil && !st.IsDir() {
				return dir
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
		}
	}
	return "."
}

func doctorCheckGo() doctorCheck {
	return doctorCheck{
		Name:   "platform",
		Status: "ok",
		Detail: fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
	}
}

func doctorCheckZerollamaBinary(repo string) doctorCheck {
	bin := filepath.Join(repo, "zerollama")
	if _, err := os.Stat(bin); err != nil {
		if p, err := exec.LookPath("zerollama"); err == nil {
			bin = p
		} else {
			return doctorCheck{
				Name:    "zerollama binary",
				Status:  "warn",
				Detail:  "not found in repo or PATH",
				FixHint: "./scripts/build/build_zerollama_mac.sh",
			}
		}
	}
	cmd := exec.Command(bin, "serve", "--help")
	if err := cmd.Run(); err != nil {
		return doctorCheck{
			Name:    "zerollama binary",
			Status:  "fail",
			Detail:  fmt.Sprintf("%s serve --help failed: %v", bin, err),
			FixHint: "./scripts/build/build_zerollama_mac.sh",
		}
	}
	// Stale binary may lack doctor subcommand.
	docCmd := exec.Command(bin, "doctor", "--help")
	if err := docCmd.Run(); err != nil {
		return doctorCheck{
			Name:    "zerollama binary",
			Status:  "warn",
			Detail:  fmt.Sprintf("%s lacks doctor subcommand (stale build?)", bin),
			FixHint: "./scripts/build/build_zerollama_mac.sh",
		}
	}
	return doctorCheck{
		Name:   "zerollama binary",
		Status: "ok",
		Detail: bin,
	}
}

func doctorCheckMacCGO(repo string) doctorCheck {
	script := filepath.Join(repo, "scripts", "mac_cgo_env.sh")
	if _, err := os.Stat(script); err != nil {
		return doctorCheck{
			Name:    "mac cgo build",
			Status:  "warn",
			Detail:  "scripts/runtime/mac_cgo_env.sh missing",
			FixHint: "pull latest repo; ./scripts/build/build_zerollama_mac.sh",
		}
	}
	cmd := exec.Command("bash", script, "--check")
	out, err := cmd.CombinedOutput()
	text := strings.TrimSpace(string(out))
	if err != nil {
		status := "fail"
		if strings.Contains(text, "warn:") {
			status = "warn"
		}
		return doctorCheck{
			Name:    "mac cgo build",
			Status:  status,
			Detail:  text,
			FixHint: "xcode-select --install; ./scripts/build/build_zerollama_mac.sh (see docs/mac-dev-setup.md)",
		}
	}
	detail := text
	if idx := strings.Index(text, "ok CC="); idx >= 0 {
		detail = strings.TrimSpace(text[idx:])
	}
	return doctorCheck{
		Name:   "mac cgo build",
		Status: "ok",
		Detail: detail,
	}
}

func doctorCheckUV() doctorCheck {
	if _, err := exec.LookPath("uv"); err != nil {
		return doctorCheck{
			Name:    "uv",
			Status:  "fail",
			Detail:  "not on PATH",
			FixHint: "install from https://docs.astral.sh/uv/",
		}
	}
	out, _ := exec.Command("uv", "--version").CombinedOutput()
	return doctorCheck{
		Name:   "uv",
		Status: "ok",
		Detail: strings.TrimSpace(string(out)),
	}
}

func doctorCheckRuntimeVenv(repo string) doctorCheck {
	py := filepath.Join(repo, "runtime", ".venv", "bin", "python")
	if _, err := os.Stat(py); err != nil {
		return doctorCheck{
			Name:    "runtime/.venv",
			Status:  "fail",
			Detail:  "missing",
			FixHint: "./scripts/runtime/runtime_uv_venv.sh or zerollama doctor --fix",
		}
	}
	cmd := exec.Command(py, "-c", "import fastapi")
	if err := cmd.Run(); err != nil {
		return doctorCheck{
			Name:    "runtime/.venv",
			Status:  "fail",
			Detail:  "fastapi import failed",
			FixHint: "RUNTIME_UV_SYNC=1 ./scripts/runtime/runtime_uv_venv.sh",
		}
	}
	return doctorCheck{
		Name:   "runtime/.venv",
		Status: "ok",
		Detail: py,
	}
}

func doctorFindLibLlama(repo string) string {
	candidates := doctorLibLlamaCandidates(repo)
	for _, p := range candidates {
		if st, err := os.Stat(p); err == nil && st.Size() > 0 {
			return p
		}
	}
	return ""
}

func doctorLibLlamaCandidates(repo string) []string {
	candidates := []string{}
	if p := strings.TrimSpace(os.Getenv("LLAMA_CPP_LIB")); p != "" {
		candidates = append(candidates, p)
	}
	root := strings.TrimSpace(os.Getenv("LLAMA_CPP_ROOT"))
	if root == "" {
		root = filepath.Clean(filepath.Join(repo, "..", "llama.cpp"))
	}
	if runtime.GOOS == "darwin" {
		candidates = append(candidates, filepath.Join(root, "build", "bin", "libllama.dylib"))
	} else {
		candidates = append(candidates, filepath.Join(root, "build", "bin", "libllama.so"))
	}
	return candidates
}

func doctorCheckLibLlama(repo string) doctorCheck {
	if p := doctorFindLibLlama(repo); p != "" {
		return doctorCheck{
			Name:   "libllama",
			Status: "ok",
			Detail: p,
		}
	}
	return doctorCheck{
		Name:    "libllama",
		Status:  "fail",
		Detail:  "Metal/CUDA libllama not found",
		FixHint: "zerollama doctor --fix (clones ../llama.cpp then builds) or ./scripts/vendor/ensure_llama_cpp_sibling.sh",
	}
}

func doctorCheckMLX(repo string) doctorCheck {
	const name = "mlx engine"
	if err := mlx.CheckInit(); err != nil {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  err.Error(),
			FixHint: doctorMLXFixHint(repo),
		}
	}
	path, err := mlx.LoadedLibraryPath()
	if err != nil || path == "" {
		return doctorCheck{
			Name:    name,
			Status:  "warn",
			Detail:  "MLX optional — no libmlxc loaded (safetensors/MLX models unavailable)",
			FixHint: doctorMLXFixHint(repo),
		}
	}
	status := "ok"
	detail := path
	var fix string
	if doctorIsStaleFlatMLXPath(repo, path) {
		status = "warn"
		detail = "loaded stale flat copy: " + path
		fix = doctorMLXFixHint(repo)
	} else if doctorHasProductionMLX(repo) && strings.Contains(path, filepath.Join("build", "lib", "ollama")) {
		status = "warn"
		detail = "dev tree MLX loaded; production layout available at dist/darwin-arm64/"
		fix = "cd dist/darwin-arm64 && ./zerollama serve for release MLX layout"
	}
	return doctorCheck{
		Name:    name,
		Status:  status,
		Detail:  detail,
		FixHint: fix,
	}
}

func doctorIsStaleFlatMLXPath(repo, loaded string) bool {
	flatDir := filepath.Join(repo, "build", "lib", "ollama")
	rel, err := filepath.Rel(flatDir, loaded)
	if err != nil || strings.HasPrefix(rel, "..") {
		return false
	}
	if strings.Contains(rel, string(filepath.Separator)) {
		return false
	}
	if !strings.HasPrefix(filepath.Base(loaded), "libmlxc") {
		return false
	}
	return doctorHasDevMLXVariant(repo) || doctorHasProductionMLX(repo)
}

func doctorHasDevMLXVariant(repo string) bool {
	matches, err := filepath.Glob(filepath.Join(repo, "build", "*", "lib", "ollama", "libmlxc.*"))
	return err == nil && len(matches) > 0
}

func doctorHasProductionMLX(repo string) bool {
	p := filepath.Join(repo, "dist", "darwin-arm64", "lib", "ollama", "mlx_metal_v3", "libmlxc.dylib")
	st, err := os.Stat(p)
	return err == nil && st.Size() > 0
}

func doctorMLXFixHint(repo string) string {
	var hints []string
	stale := filepath.Join(repo, "build", "lib", "ollama", "libmlxc.dylib")
	if _, err := os.Stat(stale); err == nil {
		hints = append(hints, "rm "+stale+" if stale (CHECK failed: mlx_distributed_group_new_)")
	}
	hints = append(hints, "BUILD_MLX=1 ./scripts/build/build_zerollama_mac.sh (dev) or ./scripts/build/build_production_mac.sh (dist/)")
	return strings.Join(hints, "; ")
}

func doctorOllamaHost() string {
	u := envconfig.ConnectableHost()
	return strings.TrimSuffix(u.String(), "/")
}

func doctorRuntimeURL() string {
	if u := strings.TrimSpace(envconfig.RuntimeURL()); u != "" {
		return strings.TrimSuffix(u, "/")
	}
	port := envconfig.RuntimeEmbedPort()
	return fmt.Sprintf("http://127.0.0.1:%d", port)
}

func doctorAddUniqueURL(urls []string, seen map[string]bool, raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return urls
	}
	if !strings.Contains(raw, "://") {
		raw = "http://" + raw
	}
	raw = strings.TrimSuffix(raw, "/")
	if seen[raw] {
		return urls
	}
	seen[raw] = true
	return append(urls, raw)
}

func doctorGoAPIProbeURLs() []string {
	seen := map[string]bool{}
	var urls []string
	urls = doctorAddUniqueURL(urls, seen, doctorOllamaHost())
	for _, hostport := range []string{
		"127.0.0.1:11434",
		"127.0.0.1:8080",
		"localhost:11434",
		"localhost:8080",
		"[::1]:11434",
	} {
		urls = doctorAddUniqueURL(urls, seen, "http://"+hostport)
	}
	return urls
}

func doctorRuntimeProbeURLs() []string {
	seen := map[string]bool{}
	var urls []string
	urls = doctorAddUniqueURL(urls, seen, doctorRuntimeURL())
	port := envconfig.RuntimeEmbedPort()
	for _, host := range []string{"127.0.0.1", "localhost", "[::1]"} {
		urls = doctorAddUniqueURL(urls, seen, fmt.Sprintf("http://%s:%d", host, port))
	}
	return urls
}

func doctorTCPReachable(hostport string) bool {
	conn, err := net.DialTimeout("tcp", hostport, 800*time.Millisecond)
	if err != nil {
		return false
	}
	_ = conn.Close()
	return true
}

func doctorHTTPReachable(url string) bool {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode >= 200 && resp.StatusCode < 500
}

func doctorProbeGoAPI() (baseURL string, tcpOnly []string) {
	for _, base := range doctorGoAPIProbeURLs() {
		if doctorHTTPReachable(base + "/api/tags") {
			return base, nil
		}
	}
	for _, base := range doctorGoAPIProbeURLs() {
		hostport := strings.TrimPrefix(strings.TrimPrefix(base, "http://"), "https://")
		if doctorTCPReachable(hostport) {
			tcpOnly = append(tcpOnly, base)
		}
	}
	return "", tcpOnly
}

func doctorProbeRuntimeSidecar() (baseURL string, tcpOnly []string) {
	for _, base := range doctorRuntimeProbeURLs() {
		if doctorHTTPReachable(base + "/health") {
			return base, nil
		}
	}
	for _, base := range doctorRuntimeProbeURLs() {
		hostport := strings.TrimPrefix(strings.TrimPrefix(base, "http://"), "https://")
		if doctorTCPReachable(hostport) {
			tcpOnly = append(tcpOnly, base)
		}
	}
	return "", tcpOnly
}

func doctorCheckEdgeBuild() doctorCheck {
	if version.IsEdgeBuild() {
		detail := "edge-marked binary — serve applies --edge defaults unless ZEROLLAMA_EDGE=0"
		if !envconfig.GgmlRunnerLinked() {
			detail += "; ggml subprocess unlinked (v1)"
		}
		return doctorCheck{
			Name:   "edge build",
			Status: "ok",
			Detail: detail,
		}
	}
	detail := "full daemon"
	if envconfig.EdgeMode() {
		detail += " (ZEROLLAMA_EDGE=1)"
	} else if envconfig.LlamaServerBackendAuto() {
		detail += " (Linux llama-server auto)"
	} else if envconfig.LlamaServerBackendExplicit() {
		detail += " (llama-server explicit)"
	}
	return doctorCheck{
		Name:   "edge build",
		Status: "ok",
		Detail: detail + "; use zerollama serve --edge for upstream-shaped GGUF",
	}
}

func doctorCheckGgmlRunner() doctorCheck {
	if envconfig.GgmlRunnerLinked() {
		return doctorCheck{
			Name:   "ggml runner",
			Status: "ok",
			Detail: "linked (default build)",
		}
	}
	return doctorCheck{
		Name:   "ggml runner",
		Status: "ok",
		Detail: "not linked (edge build); GGUF requires llama-server",
	}
}

func doctorCheckLlamaCppUnified(repo string) doctorCheck {
	_ = repo
	report := llm.LlamaCppUnificationReport()
	status := "ok"
	if report.Warn {
		status = "warn"
	}
	if report.UnifiedRoot == "" {
		status = "warn"
		report.Detail = "unified llama.cpp not resolved; run ./scripts/vendor/rebase_vendor_unified.sh --sync or ./scripts/vendor/ensure_llama_cpp_sibling.sh"
		report.FixHint = "./scripts/build/build_llama_server.sh"
	}
	return doctorCheck{
		Name:    "llama.cpp unified",
		Status:  status,
		Detail:  report.Detail,
		FixHint: report.FixHint,
	}
}

func doctorCheckLlamaServer(repo string) doctorCheck {
	_ = repo
	path, err := llm.FindLlamaServer()
	if err == nil {
		detail := path
		switch {
		case envconfig.LlamaServerBackendAuto():
			detail += " (Linux auto routes GGUF here)"
		case envconfig.EdgeMode() || envconfig.LlamaServerBackendExplicit():
			detail += " (required for current backend policy)"
		default:
			detail += " (optional; Mac default uses ggml Metal; spec tags auto-route when discoverable)"
		}
		return doctorCheck{
			Name:   "llama-server",
			Status: "ok",
			Detail: detail,
		}
	}

	status := "ok"
	fix := ""
	detail := err.Error()
	switch {
	case envconfig.EdgeMode() || envconfig.LlamaServerBackendExplicit():
		status = "fail"
		fix = "./scripts/build/build_llama_server.sh (Linux CUDA) or ./scripts/build/build_ollama_llama_server_darwin.sh; or set LLAMA_SERVER_BIN"
	case runtime.GOOS == "linux" && !envconfig.LlamaServerBackendDisabled():
		status = "warn"
		fix = "./scripts/build/build_llama_server.sh — plain Linux serve sets ZEROLLAMA_LLAMA_SERVER=auto when discoverable"
	default:
		detail = "not discovered (Mac plain GGUF uses ggml; build llama-server for DFlash/MTP/n-gram auto-route)"
	}
	return doctorCheck{
		Name:    "llama-server",
		Status:  status,
		Detail:  detail,
		FixHint: fix,
	}
}

func doctorCheckServeModes() doctorCheck {
	goBase, goTCP := doctorProbeGoAPI()
	sidecarBase, sidecarTCP := doctorProbeRuntimeSidecar()

	var parts []string
	if goBase != "" {
		parts = append(parts, "Go API "+goBase)
	} else if len(goTCP) > 0 {
		parts = append(parts, "Go API "+goTCP[0]+" (TCP up, /api/tags not ready)")
	}
	if sidecarBase != "" {
		parts = append(parts, "runtime sidecar "+sidecarBase)
	} else if len(sidecarTCP) > 0 {
		parts = append(parts, "runtime sidecar "+sidecarTCP[0]+" (TCP up, /health not ready)")
	}
	detail := "none detected"
	if len(parts) > 0 {
		detail = strings.Join(parts, "; ")
	}

	status := "ok"
	var fixes []string
	switch {
	case goBase == "" && sidecarBase == "" && len(goTCP) == 0 && len(sidecarTCP) == 0:
		status = "warn"
		fixes = append(fixes, "run zerollama serve (Mac auto-starts sidecar on :8081)")
	case goBase != "" && sidecarBase != "" && strings.Contains(goBase, ":11434"):
		detail += " (Mac default: Go :11434 + sidecar :8081)"
	case goBase != "" && strings.Contains(goBase, ":8080") && sidecarBase != "":
		detail += " (CI/smoke layout: Go :8080 + sidecar :8081)"
	case (goBase != "" || len(goTCP) > 0) && sidecarBase == "" && len(sidecarTCP) == 0:
		if envconfig.EdgeMode() || !envconfig.RuntimeDefaultOn() {
			detail += " (runtime chat off — expected for edge or ZEROLLAMA_RUNTIME=0)"
		} else if runtime.GOOS == "darwin" {
			status = "warn"
			fixes = append(fixes, "runtime sidecar missing — check serve logs for darwin sidecar bootstrap; sidecar log: "+doctorSidecarLogHint())
		} else {
			status = "warn"
			fixes = append(fixes, "start runtime sidecar or set ZEROLLAMA_RUNTIME_URL")
		}
	case sidecarBase != "" && goBase == "" && len(goTCP) == 0:
		status = "warn"
		fixes = append(fixes, "sidecar without Go proxy — run zerollama serve")
	case len(goTCP) > 0 && goBase == "":
		status = "warn"
		fixes = append(fixes, "Go port open but HTTP not ready — wait for 'Listening on' in serve logs")
	case len(sidecarTCP) > 0 && sidecarBase == "":
		status = "warn"
		fixes = append(fixes, "sidecar port open but /health not ready — see "+doctorSidecarLogHint())
	}
	return doctorCheck{
		Name:    "serve mode",
		Status:  status,
		Detail:  detail,
		FixHint: strings.Join(fixes, "; "),
	}
}

func doctorSidecarLogHint() string {
	if p := strings.TrimSpace(os.Getenv("MACOS_RT_LOG")); p != "" {
		return p
	}
	return filepath.Join(os.TempDir(), "zerollama-runtime-sidecar.log")
}

func doctorCheckDarwinSidecarBootstrap() doctorCheck {
	if u := strings.TrimSpace(os.Getenv("ZEROLLAMA_RUNTIME_URL")); u != "" {
		base := strings.TrimSuffix(u, "/")
		if doctorHTTPReachable(base+"/health") {
			return doctorCheck{
				Name:   "darwin sidecar bootstrap",
				Status: "ok",
				Detail: "using external ZEROLLAMA_RUNTIME_URL=" + u,
			}
		}
		return doctorCheck{
			Name:    "darwin sidecar bootstrap",
			Status:  "warn",
			Detail:  "ZEROLLAMA_RUNTIME_URL=" + u + " but /health unreachable",
			FixHint: "unset ZEROLLAMA_RUNTIME_URL and restart zerollama serve (auto-spawns :8081), or start sidecar manually",
		}
	}
	if v := strings.TrimSpace(os.Getenv("ZEROLLAMA_RUNTIME_DARWIN_SIDECAR")); v == "0" || strings.EqualFold(v, "false") {
		return doctorCheck{
			Name:   "darwin sidecar bootstrap",
			Status: "ok",
			Detail: "disabled (ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=0)",
		}
	}
	if v := strings.TrimSpace(os.Getenv("ZEROLLAMA_RUNTIME")); v == "0" || strings.EqualFold(v, "false") {
		return doctorCheck{
			Name:   "darwin sidecar bootstrap",
			Status: "ok",
			Detail: "runtime off (ZEROLLAMA_RUNTIME=0)",
		}
	}
	logPath := doctorSidecarLogHint()
	if _, err := os.Stat(logPath); err == nil {
		return doctorCheck{
			Name:   "darwin sidecar bootstrap",
			Status: "ok",
			Detail: "sidecar log present at " + logPath,
		}
	}
	_, sidecarTCP := doctorProbeRuntimeSidecar()
	if len(sidecarTCP) > 0 {
		return doctorCheck{
			Name:   "darwin sidecar bootstrap",
			Status: "ok",
			Detail: "sidecar port open (reused or external; no log at " + logPath + ")",
		}
	}
	_, goTCP := doctorProbeGoAPI()
	if len(goTCP) > 0 {
		return doctorCheck{
			Name:    "darwin sidecar bootstrap",
			Status:  "warn",
			Detail:  "serve still starting (Go port open, sidecar not up yet) — wait for 'Listening on' in serve logs",
			FixHint: "bootstrap runs before HTTP is ready; if stuck >2min see serve stderr for bootstrap skipped/failed",
		}
	}
	return doctorCheck{
		Name:    "darwin sidecar bootstrap",
		Status:  "warn",
		Detail:  "no sidecar log at " + logPath + " — zerollama serve did not spawn sidecar (see serve stderr for bootstrap skipped/failed)",
		FixHint: "unset ZEROLLAMA_RUNTIME_URL; restart ./zerollama serve; watch for 'darwin runtime sidecar started' or bootstrap failed",
	}
}

func doctorCheckTrainingVenv(repo string) doctorCheck {
	py := filepath.Join(repo, ".venv-training", "bin", "python")
	if _, err := os.Stat(py); err != nil {
		return doctorCheck{
			Name:    "training/.venv-training",
			Status:  "warn",
			Detail:  "missing (optional for /api/train MPS LoRA)",
			FixHint: "TRAINING_UV_PYTHON_VER=\"$(./scripts/training_uv_venv.sh --embed-py)\" ./scripts/training_uv_venv.sh --verify (ABI must match ldd zerollama libpython)",
		}
	}
	cmd := exec.Command(py, "-c", "import torch, peft")
	if err := cmd.Run(); err != nil {
		return doctorCheck{
			Name:    "training/.venv-training",
			Status:  "warn",
			Detail:  "torch/peft import failed",
			FixHint: "TRAINING_UV_SYNC=1 TRAINING_UV_PYTHON_VER=\"$(./scripts/training_uv_venv.sh --embed-py)\" ./scripts/training_uv_venv.sh --verify",
		}
	}
	return doctorCheck{
		Name:   "training/.venv-training",
		Status: "ok",
		Detail: py,
	}
}

func doctorCheckSidecarHealth() doctorCheck {
	base, tcpOnly := doctorProbeRuntimeSidecar()
	if base == "" {
		detail := "no sidecar /health on loopback"
		if len(tcpOnly) > 0 {
			detail = fmt.Sprintf("%s reachable but /health not ready", tcpOnly[0])
		}
		return doctorCheck{
			Name:    "runtime sidecar",
			Status:  "warn",
			Detail:  detail,
			FixHint: "zerollama serve (Mac auto-starts sidecar) or see " + doctorSidecarLogHint(),
		}
	}
	url := base + "/health"
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return doctorCheck{
			Name:    "runtime sidecar",
			Status:  "warn",
			Detail:  fmt.Sprintf("%s unreachable", url),
			FixHint: "see " + doctorSidecarLogHint(),
		}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return doctorCheck{
			Name:    "runtime sidecar",
			Status:  "warn",
			Detail:  fmt.Sprintf("%s returned HTTP %d", url, resp.StatusCode),
			FixHint: "see " + doctorSidecarLogHint(),
		}
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return doctorCheck{
			Name:   "runtime sidecar",
			Status: "warn",
			Detail: "could not read /health",
		}
	}
	var h map[string]any
	if err := json.Unmarshal(body, &h); err != nil {
		return doctorCheck{
			Name:   "runtime sidecar",
			Status: "warn",
			Detail: "invalid /health JSON",
		}
	}
	c := doctorEvaluateSidecarHealth(h)
	if c.Status == "ok" {
		c.Detail = base + " — " + c.Detail
	}
	return c
}

func doctorEvaluateSidecarHealth(h map[string]any) doctorCheck {
	pick := ""
	if ac, ok := h["autoconfig"].(map[string]any); ok {
		pick, _ = ac["pick"].(string)
	}
	backend, _ := h["llama_backend"].(string)
	source, _ := h["llama_backend_source"].(string)
	probe, _ := h["vram_probe_effective"].(string)
	fallback, _ := h["llama_backend_fallback"].(bool)
	requested, _ := h["llama_backend_requested"].(string)
	detail := fmt.Sprintf(
		"pick=%s backend=%s source=%s probe=%s requested=%s fallback=%v",
		pick, backend, source, probe, requested, fallback,
	)
	status := "ok"
	var fixes []string
	if pick != "" && pick != "apple_silicon" && pick != "custom" {
		status = "warn"
		fixes = append(fixes, "unset ZEROLLAMA_RUNTIME_CONFIG or use apple_silicon.yaml autoconfig")
	}
	if source == "env" && backend != "inprocess" {
		status = "warn"
		fixes = append(fixes, "unset ZEROLLAMA_RUNTIME_LLAMA_BACKEND to use apple_silicon.yaml inprocess default")
	}
	if backend == "inprocess" && source == "default" {
		status = "warn"
		fixes = append(fixes, "inprocess without yaml/env — load apple_silicon.yaml or set ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess")
	}
	if fallback {
		status = "warn"
		fixes = append(fixes, "inprocess load failed; use a text-only GGUF or check libllama (see /health llama_backend_fallback)")
	}
	return doctorCheck{
		Name:    "runtime sidecar",
		Status:  status,
		Detail:  detail,
		FixHint: strings.Join(fixes, "; "),
	}
}

func doctorCheckTextGGUF(repo string) doctorCheck {
	py := filepath.Join(repo, "runtime", ".venv", "bin", "python")
	if _, err := os.Stat(py); err != nil {
		py = "python3"
	}
	cmd := exec.Command(py, "-c", doctorPickTextGGUFSnippet())
	out, err := cmd.Output()
	if err != nil {
		return doctorCheck{
			Name:    "text GGUF model",
			Status:  "warn",
			Detail:  "could not scan ~/.ollama/models",
			FixHint: "zerollama pull a small text model; set M3_LLAMA_MODEL for sign-off",
		}
	}
	line := strings.TrimSpace(string(out))
	if line == "" {
		return doctorCheck{
			Name:    "text GGUF model",
			Status:  "warn",
			Detail:  "no local text GGUF found",
			FixHint: "zerollama pull llama3.2:3b (or similar text-only model)",
		}
	}
	return doctorCheck{
		Name:   "text GGUF model",
		Status: "ok",
		Detail: line,
	}
}

func doctorPickTextGGUFSnippet() string {
	return `
import json
from pathlib import Path
root = Path.home() / ".ollama/models/manifests/registry.ollama.ai/library"
best = None
for mf in sorted(root.rglob("latest")):
    try:
        m = json.loads(mf.read_text())
        if any("projector" in (layer.get("mediaType") or "") for layer in m.get("layers", [])):
            continue
        cfg_path = Path.home() / ".ollama/models/blobs" / m["config"]["digest"].replace("sha256:", "sha256-")
        cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
        fam = (cfg.get("model_family") or "").lower()
        if fam in ("nomic-bert", "bert", "embed"):
            continue
        if "gemma" in fam and cfg.get("model_type") not in (None, "", "llama"):
            continue
        for layer in m.get("layers", []):
            if layer.get("mediaType") != "application/vnd.ollama.image.model":
                continue
            d = layer["digest"].replace("sha256:", "sha256-")
            path = Path.home() / ".ollama/models/blobs" / d
            size = int(layer.get("size") or 0)
            if path.is_file() and (best is None or size < best[0]):
                best = (size, str(path), mf.parent.name)
            break
    except Exception:
        pass
if best:
    print(f"{best[2]} -> {best[1]}")
`
}
