package modelhealth

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// Trap IDs map to https://github.com/Blackwellboy/model-serving-minefield entries.
const (
	TrapQuantLabelMismatch = "10" // quant label is not the kernel path / file_type
	TrapNoGenerationConfig = "21" // no generation_config → server defaults win
	TrapContextMismatch    = "55" // advertised / trained / served context differ
	TrapNoChatTemplate     = "56" // checkpoint ships no renderable chat template
)

// samplingParamKeys are generation_config-equivalent keys in an Ollama params layer.
var samplingParamKeys = []string{
	"temperature", "top_p", "top_k", "min_p", "typical_p",
	"repeat_penalty", "presence_penalty", "frequency_penalty",
	"seed", "mirostat", "mirostat_tau", "mirostat_eta",
}

// quantTagRe extracts a quant hint from a model tag (e.g. q4_k_m, Q8_0, iq4_xs).
var quantTagRe = regexp.MustCompile(`(?i)(?:^|[-_:])(q[2-8](?:_[a-z0-9]+)+|q[2-8]|iq[1-4]_[a-z]+|f16|f32|bf16|mxfp4)(?:$|[-_:])`)

// CheckConfigTrapsAll walks every local manifest and returns minefield-style
// config trap reports. Models with missing blobs are skipped (blob integrity
// is covered by CheckAll).
func CheckConfigTrapsAll() ([]Report, error) {
	mfs, err := manifest.ManifestsSearch(envconfig.ModelsSearchDirs(), true)
	if err != nil {
		return nil, err
	}

	names := make([]model.Name, 0, len(mfs))
	for n := range mfs {
		names = append(names, n)
	}
	slices.SortFunc(names, func(a, b model.Name) int {
		return strings.Compare(a.DisplayShortest(), b.DisplayShortest())
	})

	var out []Report
	for _, n := range names {
		mf := mfs[n]
		display := n.DisplayShortest()
		modelsDir := modelsDirFor(n, mf)
		if modelsDir == "" {
			continue
		}
		if HasMissingBlobsIn(modelsDir, mf) {
			continue
		}
		out = append(out, checkConfigTrapsIn(modelsDir, display, mf)...)
	}
	return out, nil
}

// HasMissingBlobsIn is like HasMissingBlobs but scoped to an explicit models dir.
func HasMissingBlobsIn(modelsDir string, mf *manifest.Manifest) bool {
	missing, err := missingBlobPathsIn(modelsDir, mf)
	return err != nil || len(missing) > 0
}

func modelsDirFor(n model.Name, mf *manifest.Manifest) string {
	for _, modelsDir := range envconfig.ModelsSearchDirs() {
		found, err := manifest.ParseNamedManifestIn(modelsDir, n)
		if err != nil {
			continue
		}
		if found.Digest() == mf.Digest() {
			return modelsDir
		}
	}
	return ""
}

// checkConfigTrapsIn evaluates traps 10, 21, 55/61, and 56 for one manifest.
func checkConfigTrapsIn(modelsDir, display string, mf *manifest.Manifest) []Report {
	cfg, err := readConfigV2(modelsDir, mf)
	if err != nil {
		return []Report{{
			Name:    fmt.Sprintf("model %s config-traps", display),
			Status:  StatusRepairable,
			Detail:  "could not read config: " + err.Error(),
			FixHint: "zerollama show " + display,
		}}
	}

	modelPath := layerPath(modelsDir, mf, "application/vnd.ollama.image.model")
	paramsPath := layerPath(modelsDir, mf, "application/vnd.ollama.image.params")
	hasGoTemplate := layerPath(modelsDir, mf, "application/vnd.ollama.image.template") != "" ||
		layerPath(modelsDir, mf, "application/vnd.ollama.image.prompt") != ""

	var ggufChatTemplate string
	var ggufFileType string
	var ggufCtxLen int
	isGGUF := modelPath != "" && (strings.EqualFold(cfg.ModelFormat, "gguf") ||
		(cfg.ModelFormat == "" && looksLikeGGUF(modelPath)))
	if isGGUF {
		if f, err := gguf.Open(modelPath); err == nil {
			ggufChatTemplate = f.KeyValue("tokenizer.chat_template").String()
			if ft := f.KeyValue("general.file_type").Uint(); ft > 0 {
				ggufFileType = ggml.FileType(ft).String()
			}
			if n := f.KeyValue("context_length").Uint(); n > 0 {
				ggufCtxLen = int(n)
			}
			_ = f.Close()
		}
	}

	params, _ := readParams(paramsPath)

	var out []Report
	out = append(out, trap21NoGenerationConfig(display, paramsPath, params))
	out = append(out, trap10QuantLabel(display, nTagQuant(display), cfg.FileType, ggufFileType))
	out = append(out, trap56NoChatTemplate(display, hasGoTemplate, ggufChatTemplate, cfg.Renderer))
	out = append(out, trap55ContextMismatch(display, cfg.ContextLen, paramsNumCtx(params), ggufCtxLen))
	return out
}

func trap21NoGenerationConfig(display, paramsPath string, params map[string]any) Report {
	name := fmt.Sprintf("model %s trap-21 (generation defaults)", display)
	if paramsPath == "" || len(params) == 0 {
		return Report{
			Name:    name,
			Status:  StatusRepairable,
			Detail:  "no params layer — server built-in sampling defaults win (minefield trap 21)",
			FixHint: "pin temperature/top_p/top_k in the Modelfile PARAMETER lines if you need reproducible defaults",
		}
	}
	for _, k := range samplingParamKeys {
		if _, ok := params[k]; ok {
			return Report{Name: name, Status: StatusOK, Detail: "params include sampling keys"}
		}
	}
	return Report{
		Name:    name,
		Status:  StatusRepairable,
		Detail:  "params layer has no sampling keys — temperature/top_p come from server defaults (minefield trap 21)",
		FixHint: "add PARAMETER temperature / top_p / top_k to the Modelfile if defaults must be pinned",
	}
}

func trap10QuantLabel(display, tagQuant, configFileType, ggufFileType string) Report {
	name := fmt.Sprintf("model %s trap-10 (quant label)", display)
	actual := firstNonEmpty(ggufFileType, configFileType)
	if actual == "" || actual == "unknown" {
		return Report{Name: name, Status: StatusOK, Detail: "no GGUF file_type to compare"}
	}
	if tagQuant == "" {
		if configFileType != "" && ggufFileType != "" && !quantEqual(configFileType, ggufFileType) {
			return Report{
				Name:    name,
				Status:  StatusRepairable,
				Detail:  fmt.Sprintf("config file_type=%s differs from GGUF general.file_type=%s (minefield trap 10)", configFileType, ggufFileType),
				FixHint: "trust GGUF general.file_type; re-create or repair the manifest if the label is wrong",
			}
		}
		return Report{Name: name, Status: StatusOK, Detail: "file_type=" + actual}
	}
	if !quantEqual(tagQuant, actual) {
		return Report{
			Name:    name,
			Status:  StatusRepairable,
			Detail:  fmt.Sprintf("tag hint %q does not match actual file_type %s (minefield trap 10)", tagQuant, actual),
			FixHint: "read config/GGUF file_type, not the tag or repo name, when comparing quants",
		}
	}
	return Report{Name: name, Status: StatusOK, Detail: fmt.Sprintf("tag %s matches file_type %s", tagQuant, actual)}
}

func trap56NoChatTemplate(display string, hasGoTemplate bool, ggufChatTemplate, renderer string) Report {
	name := fmt.Sprintf("model %s trap-56 (chat template)", display)
	if hasGoTemplate || renderer != "" {
		return Report{Name: name, Status: StatusOK, Detail: "Go template or renderer present"}
	}
	if strings.TrimSpace(ggufChatTemplate) != "" {
		// GGUF Jinja only — fine for llama-server --jinja, but note for Go-render path.
		if looksLikePythonOnlyTemplate(ggufChatTemplate) {
			return Report{
				Name:    name,
				Status:  StatusRepairable,
				Detail:  "GGUF chat_template looks Python-only (not portable Jinja) (minefield trap 56)",
				FixHint: "supply a Modelfile TEMPLATE or use a stack that embeds the checkpoint's Python renderer",
			}
		}
		return Report{Name: name, Status: StatusOK, Detail: "GGUF tokenizer.chat_template present"}
	}
	return Report{
		Name:    name,
		Status:  StatusRepairable,
		Detail:  "no Go template layer and no GGUF chat_template (minefield trap 56)",
		FixHint: "add a Modelfile TEMPLATE, or serve via a path that ships a family renderer",
	}
}

func trap55ContextMismatch(display string, advertised, served, trained int) Report {
	name := fmt.Sprintf("model %s trap-55/61 (context)", display)
	parts := []string{}
	if advertised > 0 {
		parts = append(parts, fmt.Sprintf("advertised=%d", advertised))
	}
	if served > 0 {
		parts = append(parts, fmt.Sprintf("served(num_ctx)=%d", served))
	}
	if trained > 0 {
		parts = append(parts, fmt.Sprintf("trained(gguf)=%d", trained))
	}
	if len(parts) < 2 {
		return Report{Name: name, Status: StatusOK, Detail: "insufficient context fields to compare"}
	}
	detail := strings.Join(parts, " ")
	warn := false
	if advertised > 0 && trained > 0 && divergeContext(advertised, trained) {
		warn = true
	}
	if served > 0 && trained > 0 && served > trained {
		warn = true
	}
	if advertised > 0 && served > 0 && divergeContext(advertised, served) {
		warn = true
	}
	if warn {
		return Report{
			Name:    name,
			Status:  StatusRepairable,
			Detail:  detail + " — advertised/trained/served context differ (minefield trap 55/61)",
			FixHint: "treat advertised, trained (GGUF context_length), and served (params num_ctx / loaded runner) as three numbers; keep manifest num_ctx modest",
		}
	}
	return Report{Name: name, Status: StatusOK, Detail: detail}
}

func divergeContext(a, b int) bool {
	if a <= 0 || b <= 0 {
		return false
	}
	lo, hi := a, b
	if lo > hi {
		lo, hi = hi, lo
	}
	// Flag when they differ by more than 2× (common: 4k served vs 128k+ advertised).
	return hi > lo*2
}

func readConfigV2(modelsDir string, mf *manifest.Manifest) (model.ConfigV2, error) {
	var cfg model.ConfigV2
	path, err := manifest.BlobsPathIn(modelsDir, mf.Config.Digest)
	if err != nil {
		return cfg, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, err
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, err
	}
	return cfg, nil
}

func layerPath(modelsDir string, mf *manifest.Manifest, mediaType string) string {
	for _, layer := range mf.Layers {
		if layer.MediaType != mediaType {
			continue
		}
		path, err := manifest.BlobsPathIn(modelsDir, layer.Digest)
		if err != nil {
			return ""
		}
		if blobAccessible(path) {
			return path
		}
	}
	return ""
}

func readParams(path string) (map[string]any, error) {
	if path == "" {
		return nil, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var params map[string]any
	if err := json.Unmarshal(data, &params); err != nil {
		return nil, err
	}
	return params, nil
}

func paramsNumCtx(params map[string]any) int {
	if params == nil {
		return 0
	}
	v, ok := params["num_ctx"]
	if !ok {
		return 0
	}
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	case json.Number:
		i, _ := n.Int64()
		return int(i)
	case string:
		i, _ := strconv.Atoi(n)
		return i
	default:
		return 0
	}
}

func nTagQuant(display string) string {
	base := filepath.Base(display)
	m := quantTagRe.FindStringSubmatch(base)
	if len(m) < 2 {
		return ""
	}
	return strings.ToUpper(strings.ReplaceAll(m[1], "-", "_"))
}

func quantEqual(a, b string) bool {
	na := normalizeQuant(a)
	nb := normalizeQuant(b)
	if na == "" || nb == "" {
		return false
	}
	if na == nb {
		return true
	}
	// Q4_K and Q4_K_M are often used interchangeably in tags.
	if (na == "Q4_K" && nb == "Q4_K_M") || (na == "Q4_K_M" && nb == "Q4_K") {
		return true
	}
	return false
}

func normalizeQuant(s string) string {
	s = strings.TrimSpace(strings.ToUpper(s))
	s = strings.ReplaceAll(s, "-", "_")
	s = strings.TrimPrefix(s, "GGML_")
	return s
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func looksLikeGGUF(path string) bool {
	if path == "" {
		return false
	}
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	var magic [4]byte
	if _, err := f.Read(magic[:]); err != nil {
		return false
	}
	// gguf.Open rejects lowercase "gguf"; real files use "GGUF".
	return string(magic[:]) == "GGUF" || string(magic[:]) == "gguf"
}

// looksLikePythonOnlyTemplate detects checkpoint templates that are Python source
// rather than Jinja (minefield trap 56).
func looksLikePythonOnlyTemplate(s string) bool {
	trim := strings.TrimSpace(s)
	if strings.HasPrefix(trim, "def ") || strings.Contains(trim, "raise Exception") {
		return true
	}
	if strings.Contains(trim, "import ") && !strings.Contains(trim, "{{") {
		return true
	}
	return false
}
