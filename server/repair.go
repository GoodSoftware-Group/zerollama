// Manifest repair rewrites params/config/template layers from GGUF headers without
// re-downloading weights. Why: in-memory guess fixes load-time behavior but stale
// on-disk manifests confuse /api/show and fleet metadata until re-pull — see
// docs/localai-borrowings.md#manifest-hygiene-existing-tags.
package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"slices"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// RepairOptions controls manifest repair behavior.
type RepairOptions struct {
	// Write applies changes; default dry-run only reports diffs.
	Write bool
}

// RepairChange is one field or layer update proposed by repair.
type RepairChange struct {
	Field string `json:"field"`
	From  string `json:"from,omitempty"`
	To    string `json:"to,omitempty"`
}

// RepairResult summarizes repair for one model tag.
type RepairResult struct {
	Name    string         `json:"name"`
	Skipped bool           `json:"skipped,omitempty"`
	Reason  string         `json:"reason,omitempty"`
	Changes []RepairChange `json:"changes,omitempty"`
	Written bool           `json:"written,omitempty"`
}

// RepairModel plans or applies manifest metadata fixes for one tag.
func RepairModel(name string, opts RepairOptions) (*RepairResult, error) {
	n := model.ParseName(name)
	if !n.IsValid() {
		return nil, fmt.Errorf("invalid model name %q", name)
	}

	mf, err := manifest.ParseNamedManifest(n)
	if err != nil {
		return nil, err
	}

	result := &RepairResult{Name: n.String()}
	if ggufGuessingDisabled() {
		result.Skipped = true
		result.Reason = "GGUF guessing disabled (ZEROLLAMA_DISABLE_GGUF_GUESS or LOCALAI_DISABLE_GUESSING)"
		return result, nil
	}

	cfg, err := readManifestConfig(mf)
	if err != nil {
		return nil, err
	}
	oldCfg := cloneConfigV2(cfg)

	params := readManifestParams(mf.Layers)
	oldParams := cloneParams(params)
	hasParams := len(params) > 0
	hasTemplate := manifestHasLayer(mf.Layers, "application/vnd.ollama.image.template")

	ggufLayers, err := ggufMetadataLayersFromManifest(mf)
	if err != nil {
		return nil, err
	}
	if len(ggufLayers) == 0 {
		result.Skipped = true
		result.Reason = "no GGUF weight layers"
		return result, nil
	}

	newCfg := cloneConfigV2(cfg)
	newParams := cloneParams(params)
	guessFromBaseLayers(&newCfg, newParams, ggufLayers)
	repairGuessArchExtras(&newCfg, ggufLayers)
	repairCapNumCtx(newParams)

	var extraLayers []manifest.Layer
	if !hasTemplate {
		before := len(ggufLayers)
		augmented, err := detectChatTemplate(ggufLayers)
		if err != nil {
			return nil, err
		}
		extraLayers = extractRepairLayers(ggufLayers, augmented, hasParams)
		if len(augmented) > before {
			ggufLayers = augmented
		}
	}

	changes := diffConfig(oldCfg, newCfg)
	changes = append(changes, diffParams(oldParams, newParams)...)
	for _, layer := range extraLayers {
		changes = append(changes, RepairChange{
			Field: "layer." + layerMediaLabel(layer.MediaType),
			From:  "(missing)",
			To:    layer.Status,
		})
	}
	result.Changes = changes

	if len(changes) == 0 {
		return result, nil
	}
	if !opts.Write {
		return result, nil
	}

	layers := slices.Clone(mf.Layers)
	layers, err = setParameters(layers, newParams)
	if err != nil {
		return nil, err
	}
	layers = append(layers, extraLayers...)

	configLayer, err := createConfigLayer(layers, newCfg)
	if err != nil {
		return nil, err
	}
	if err := manifest.WriteManifest(n, *configLayer, layers); err != nil {
		return nil, err
	}
	result.Written = true
	return result, nil
}

// EnrichManifestAfterPull applies GGUF metadata hints to a freshly pulled manifest.
// Why: registry manifests often omit template layers and ship train-context num_ctx;
// repair logic is reused so pull/create/repair stay consistent. Failures are logged only.
func EnrichManifestAfterPull(name model.Name, fn func(api.ProgressResponse)) {
	if ggufGuessingDisabled() {
		return
	}
	if fn != nil {
		fn(api.ProgressResponse{Status: "enriching manifest"})
	}
	result, err := RepairModel(name.String(), RepairOptions{Write: true})
	if err != nil {
		slog.Warn("pull manifest enrich failed", "model", name.DisplayShortest(), "error", err)
		return
	}
	if result.Skipped {
		slog.Debug("pull manifest enrich skipped", "model", name.DisplayShortest(), "reason", result.Reason)
		return
	}
	if len(result.Changes) == 0 {
		return
	}
	slog.Info("pull manifest enriched", "model", name.DisplayShortest(), "changes", len(result.Changes), "written", result.Written)
}

// RepairAll repairs every local manifest tag.
func RepairAll(opts RepairOptions) ([]*RepairResult, error) {
	mfs, err := manifest.Manifests(false)
	if err != nil {
		return nil, err
	}

	names := make([]model.Name, 0, len(mfs))
	for name := range mfs {
		names = append(names, name)
	}
	slices.SortFunc(names, func(a, b model.Name) int {
		return strings.Compare(a.String(), b.String())
	})

	results := make([]*RepairResult, 0, len(names))
	for _, name := range names {
		r, err := RepairModel(name.String(), opts)
		if err != nil {
			return results, fmt.Errorf("%s: %w", name, err)
		}
		results = append(results, r)
	}
	return results, nil
}

// ExpireRunnerAfterRepair unloads a resident runner so the next load uses the repaired manifest.
func ExpireRunnerAfterRepair(s *Scheduler, name string) {
	if s == nil {
		return
	}
	m, err := GetModel(name)
	if err != nil {
		return
	}
	s.expireRunner(m)
}

func readManifestConfig(mf *manifest.Manifest) (model.ConfigV2, error) {
	var cfg model.ConfigV2
	if mf.Config.Digest == "" {
		return cfg, nil
	}
	path, err := manifest.BlobsPath(mf.Config.Digest)
	if err != nil {
		return cfg, err
	}
	f, err := os.Open(path)
	if err != nil {
		return cfg, err
	}
	defer f.Close()
	if err := json.NewDecoder(f).Decode(&cfg); err != nil {
		return cfg, err
	}
	return cfg, nil
}

func readManifestParams(layers []manifest.Layer) map[string]any {
	var params map[string]any
	for _, layer := range layers {
		if layer.MediaType != "application/vnd.ollama.image.params" {
			continue
		}
		path, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			continue
		}
		f, err := os.Open(path)
		if err != nil {
			continue
		}
		var decoded map[string]any
		if err := json.NewDecoder(f).Decode(&decoded); err != nil {
			_ = f.Close()
			continue
		}
		_ = f.Close()
		if params == nil {
			params = decoded
		} else {
			for k, v := range decoded {
				params[k] = v
			}
		}
	}
	if params == nil {
		return make(map[string]any)
	}
	return params
}

func ggufMetadataLayersFromManifest(mf *manifest.Manifest) ([]*layerGGML, error) {
	var layers []*layerGGML
	for _, src := range mf.Layers {
		switch src.MediaType {
		case "application/vnd.ollama.image.model",
			"application/vnd.ollama.image.projector",
			"application/vnd.ollama.image.adapter",
			manifest.MediaTypeImageDraft:
			path, err := manifest.BlobsPath(src.Digest)
			if err != nil {
				return nil, err
			}
			f, err := loadGGUFMetadataAt(path)
			if err != nil {
				continue
			}
			layer, err := manifest.NewLayerFromLayer(src.Digest, src.MediaType, src.Name)
			if err != nil {
				return nil, err
			}
			layers = append(layers, &layerGGML{Layer: layer, GGML: f})
		}
	}
	return layers, nil
}

func manifestHasLayer(layers []manifest.Layer, mediaType string) bool {
	return slices.ContainsFunc(layers, func(l manifest.Layer) bool {
		return l.MediaType == mediaType
	})
}

func repairGuessArchExtras(cfg *model.ConfigV2, ggufLayers []*layerGGML) {
	if cfg == nil {
		return
	}
	layer := guessPrimaryGGUFLayer(ggufLayers)
	if layer == nil || layer.GGML == nil {
		return
	}
	switch layer.GGML.KV().Architecture() {
	case "gemma4":
		if cfg.Renderer == "" {
			cfg.Renderer = gemma4RendererLegacy
		}
		if cfg.Parser == "" {
			cfg.Parser = "gemma4"
		}
	case "qwen35", "qwen35moe":
		if cfg.Renderer == "" {
			cfg.Renderer = "qwen3.5"
		}
		if cfg.Parser == "" {
			cfg.Parser = "qwen3.5"
		}
	default:
		if isGptOSSFamily(layer.GGML.KV().Architecture()) {
			if cfg.Renderer == "" {
				cfg.Renderer = "harmony"
			}
			if cfg.Parser == "" {
				cfg.Parser = "harmony"
			}
		}
	}
}

// repairCapNumCtx lowers manifest num_ctx above the guess cap (unlike GuessParametersFromGGUF).
func repairCapNumCtx(params map[string]any) {
	if params == nil {
		return
	}
	n, ok := paramAsInt(params["num_ctx"])
	if !ok || n <= defaultManifestNumCtxCap {
		return
	}
	params["num_ctx"] = defaultManifestNumCtxCap
}

func extractRepairLayers(before, after []*layerGGML, skipParams bool) []manifest.Layer {
	var added []manifest.Layer
	for i := len(before); i < len(after); i++ {
		l := after[i]
		if skipParams && l.MediaType == "application/vnd.ollama.image.params" {
			continue
		}
		added = append(added, l.Layer)
	}
	return added
}

func cloneConfigV2(cfg model.ConfigV2) model.ConfigV2 {
	b, err := json.Marshal(cfg)
	if err != nil {
		return cfg
	}
	var out model.ConfigV2
	if err := json.Unmarshal(b, &out); err != nil {
		return cfg
	}
	return out
}

func cloneParams(p map[string]any) map[string]any {
	if p == nil {
		return make(map[string]any)
	}
	b, err := json.Marshal(p)
	if err != nil {
		return mapsClone(p)
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		return mapsClone(p)
	}
	return out
}

func mapsClone(p map[string]any) map[string]any {
	out := make(map[string]any, len(p))
	for k, v := range p {
		out[k] = v
	}
	return out
}

func paramAsInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int32:
		return int(n), true
	case int64:
		return int(n), true
	case float32:
		return int(n), true
	case float64:
		return int(n), true
	case json.Number:
		i, err := n.Int64()
		if err != nil {
			return 0, false
		}
		return int(i), true
	default:
		return 0, false
	}
}

func diffConfig(old, new model.ConfigV2) []RepairChange {
	var changes []RepairChange
	add := func(field, from, to string) {
		if from == to {
			return
		}
		changes = append(changes, RepairChange{Field: "config." + field, From: from, To: to})
	}
	add("model_family", old.ModelFamily, new.ModelFamily)
	add("model_type", old.ModelType, new.ModelType)
	add("file_type", old.FileType, new.FileType)
	add("renderer", old.Renderer, new.Renderer)
	add("parser", old.Parser, new.Parser)
	if old.ContextLen != new.ContextLen {
		add("context_length", fmt.Sprint(old.ContextLen), fmt.Sprint(new.ContextLen))
	}
	if old.EmbedLen != new.EmbedLen {
		add("embedding_length", fmt.Sprint(old.EmbedLen), fmt.Sprint(new.EmbedLen))
	}
	if !slices.Equal(old.ModelFamilies, new.ModelFamilies) {
		add("model_families", fmt.Sprint(old.ModelFamilies), fmt.Sprint(new.ModelFamilies))
	}
	return changes
}

func diffParams(old, new map[string]any) []RepairChange {
	var changes []RepairChange
	keys := make([]string, 0, len(new))
	for k := range new {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, k := range keys {
		oldVal, had := old[k]
		newVal := new[k]
		if had && jsonEqual(oldVal, newVal) {
			continue
		}
		from := ""
		if had {
			from = fmt.Sprintf("%v", oldVal)
		}
		changes = append(changes, RepairChange{
			Field: "params." + k,
			From:  from,
			To:    fmt.Sprintf("%v", newVal),
		})
	}
	return changes
}

func jsonEqual(a, b any) bool {
	ab, err := json.Marshal(a)
	if err != nil {
		return false
	}
	bb, err := json.Marshal(b)
	if err != nil {
		return false
	}
	return string(ab) == string(bb)
}

func layerMediaLabel(mediaType string) string {
	switch mediaType {
	case "application/vnd.ollama.image.template":
		return "template"
	case "application/vnd.ollama.image.params":
		return "params"
	default:
		return mediaType
	}
}

// RepairHandler rewrites manifest metadata layers from GGUF headers.
func (s *Server) RepairHandler(c *gin.Context) {
	var req api.RepairRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	opts := RepairOptions{Write: req.Write}
	var results []*RepairResult
	var err error
	if req.All {
		results, err = RepairAll(opts)
	} else {
		models := req.Models
		if len(models) == 0 && strings.TrimSpace(req.Model) != "" {
			models = []string{req.Model}
		}
		if len(models) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model, models, or all is required"})
			return
		}
		for _, name := range models {
			r, rerr := RepairModel(name, opts)
			if rerr != nil {
				c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": rerr.Error()})
				return
			}
			results = append(results, r)
		}
	}
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	for _, r := range results {
		if r != nil && r.Written {
			ExpireRunnerAfterRepair(s.sched, r.Name)
		}
	}
	c.JSON(http.StatusOK, api.RepairResponse{Results: repairResultsToAPI(results)})
}

func repairResultsToAPI(results []*RepairResult) []api.RepairResult {
	out := make([]api.RepairResult, 0, len(results))
	for _, r := range results {
		if r == nil {
			continue
		}
		changes := make([]api.RepairChange, len(r.Changes))
		for i, c := range r.Changes {
			changes[i] = api.RepairChange{Field: c.Field, From: c.From, To: c.To}
		}
		out = append(out, api.RepairResult{
			Name:    r.Name,
			Skipped: r.Skipped,
			Reason:  r.Reason,
			Changes: changes,
			Written: r.Written,
		})
	}
	return out
}
