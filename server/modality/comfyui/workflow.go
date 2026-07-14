package comfyui

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Template is an on-disk workflow definition: a ComfyUI API-format graph plus a
// small "bindings" map describing which node input each logical field (prompt,
// seed, lora, ...) lives at.
//
// WHY bindings (not “edit the whole graph from the agent”): every workflow has
// different node ids and class_types (CLIPTextEncode vs TextEncodeQwenImageEdit).
// Agents select options.workflow + options.*; operators own the graph JSON.
type Template struct {
	// Graph is the raw ComfyUI API-format prompt graph (node id -> {class_type, inputs}).
	Graph map[string]any `json:"graph"`
	// Bindings maps logical field names to a node id + input key in Graph.
	Bindings map[string]Binding `json:"bindings"`
	// Description is shown by GET /api/image/workflows for agent discovery.
	Description string `json:"description,omitempty"`
	// Requires lists logical fields a caller must supply (e.g. "image" for edit workflows).
	Requires []string `json:"requires,omitempty"`
}

// Binding locates one workflow input: Graph[NodeID]["inputs"][Field] = value.
type Binding struct {
	NodeID string `json:"node_id"`
	Field  string `json:"field"`
}

// Known logical field names. Not all workflows support all fields — see Template.Bindings.
const (
	FieldPrompt          = "prompt"
	FieldNegativePrompt  = "negative_prompt"
	FieldSeed            = "seed"
	FieldWidth           = "width"
	FieldHeight          = "height"
	FieldSteps           = "steps"
	FieldImage           = "image" // LoadImage widget value (uploaded filename)
	FieldLoRAName        = "lora_name"
	FieldLoRAStrength    = "lora_strength"
	FieldControlImage    = "control_image" // LoadImage widget value for ControlNet input
	FieldControlStrength = "control_strength"
)

// LoadTemplate reads a workflow template JSON file from disk.
func LoadTemplate(path string) (*Template, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("comfyui: read workflow template %s: %w", path, err)
	}
	var t Template
	if err := json.Unmarshal(data, &t); err != nil {
		return nil, fmt.Errorf("comfyui: parse workflow template %s: %w", path, err)
	}
	if t.Graph == nil {
		return nil, fmt.Errorf("comfyui: workflow template %s has no graph", path)
	}
	return &t, nil
}

// LoadTemplateDir loads every "<name>.json" file in dir, keyed by name (without extension).
// Directory layout matches backend_paths.comfy_workflow_dir in a model manifest, e.g.
// t2i.json, edit.json, img2img.json, controlnet.json, upscale.json.
func LoadTemplateDir(dir string) (map[string]*Template, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("comfyui: read workflow dir %s: %w", dir, err)
	}
	out := make(map[string]*Template)
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		name := e.Name()[:len(e.Name())-len(".json")]
		tmpl, err := LoadTemplate(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, err
		}
		out[name] = tmpl
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("comfyui: no workflow templates (*.json) found in %s", dir)
	}
	return out, nil
}

// Inputs holds the resolved values an agent request supplies for one generation.
// Zero values mean "leave the template's default in place" except for Seed, which
// is only applied when SeedSet is true (0 is a valid explicit seed).
type Inputs struct {
	Prompt             string
	NegativePrompt     string
	Seed               int64
	SeedSet            bool
	Width              int32
	Height             int32
	Steps              int32
	Image              string // uploaded filename (LoadImage widget value)
	LoRAName           string
	LoRAStrength       float64
	LoRAStrengthSet    bool
	ControlImage       string
	ControlStrength    float64
	ControlStrengthSet bool
}

// Render deep-copies the template graph and applies inputs at their bound
// node/field locations. Unbound or empty fields are left untouched so a
// workflow's own defaults (steps, negative prompt, etc.) still apply.
func (t *Template) Render(in Inputs) (map[string]any, error) {
	graph, err := deepCopyGraph(t.Graph)
	if err != nil {
		return nil, err
	}

	set := func(field string, value any, has bool) error {
		if !has {
			return nil
		}
		b, ok := t.Bindings[field]
		if !ok {
			// WHY ignore unbound fields: one GenerateRequest shape serves t2i and
			// controlnet; optional options.lora must not fail a graph without LoRA.
			return nil
		}
		return setNodeInput(graph, b.NodeID, b.Field, value)
	}

	if err := set(FieldPrompt, in.Prompt, in.Prompt != ""); err != nil {
		return nil, err
	}
	if err := set(FieldNegativePrompt, in.NegativePrompt, in.NegativePrompt != ""); err != nil {
		return nil, err
	}
	if err := set(FieldSeed, in.Seed, in.SeedSet); err != nil {
		return nil, err
	}
	if err := set(FieldWidth, in.Width, in.Width > 0); err != nil {
		return nil, err
	}
	if err := set(FieldHeight, in.Height, in.Height > 0); err != nil {
		return nil, err
	}
	if err := set(FieldSteps, in.Steps, in.Steps > 0); err != nil {
		return nil, err
	}
	if err := set(FieldImage, in.Image, in.Image != ""); err != nil {
		return nil, err
	}
	if err := set(FieldLoRAName, in.LoRAName, in.LoRAName != ""); err != nil {
		return nil, err
	}
	if err := set(FieldLoRAStrength, in.LoRAStrength, in.LoRAStrengthSet); err != nil {
		return nil, err
	}
	if err := set(FieldControlImage, in.ControlImage, in.ControlImage != ""); err != nil {
		return nil, err
	}
	if err := set(FieldControlStrength, in.ControlStrength, in.ControlStrengthSet); err != nil {
		return nil, err
	}

	for _, req := range t.Requires {
		if _, ok := t.Bindings[req]; !ok {
			continue
		}
		if !fieldProvided(req, in) {
			return nil, fmt.Errorf("comfyui: workflow requires %q input", req)
		}
	}

	return graph, nil
}

func fieldProvided(field string, in Inputs) bool {
	switch field {
	case FieldPrompt:
		return in.Prompt != ""
	case FieldImage:
		return in.Image != ""
	case FieldControlImage:
		return in.ControlImage != ""
	default:
		return true
	}
}

func setNodeInput(graph map[string]any, nodeID, field string, value any) error {
	nodeRaw, ok := graph[nodeID]
	if !ok {
		return fmt.Errorf("comfyui: binding references unknown node id %q", nodeID)
	}
	node, ok := nodeRaw.(map[string]any)
	if !ok {
		return fmt.Errorf("comfyui: node %q is not an object", nodeID)
	}
	inputsRaw, ok := node["inputs"]
	if !ok {
		inputsRaw = map[string]any{}
		node["inputs"] = inputsRaw
	}
	inputs, ok := inputsRaw.(map[string]any)
	if !ok {
		return fmt.Errorf("comfyui: node %q inputs is not an object", nodeID)
	}
	inputs[field] = value
	return nil
}

// deepCopyGraph round-trips through JSON so per-request Render never mutates the
// shared template. WHY JSON (not maps.Clone): Comfy graphs nest maps/slices;
// a shallow clone would let prompt injection leak across concurrent requests.
func deepCopyGraph(graph map[string]any) (map[string]any, error) {
	data, err := json.Marshal(graph)
	if err != nil {
		return nil, fmt.Errorf("comfyui: copy graph: %w", err)
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("comfyui: copy graph: %w", err)
	}
	return out, nil
}
