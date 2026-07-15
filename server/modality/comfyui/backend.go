package comfyui

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
)

// Request is the modality-agnostic input to a ComfyUI generation, already extracted
// from api.GenerateRequest / options by the caller (server/routes.go) so this package
// has no dependency on the server's HTTP types.
type Request struct {
	// WorkflowDir is backend_paths.comfy_workflow_dir from the model manifest.
	WorkflowDir string
	// Workflow is the requested template name (options.workflow), or "" for the model's default.
	Workflow string
	// DefaultWorkflow is used when Workflow is empty. Falls back to "t2i" if also empty.
	DefaultWorkflow string

	Prompt         string
	NegativePrompt string
	Width          int32
	Height         int32
	Steps          int32
	Seed           int64
	SeedSet        bool

	// Image is a raw input image (edit/img2img workflows). Empty when not editing.
	Image []byte
	// ControlImage is a raw ControlNet conditioning image.
	ControlImage []byte

	LoRAName           string
	LoRAStrength       float64
	LoRAStrengthSet    bool
	ControlStrength    float64
	ControlStrengthSet bool
}

// Result is a completed generation.
type Result struct {
	PNG      []byte
	Workflow string
}

// Generate renders the requested (or default) workflow template, queues it on the
// configured ComfyUI server, waits for completion, and downloads the output PNG.
//
// WHY no VRAM broker call here: callers (handleImageGenerate) already run
// vram.PrepareForImageGen before dispatching to any image backend — Comfy is treated
// like MLX imagegen for exclusive-GPU purposes, unlike external-image which predates
// that convention.
func Generate(ctx context.Context, req Request) (Result, error) {
	if req.WorkflowDir == "" {
		return Result{}, fmt.Errorf("comfyui: model manifest is missing backend_paths.comfy_workflow_dir")
	}
	workflowDir := resolveWorkflowDir(req.WorkflowDir)
	templates, err := LoadTemplateDir(workflowDir)
	if err != nil {
		return Result{}, err
	}

	name := req.Workflow
	if name == "" {
		name = req.DefaultWorkflow
	}
	if name == "" {
		name = "t2i"
	}
	tmpl, ok := templates[name]
	if !ok {
		return Result{}, fmt.Errorf("comfyui: workflow %q not found in %s (available: %s)", name, workflowDir, availableNames(templates))
	}

	baseURL := envconfig.ComfyUIURL()
	timeout := envconfig.ModalityComfyUITimeout()
	client := NewClient(baseURL, timeout)

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	in := Inputs{
		Prompt:             req.Prompt,
		NegativePrompt:     req.NegativePrompt,
		Seed:               req.Seed,
		SeedSet:            req.SeedSet,
		Width:              req.Width,
		Height:             req.Height,
		Steps:              req.Steps,
		LoRAName:           req.LoRAName,
		LoRAStrength:       req.LoRAStrength,
		LoRAStrengthSet:    req.LoRAStrengthSet,
		ControlStrength:    req.ControlStrength,
		ControlStrengthSet: req.ControlStrengthSet,
	}

	if len(req.Image) > 0 {
		uploaded, err := client.UploadImage(ctx, "agent-input.png", req.Image)
		if err != nil {
			return Result{}, err
		}
		in.Image = uploaded.Name
	}
	if len(req.ControlImage) > 0 {
		uploaded, err := client.UploadImage(ctx, "agent-control.png", req.ControlImage)
		if err != nil {
			return Result{}, err
		}
		in.ControlImage = uploaded.Name
	}

	graph, err := tmpl.Render(in)
	if err != nil {
		return Result{}, fmt.Errorf("comfyui: workflow %q: %w", name, err)
	}

	promptID, err := client.QueuePrompt(ctx, graph, "zerollama")
	if err != nil {
		return Result{}, fmt.Errorf("comfyui: workflow %q: %w", name, err)
	}

	img, err := client.PollHistory(ctx, promptID, 1500*time.Millisecond)
	if err != nil {
		return Result{}, fmt.Errorf("comfyui: workflow %q: %w", name, err)
	}

	png, err := client.FetchImage(ctx, img)
	if err != nil {
		return Result{}, fmt.Errorf("comfyui: workflow %q: %w", name, err)
	}

	return Result{PNG: png, Workflow: name}, nil
}

// ListWorkflows loads templates from dir and returns their names, descriptions, and
// required fields — used by GET /api/image/workflows so agents can discover what a
// model supports without reading ComfyUI graph JSON.
func ListWorkflows(dir string) ([]WorkflowInfo, error) {
	templates, err := LoadTemplateDir(resolveWorkflowDir(dir))
	if err != nil {
		return nil, err
	}
	out := make([]WorkflowInfo, 0, len(templates))
	for name, tmpl := range templates {
		fields := make([]string, 0, len(tmpl.Bindings))
		for field := range tmpl.Bindings {
			fields = append(fields, field)
		}
		out = append(out, WorkflowInfo{
			Name:        name,
			Description: tmpl.Description,
			Requires:    tmpl.Requires,
			Fields:      fields,
		})
	}
	return out, nil
}

// WorkflowInfo is the discovery payload for GET /api/image/workflows.
type WorkflowInfo struct {
	Name        string   `json:"name"`
	Description string   `json:"description,omitempty"`
	Requires    []string `json:"requires,omitempty"`
	Fields      []string `json:"fields"`
}

// resolveWorkflowDir expands "~/" and joins relative dirs against
// OLLAMA_COMFYUI_WORKFLOWS_ROOT when set.
//
// WHY: manifests ship short paths (scripts/comfyui/qwen-image). Depending on cwd
// alone breaks production serve from ~/bin; an explicit root (or absolute paths)
// is the Wan-style operator fix without rewriting every modelfile.
func resolveWorkflowDir(dir string) string {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return dir
	}
	if strings.HasPrefix(dir, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, dir[2:])
		}
		return dir
	}
	if filepath.IsAbs(dir) {
		return dir
	}
	if root := envconfig.ComfyUIWorkflowsRoot(); root != "" {
		return filepath.Join(root, dir)
	}
	return dir
}

func availableNames(templates map[string]*Template) string {
	names := make([]string, 0, len(templates))
	for name := range templates {
		names = append(names, name)
	}
	return fmt.Sprintf("%v", names)
}
