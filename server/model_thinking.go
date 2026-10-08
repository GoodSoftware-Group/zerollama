package server

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/types/model"
)

func (s *Server) thinkingInputError(ctx context.Context, name string, inputErr error) error {
	ref, err := parseAndValidateModelRef(name)
	if err != nil {
		return inputErr
	}
	var thinking *model.Thinking
	if canonical, err := getExistingName(ref.Name); err == nil {
		if m, err := GetModel(canonical.String()); err == nil {
			thinking = m.Thinking()
		}
	}
	if !thinking.Valid() {
		return inputErr
	}
	values, _ := json.Marshal(thinking.Values)
	return fmt.Errorf("%w; supported values: %s", inputErr, values)
}

// Thinking returns the effective local serving contract. Remote models obtain
// their current contract from the remote show response.
func (m *Model) Thinking() *model.Thinking {
	if m == nil || m.Config.RemoteHost != "" {
		return nil
	}
	if shouldUseHarmony(m) {
		return renderers.ThinkingForRenderer("harmony")
	}
	if name := resolveRendererName(m); name != "" {
		thinking := renderers.ThinkingForRenderer(name)
		if thinking == nil {
			return nil
		}
		// Discovery must match the controls accepted by the serving boundary.
		if !slices.Contains(m.Capabilities(), model.CapabilityThinking) {
			return &model.Thinking{Values: []any{false}, Default: false}
		}
		// Preserve the local endpoint's historical default-on behavior.
		if thinking.Default == false && thinking.Supports(true) {
			thinking.Default = true
		}
		return thinking
	}
	return nil
}

// genericThinking excludes Harmony and template-only paths from new fallback rules.
func (m *Model) genericThinking() *model.Thinking {
	if m == nil || m.Config.Renderer == "" || shouldUseHarmony(m) {
		return nil
	}
	return m.Thinking()
}

// lookupThinking lets compatibility middleware preserve named efforts only for
// local generic renderers. Other models keep the existing protocol conversions.
func lookupThinking(name string) *model.Thinking {
	ref, err := parseAndValidateModelRef(name)
	if err != nil || ref.Source == modelSourceCloud {
		return nil
	}
	canonical, err := getExistingName(ref.Name)
	if err != nil {
		return nil
	}
	m, err := GetModel(canonical.String())
	if err != nil {
		return nil
	}
	return m.genericThinking()
}
