package autoroute

import (
	"fmt"
	"slices"
	"strings"

	"github.com/ollama/ollama/types/model"
)

// Surface is the HTTP entry point a request arrived on. It decides which model classes
// may serve it — a router name on /v1/images/* must never resolve to a chat model.
type Surface string

const (
	SurfaceChat      Surface = "chat"       // /api/chat, /v1/chat/completions, /v1/responses, /v1/messages
	SurfaceGenerate  Surface = "generate"   // /api/generate, /v1/completions (text)
	SurfaceImage     Surface = "image"      // /v1/images/generations, /api/generate with width/height
	SurfaceImageEdit Surface = "image_edit" // /v1/images/edits
)

// defaultReplyTokens is the reply budget assumed for context fit when the client sets no
// num_predict. WHY 1024: enough that a model whose window exactly equals the prompt is not
// chosen, small enough not to exclude 4k models for short prompts.
const defaultReplyTokens = 1024

// Signals are the routing-relevant facts read from the request itself — no model call.
type Signals struct {
	Surface Surface
	Text    string // last user message / prompt (what a classifier tier would see)

	HasImages  bool // stills in messages / images[]
	HasVideo   bool // video payload (pre-expansion videos[] / video spans)
	HasAudio   bool // audio clips
	HasTools   bool
	WantsThink bool // thinking explicitly requested (not alias no-ops)

	PromptTokens int // estimate; 0 = unknown (skips context fit)
	ReplyTokens  int // num_predict when > 0, else defaultReplyTokens
}

// Required returns the capabilities a candidate must have to serve the request. The set
// mirrors the handler gates (ChatHandler / GenerateHandler CheckCapabilities) so a routed
// request never fails capability checks downstream.
func (s Signals) Required() []model.Capability {
	switch s.Surface {
	case SurfaceImage, SurfaceImageEdit:
		return []model.Capability{model.CapabilityImage}
	}
	req := []model.Capability{model.CapabilityCompletion}
	switch {
	case s.HasVideo:
		req = append(req, model.CapabilityVision, model.CapabilityVideo)
	case s.HasImages:
		req = append(req, model.CapabilityVision)
	}
	if s.HasAudio {
		req = append(req, model.CapabilityAudio)
	}
	if s.HasTools {
		req = append(req, model.CapabilityTools)
	}
	if s.WantsThink {
		req = append(req, model.CapabilityThinking)
	}
	return req
}

// excludedForText are capabilities that mark a model as a different product even when its
// GGUF also reports completion (e.g. Laya decision encoders, MLX imagegen tags). Routing a
// chat turn to them would 501 or produce garbage.
var excludedForText = []model.Capability{
	model.CapabilityDecision,
	model.CapabilityEmbedding,
	model.CapabilityImage,
	model.CapabilityVideoGen,
	model.CapabilitySpeech,
}

// surfaceMismatch returns the first capability that disqualifies c from s's surface.
func (s Signals) surfaceMismatch(c Candidate) (model.Capability, bool) {
	switch s.Surface {
	case SurfaceChat, SurfaceGenerate:
		for _, x := range excludedForText {
			if slices.Contains(c.Capabilities, x) {
				return x, true
			}
		}
	}
	return "", false
}

// Admits reports whether c may serve the request (surface class + required capabilities),
// with the reason when it may not. Context fit and availability are not checked — this is
// the gate LA11 routers apply to the candidate their classifier picked.
func (s Signals) Admits(c Candidate) (string, bool) {
	if x, bad := s.surfaceMismatch(c); bad {
		return fmt.Sprintf("%s model cannot serve %s", x, s.Surface), false
	}
	if missing := missingCaps(c, s.Required()); len(missing) > 0 {
		return "missing capability: " + strings.Join(missing, ", "), false
	}
	return "", true
}

func (s Signals) contextNeed() int {
	if s.PromptTokens <= 0 {
		return 0
	}
	reply := s.ReplyTokens
	if reply <= 0 {
		reply = defaultReplyTokens
	}
	return s.PromptTokens + reply
}
