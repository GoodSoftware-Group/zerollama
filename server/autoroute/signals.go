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

	SurfaceVideoGen   Surface = "video_gen"  // /v1/videos
	SurfaceSpeech     Surface = "speech"     // /v1/audio/speech
	SurfaceMusic      Surface = "music"      // /v1/audio/generations
	SurfaceTranscribe Surface = "transcribe" // /v1/audio/transcriptions
	SurfaceEmbed      Surface = "embed"      // /api/embed, /api/embeddings, /v1/embeddings
	SurfaceRerank     Surface = "rerank"     // /v1/rerank and aliases
	SurfaceDecisions  Surface = "decisions"  // /v1/decisions, /v1/systemone
)

// IsText reports whether s is an autoregressive text surface (chat / generate).
func (s Surface) IsText() bool { return s == SurfaceChat || s == SurfaceGenerate }

// Pinned reports whether s must resolve to one fixed model rather than a per-request pick.
// WHY: embedding spaces are incompatible across models — two calls in one RAG pipeline
// routed to different embedders return vectors that silently do not compare. Rerank
// scores are likewise only comparable within one model.
func (s Surface) Pinned() bool { return s == SurfaceEmbed || s == SurfaceRerank }

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

	Voice string // speech: requested voice; soft filter — models listing other voices lose to ones offering it

	// BeforeRuntimeProxy is true when routing runs ahead of the runtime / SGLang proxy
	// middlewares, so runtime-sidecar models are servable on this path.
	BeforeRuntimeProxy bool

	// Check is an optional surface-specific validator supplied by the server for facts that
	// need the model config (video admission, decider image support). A non-empty return
	// rejects the candidate with that reason.
	Check func(Candidate) string `json:"-"`

	// Bonus optionally adds a score and a reason (e.g. calibrated deciders first).
	Bonus func(Candidate) (float64, string) `json:"-"`

	// Weights overrides DefaultWeights (router.yaml weights for classifier: jev routers).
	Weights *Weights `json:"-"`

	// SessionKey identifies the conversation for sticky routing (AR6): the client's
	// prompt_cache_key / session id, else a hash of system + first user message. Empty ⇒
	// no stickiness (single-shot surfaces).
	SessionKey string
	// Fresh (options.zerollama.route: "fresh") skips remembered picks for this request;
	// the new decision is still remembered.
	Fresh bool
}

// Required returns the capabilities a candidate must have to serve the request. The set
// mirrors the handler gates (ChatHandler / GenerateHandler CheckCapabilities) so a routed
// request never fails capability checks downstream.
func (s Signals) Required() []model.Capability {
	switch s.Surface {
	case SurfaceImage, SurfaceImageEdit:
		return []model.Capability{model.CapabilityImage}
	case SurfaceVideoGen:
		return []model.Capability{model.CapabilityVideoGen}
	case SurfaceSpeech, SurfaceMusic:
		return []model.Capability{model.CapabilitySpeech}
	case SurfaceEmbed, SurfaceRerank:
		return []model.Capability{model.CapabilityEmbedding}
	case SurfaceDecisions:
		return []model.Capability{model.CapabilityDecision}
	case SurfaceTranscribe:
		return nil // whisper backend OR audio-capable chat model — see surfaceMismatch
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

// surfaceMismatch returns why c's model class cannot serve s's surface, mirroring the
// backend checks each handler makes after its capability gate.
func (s Signals) surfaceMismatch(c Candidate) (string, bool) {
	switch s.Surface {
	case SurfaceChat, SurfaceGenerate:
		for _, x := range excludedForText {
			if slices.Contains(c.Capabilities, x) {
				return fmt.Sprintf("%s model cannot serve %s", x, s.Surface), true
			}
		}
	case SurfaceSpeech:
		if c.Backend(model.ModalitySpeech) == model.BackendMusic3 {
			return "music3 backend serves /v1/audio/generations, not speech", true
		}
	case SurfaceMusic:
		if c.Backend(model.ModalitySpeech) != model.BackendMusic3 {
			return "speech backend is not music3", true
		}
	case SurfaceTranscribe:
		// TranscriptionHandler runs whisper for modality_backends.transcribe=whisper and
		// otherwise falls through to ChatHandler, which needs an audio-capable chat model.
		if c.Backend(model.ModalityTranscribe) != model.BackendWhisper &&
			!c.Has(model.CapabilityCompletion, model.CapabilityAudio) {
			return "neither a whisper transcribe backend nor an audio-capable chat model", true
		}
	case SurfaceEmbed, SurfaceRerank:
		if slices.Contains(c.Capabilities, model.CapabilityDecision) {
			return "decision model cannot serve " + string(s.Surface), true
		}
	}
	return "", false
}

// Admits reports whether c may serve the request (surface class + required capabilities),
// with the reason when it may not. Context fit and availability are not checked — this is
// the gate LA11 routers apply to the candidate their classifier picked.
func (s Signals) Admits(c Candidate) (string, bool) {
	if missing := missingCaps(c, s.Required()); len(missing) > 0 {
		return "missing capability: " + strings.Join(missing, ", "), false
	}
	if reason, bad := s.surfaceMismatch(c); bad {
		return reason, false
	}
	if s.Check != nil {
		if reason := s.Check(c); reason != "" {
			return reason, false
		}
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
