package voicepipe

import "sync"

// Duplex pipeline stages forwarded as response.status (and a few VAD-named
// events). Clients use these for logs and metrics without parsing audio.
const (
	StageSilenceStarted = "silence_started"
	StageTranscribing   = "transcribing"
	StageGenerating     = "generating"
	StageFirstToken     = "first_token"
	StageAudioDone      = "audio_done"
)

// StageReporter lets the Go chat path (outside the session) emit the same
// response.status events the socket owns. Serve wires Set to the live session.
type StageReporter struct {
	mu sync.Mutex
	fn func(status string, extra map[string]any)
}

// Set replaces the sink. Pass nil to detach.
func (r *StageReporter) Set(fn func(status string, extra map[string]any)) {
	if r == nil {
		return
	}
	r.mu.Lock()
	r.fn = fn
	r.mu.Unlock()
}

// Report sends one stage. Extra keys are merged onto the socket event.
func (r *StageReporter) Report(status string, extra map[string]any) {
	if r == nil || status == "" {
		return
	}
	r.mu.Lock()
	fn := r.fn
	r.mu.Unlock()
	if fn != nil {
		fn(status, extra)
	}
}
