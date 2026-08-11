package server

import (
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// Empty generation / host stability classification (wishlist item 8).
//
// Why: agents that map empty text → "model refused" poison Inventory; runner deaths
// that look like semanticOk fail blame the wrong layer. Classify at the host so
// clients can set kindExt.hostUnstable only for infra causes.
const (
	doneReasonEmptyGeneration = "empty_generation"
	causeHostUnstable         = "host_unstable"
)

type emptyGenClass int

const (
	emptyGenOK emptyGenClass = iota
	emptyGenEmpty
	emptyGenUnstable
)

type emptyGenInput struct {
	Response    string
	Thinking    string
	EvalCount   int
	NumPredict  int
	LoadDone    bool
	StreamError string
	DoneReason  string
}

func classifyEmptyGeneration(in emptyGenInput) emptyGenClass {
	errMsg := strings.ToLower(strings.TrimSpace(in.StreamError))
	if errMsg != "" {
		if isHostUnstableError(errMsg) {
			return emptyGenUnstable
		}
		return emptyGenOK // other errors handled elsewhere
	}
	// Thinking-only is valid output for thinking models — not empty_generation.
	if strings.TrimSpace(in.Thinking) != "" {
		return emptyGenOK
	}
	if in.EvalCount > 0 {
		return emptyGenOK
	}
	if strings.TrimSpace(in.Response) != "" {
		return emptyGenOK
	}
	// Empty response+thinking, zero eval.
	if !in.LoadDone {
		return emptyGenOK
	}
	if in.NumPredict >= 0 && in.NumPredict <= 1 {
		return emptyGenEmpty
	}
	return emptyGenEmpty
}

// isHostUnstableError uses tight needles only.
// Why: broad substrings ("llama server", "subprocess") false-positive on config errors.
func isHostUnstableError(errMsg string) bool {
	needles := []string{
		"runner exited",
		"runner crashed",
		"process exited",
		"signal: killed",
		"broken pipe",
		"connection reset by peer",
		"llama-server exited",
		"llama server died",
		"subprocess exited",
	}
	for _, n := range needles {
		if strings.Contains(errMsg, n) {
			return true
		}
	}
	return false
}

// inferenceErrorExtra is optional timing/cause fields merged into error JSON.
// Why: success responses already expose Metrics; bare {"error"} left agents blind on fail paths.
type inferenceErrorExtra struct {
	Cause            string
	PreemptedReason  string
	TotalDuration    time.Duration
	LoadDuration     time.Duration
	TimeToFirstToken time.Duration
	HasTTFT          bool
	RetryAfter       int
}

func inferenceErrorMap(errMsg string, status int, extra inferenceErrorExtra) map[string]any {
	out := map[string]any{"error": errMsg}
	if status != 0 {
		out["status"] = status
	}
	if extra.Cause != "" {
		out["cause"] = extra.Cause
	}
	if extra.PreemptedReason != "" {
		out["preempted_reason"] = extra.PreemptedReason
	}
	if extra.RetryAfter > 0 {
		out["retry_after"] = extra.RetryAfter
	}
	if extra.TotalDuration > 0 {
		out["total_duration"] = extra.TotalDuration.Nanoseconds()
	}
	if extra.LoadDuration > 0 {
		out["load_duration"] = extra.LoadDuration.Nanoseconds()
	}
	if extra.HasTTFT && extra.TimeToFirstToken > 0 {
		out["time_to_first_token"] = extra.TimeToFirstToken.Nanoseconds()
	}
	return out
}

// defaultBusyRetryAfterSec is the Retry-After value on busy 503s (header + JSON).
// Why constant 2s: stable client contract in Phase A; avoid inventing queue-depth→delay math.
const defaultBusyRetryAfterSec = 2

func applyEmptyGenClassifyGenerate(res *api.GenerateResponse, numPredict int, loadDone bool) {
	if res == nil || !res.Done {
		return
	}
	class := classifyEmptyGeneration(emptyGenInput{
		Response:   res.Response,
		Thinking:   res.Thinking,
		EvalCount:  res.EvalCount,
		NumPredict: numPredict,
		LoadDone:   loadDone,
		DoneReason: res.DoneReason,
	})
	switch class {
	case emptyGenEmpty:
		res.DoneReason = doneReasonEmptyGeneration
		metricsIncRequestResult("empty_generation")
	case emptyGenUnstable:
		res.DoneReason = causeHostUnstable
		metricsIncRequestResult("host_unstable")
	case emptyGenOK:
		if res.DoneReason != "" && res.DoneReason != doneReasonEmptyGeneration {
			metricsIncRequestResult("ok")
		}
	}
}

func applyEmptyGenClassifyChat(res *api.ChatResponse, numPredict int, loadDone bool) {
	if res == nil || !res.Done {
		return
	}
	class := classifyEmptyGeneration(emptyGenInput{
		Response:   res.Message.Content,
		Thinking:   res.Message.Thinking,
		EvalCount:  res.EvalCount,
		NumPredict: numPredict,
		LoadDone:   loadDone,
		DoneReason: res.DoneReason,
	})
	switch class {
	case emptyGenEmpty:
		res.DoneReason = doneReasonEmptyGeneration
		metricsIncRequestResult("empty_generation")
	case emptyGenUnstable:
		res.DoneReason = causeHostUnstable
		metricsIncRequestResult("host_unstable")
	case emptyGenOK:
		if res.DoneReason != "" && res.DoneReason != doneReasonEmptyGeneration {
			metricsIncRequestResult("ok")
		}
	}
}

func errorExtraFromCheckpoints(start, loaded, firstTokenAt time.Time, sawFirstToken bool) inferenceErrorExtra {
	extra := inferenceErrorExtra{}
	if !start.IsZero() {
		extra.TotalDuration = time.Since(start)
	}
	if !loaded.IsZero() && !start.IsZero() {
		extra.LoadDuration = loaded.Sub(start)
	}
	if sawFirstToken && !firstTokenAt.IsZero() && !start.IsZero() {
		extra.TimeToFirstToken = firstTokenAt.Sub(start)
		extra.HasTTFT = true
	}
	return extra
}
