package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// testArgs creates ToolCallFunctionArguments from a map (convenience function for tests)
func testArgs(m map[string]any) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	for k, v := range m {
		args.Set(k, v)
	}
	return args
}

// argsComparer provides cmp options for comparing ToolCallFunctionArguments by value
var argsComparer = cmp.Comparer(func(a, b api.ToolCallFunctionArguments) bool {
	return cmp.Equal(a.ToMap(), b.ToMap())
})

const (
	prefix = `data:image/jpeg;base64,`
	image  = `iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=`
)

func TestFromChatRequest_Basic(t *testing.T) {
	req := ChatCompletionRequest{
		Model: "test-model",
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
	}

	result, err := FromChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Model != "test-model" {
		t.Errorf("expected model 'test-model', got %q", result.Model)
	}

	if len(result.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result.Messages))
	}

	if result.Messages[0].Role != "user" || result.Messages[0].Content != "Hello" {
		t.Errorf("unexpected message: %+v", result.Messages[0])
	}
}

func TestFromCompleteRequest_NGreaterThanOne(t *testing.T) {
	n := 2
	_, err := FromCompleteRequest(CompletionRequest{Model: "m", Prompt: "hi", N: &n})
	if err == nil || !strings.Contains(err.Error(), "n=2") {
		t.Fatalf("err = %v", err)
	}
}

func TestFromCompleteRequest_BestOfGreaterThanOne(t *testing.T) {
	n := 2
	_, err := FromCompleteRequest(CompletionRequest{Model: "m", Prompt: "hi", BestOf: &n})
	if err == nil || !strings.Contains(err.Error(), "best_of=2") {
		t.Fatalf("err = %v", err)
	}
}

func TestFromChatRequest_StoreTrue(t *testing.T) {
	on := true
	_, err := FromChatRequest(ChatCompletionRequest{
		Model:    "m",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Store:    &on,
	})
	if err == nil || !strings.Contains(err.Error(), "store:true") {
		t.Fatalf("err = %v", err)
	}
	off := false
	if _, err := FromChatRequest(ChatCompletionRequest{
		Model:    "m",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Store:    &off,
	}); err != nil {
		t.Fatal(err)
	}
}

func TestFromCompleteRequest_ServiceTierFlex(t *testing.T) {
	_, err := FromCompleteRequest(CompletionRequest{Model: "m", Prompt: "hi", ServiceTier: "flex"})
	if err == nil || !strings.Contains(err.Error(), "service_tier") {
		t.Fatalf("err = %v", err)
	}
}

func TestFromChatRequest_WithImage(t *testing.T) {
	imgData, _ := base64.StdEncoding.DecodeString(image)

	req := ChatCompletionRequest{
		Model: "test-model",
		Messages: []Message{
			{
				Role: "user",
				Content: []any{
					map[string]any{"type": "text", "text": "Hello"},
					map[string]any{
						"type":      "image_url",
						"image_url": map[string]any{"url": prefix + image},
					},
				},
			},
		},
	}

	result, err := FromChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result.Messages))
	}

	if result.Messages[0].Content != "Hello" {
		t.Errorf("expected message content 'Hello', got %q", result.Messages[0].Content)
	}

	if len(result.Messages[0].Images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(result.Messages[0].Images))
	}

	if string(result.Messages[0].Images[0]) != string(imgData) {
		t.Error("image data mismatch")
	}
}

func TestFromChatRequest_JoinsTextParts(t *testing.T) {
	req := ChatCompletionRequest{
		Model: "test-model",
		Messages: []Message{
			{
				Role: "user",
				Content: []any{
					map[string]any{"type": "text", "text": "plan:"},
					map[string]any{"type": "text", "text": " do the thing"},
				},
			},
		},
	}
	result, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Messages[0].Content != "plan: do the thing" {
		t.Fatalf("joined content = %q", result.Messages[0].Content)
	}
	if _, ok := result.Options["temperature"]; ok {
		t.Fatal("omitted temperature must not be injected so generation_config can apply")
	}
}

func TestFromChatRequest_WithVideoURL_MergeOrder(t *testing.T) {
	imgData, _ := base64.StdEncoding.DecodeString(image)
	vidBytes := []byte{0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70} // minimal ftyp-like prefix for parser
	vidB64 := base64.StdEncoding.EncodeToString(vidBytes)

	req := ChatCompletionRequest{
		Model: "test-model",
		Messages: []Message{
			{
				Role: "user",
				Content: []any{
					map[string]any{"type": "text", "text": "look"},
					map[string]any{
						"type":      "image_url",
						"image_url": map[string]any{"url": prefix + image},
					},
					map[string]any{
						"type":      "video_url",
						"video_url": map[string]any{"url": "data:video/mp4;base64," + vidB64},
					},
				},
			},
		},
	}

	result, err := FromChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result.Messages))
	}
	if result.Messages[0].Content != "look" {
		t.Fatalf("content: got %q", result.Messages[0].Content)
	}
	if len(result.Messages[0].Images) != 1 || string(result.Messages[0].Images[0]) != string(imgData) {
		t.Fatalf("expected one image before video expansion")
	}
	if len(result.Messages[0].Videos) != 1 || string(result.Messages[0].Videos[0]) != string(vidBytes) {
		t.Fatalf("video payload mismatch")
	}
}

func TestFromChatRequest_WithInputAudio(t *testing.T) {
	audioBytes := []byte{0x52, 0x49, 0x46, 0x46} // RIFF
	audioB64 := base64.StdEncoding.EncodeToString(audioBytes)
	req := ChatCompletionRequest{
		Model: "test-model",
		Messages: []Message{
			{
				Role: "user",
				Content: []any{
					map[string]any{"type": "text", "text": "listen"},
					map[string]any{
						"type": "input_audio",
						"input_audio": map[string]any{
							"data":   audioB64,
							"format": "wav",
						},
					},
				},
			},
		},
	}
	result, err := FromChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result.Messages))
	}
	if len(result.Messages[0].Images) != 0 {
		t.Fatalf("audio should not be in Images, got %d", len(result.Messages[0].Images))
	}
	if len(result.Messages[0].AudioClips) != 1 || string(result.Messages[0].AudioClips[0]) != string(audioBytes) {
		t.Fatalf("audio clip mismatch")
	}
}

func TestChatCompletionRequestHasVideoURL(t *testing.T) {
	req := ChatCompletionRequest{
		Model: "m",
		Messages: []Message{
			{Role: "user", Content: []any{map[string]any{"type": "text", "text": "hi"}}},
		},
	}
	if ChatCompletionRequestHasVideoURL(&req) {
		t.Fatal("expected false")
	}
	req.Messages[0].Content = []any{map[string]any{"type": "video_url", "video_url": map[string]any{"url": "data:video/mp4;base64,AAAA"}}}
	if !ChatCompletionRequestHasVideoURL(&req) {
		t.Fatal("expected true")
	}
}

func TestFromChatRequest_PromptCacheKeyAndOptions(t *testing.T) {
	key := "agent-thread-42"
	salt := "tenant-9"
	enablePrefix := true
	req := ChatCompletionRequest{
		Model: "m",
		Messages: []Message{
			{Role: "user", Content: "hi"},
		},
		PromptCacheKey:      &key,
		CacheSalt:           &salt,
		EnablePrefixMMCache: &enablePrefix,
		Options: map[string]any{
			"num_ctx": float64(8192),
		},
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if out.Options["prompt_cache_key"] != key {
		t.Fatalf("prompt_cache_key=%v", out.Options["prompt_cache_key"])
	}
	if out.Options["cache_salt"] != salt {
		t.Fatalf("cache_salt=%v", out.Options["cache_salt"])
	}
	if out.Options["enable_prefix_mm_cache"] != true {
		t.Fatalf("enable_prefix_mm_cache=%v", out.Options["enable_prefix_mm_cache"])
	}
	if out.Options["num_ctx"] != float64(8192) {
		t.Fatalf("num_ctx=%v", out.Options["num_ctx"])
	}
}

func TestFromChatRequest_SessionIDAliasesPromptCacheKey(t *testing.T) {
	sid := "sglang-session-9"
	req := ChatCompletionRequest{
		Model:     "m",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		SessionID: &sid,
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if out.Options["prompt_cache_key"] != sid {
		t.Fatalf("prompt_cache_key=%v want session_id alias", out.Options["prompt_cache_key"])
	}
}

func TestFromChatRequest_PromptCacheKeyWinsOverSessionID(t *testing.T) {
	key := "explicit-cache"
	sid := "sglang-session"
	req := ChatCompletionRequest{
		Model:          "m",
		Messages:       []Message{{Role: "user", Content: "hi"}},
		PromptCacheKey: &key,
		SessionID:      &sid,
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if out.Options["prompt_cache_key"] != key {
		t.Fatalf("prompt_cache_key=%v want %q", out.Options["prompt_cache_key"], key)
	}
}

func TestFromChatRequest_KeepAlive(t *testing.T) {
	ka := api.Duration{Duration: 30 * time.Minute}
	req := ChatCompletionRequest{
		Model:     "gemma4:26b-optiq",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		KeepAlive: &ka,
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if out.KeepAlive == nil || out.KeepAlive.Duration != ka.Duration {
		t.Fatalf("KeepAlive=%v want 30m", out.KeepAlive)
	}

	req2 := ChatCompletionRequest{
		Model:    "gemma4:26b-optiq",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Options:  map[string]any{"keep_alive": "45m"},
	}
	out2, err := FromChatRequest(req2)
	if err != nil {
		t.Fatal(err)
	}
	if out2.KeepAlive == nil || out2.KeepAlive.Duration != 45*time.Minute {
		t.Fatalf("KeepAlive from options=%v want 45m", out2.KeepAlive)
	}
	if _, ok := out2.Options["keep_alive"]; ok {
		t.Fatal("keep_alive should be lifted out of options")
	}
}

func TestFromChatRequest_TopLevelThink(t *testing.T) {
	// WHY: Hermes sends "think" on /v1; previously passthrough-only → silent drop.
	body := []byte(`{
		"model": "qwen3-coder-next",
		"messages": [{"role":"user","content":"hi"}],
		"think": true
	}`)
	req, err := BindChatCompletionRequest(body)
	if err != nil {
		t.Fatal(err)
	}
	if req.Think == nil || !req.Think.Bool() {
		t.Fatalf("Think=%v want true", req.Think)
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if out.Think == nil || !out.Think.Bool() {
		t.Fatalf("mapped Think=%v want true", out.Think)
	}
	if out.ThinkFromAlias {
		t.Fatal("explicit think must not set ThinkFromAlias")
	}

	// Explicit think wins over enable_thinking.
	body2 := []byte(`{
		"model": "qwen3-coder-next",
		"messages": [{"role":"user","content":"hi"}],
		"think": false,
		"enable_thinking": true
	}`)
	req2, err := BindChatCompletionRequest(body2)
	if err != nil {
		t.Fatal(err)
	}
	out2, err := FromChatRequest(req2)
	if err != nil {
		t.Fatal(err)
	}
	if out2.Think == nil || out2.Think.Bool() {
		t.Fatalf("think must win over enable_thinking: %v", out2.Think)
	}
}

func TestDecodeVideoURL_RejectsLoopback(t *testing.T) {
	t.Setenv("OLLAMA_VIDEO_ALLOW_INSECURE_HTTP", "1")
	_, err := decodeVideoURL(context.Background(), "http://127.0.0.1:8080/v.mp4")
	if err == nil || !strings.Contains(err.Error(), "internal") {
		t.Fatalf("expected internal error, got %v", err)
	}
}

func TestDecodeVideoURL_HTTPRequiresOptIn(t *testing.T) {
	t.Setenv("OLLAMA_VIDEO_ALLOW_INSECURE_HTTP", "")
	_, err := decodeVideoURL(context.Background(), "http://example.com/clip.mp4")
	if err == nil || !strings.Contains(err.Error(), "https") {
		t.Fatalf("expected https/opt-in error, got %v", err)
	}
}

func TestCheckMediaHostAllowed(t *testing.T) {
	t.Setenv("OLLAMA_MEDIA_ALLOWED_HOSTS", "")
	if err := checkMediaHostAllowed("cdn.example.com"); err != nil {
		t.Fatalf("empty allowlist should permit: %v", err)
	}

	t.Setenv("OLLAMA_MEDIA_ALLOWED_HOSTS", "cdn.example.com, media.corp")
	if err := checkMediaHostAllowed("cdn.example.com"); err != nil {
		t.Fatalf("exact host: %v", err)
	}
	if err := checkMediaHostAllowed("edge.cdn.example.com"); err != nil {
		t.Fatalf("subdomain of allowlisted apex: %v", err)
	}
	if err := checkMediaHostAllowed("evil.com"); err == nil {
		t.Fatal("expected allowlist reject")
	}
}

func TestCheckRemoteMediaURL_RedirectTargetPrivate(t *testing.T) {
	t.Setenv("OLLAMA_VIDEO_ALLOW_INSECURE_HTTP", "1")
	t.Setenv("OLLAMA_MEDIA_ALLOWED_HOSTS", "")
	u, err := url.Parse("http://127.0.0.1/clip.mp4")
	if err != nil {
		t.Fatal(err)
	}
	if err := checkRemoteMediaURL(u); err == nil || !strings.Contains(err.Error(), "internal") {
		t.Fatalf("redirect to loopback must fail: %v", err)
	}
}

func TestFetchVideoURL_AllowlistAndSize(t *testing.T) {
	t.Setenv("OLLAMA_VIDEO_ALLOW_INSECURE_HTTP", "1")
	t.Setenv("OLLAMA_MEDIA_ALLOWED_HOSTS", "127.0.0.1") // still blocked by SSRF before allowlist helps
	_, err := decodeVideoURL(context.Background(), "http://127.0.0.1:9/v.mp4")
	if err == nil || !strings.Contains(err.Error(), "internal") {
		t.Fatalf("SSRF must win even when host is allowlisted: %v", err)
	}

	t.Setenv("OLLAMA_MEDIA_ALLOWED_HOSTS", "allowed.example")
	u, err := url.Parse("https://other.example/v.mp4")
	if err != nil {
		t.Fatal(err)
	}
	if err := checkMediaHostAllowed(u.Hostname()); err == nil {
		t.Fatal("want allowlist miss")
	}
}

func TestFromChatRequest_ToolMultipartKeepsMetadata(t *testing.T) {
	// SGLang #33898: tool role with image parts must keep tool_call_id without ToolCalls.
	req := ChatCompletionRequest{
		Model: "qwen3-vl",
		Messages: []Message{{
			Role:       "tool",
			ToolCallID: "call_1",
			Name:       "see",
			Content: []any{
				map[string]any{"type": "text", "text": "shot"},
				map[string]any{"type": "image_url", "image_url": map[string]any{
					"url": prefix + image,
				}},
			},
		}},
	}
	out, err := FromChatRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Messages) != 1 {
		t.Fatalf("messages=%d", len(out.Messages))
	}
	m := out.Messages[0]
	if m.ToolCallID != "call_1" || m.ToolName != "see" {
		t.Fatalf("tool meta ToolCallID=%q ToolName=%q", m.ToolCallID, m.ToolName)
	}
	if len(m.Images) != 1 {
		t.Fatalf("images=%d", len(m.Images))
	}
}

func TestFromCompleteRequest_Basic(t *testing.T) {
	temp := float32(0.8)
	req := CompletionRequest{
		Model:       "test-model",
		Prompt:      "Hello",
		Temperature: &temp,
	}

	result, err := FromCompleteRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Model != "test-model" {
		t.Errorf("expected model 'test-model', got %q", result.Model)
	}

	if result.Prompt != "Hello" {
		t.Errorf("expected prompt 'Hello', got %q", result.Prompt)
	}

	if tempVal, ok := result.Options["temperature"].(float64); !ok || tempVal < 0.799 || tempVal > 0.801 {
		t.Errorf("expected temperature 0.8, got %v", result.Options["temperature"])
	}
}

func TestFromCompleteRequest_OmittedSamplingDoesNotInject(t *testing.T) {
	result, err := FromCompleteRequest(CompletionRequest{
		Model:  "test-model",
		Prompt: "Hello",
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"temperature", "top_p", "top_k", "min_p", "typical_p", "frequency_penalty", "presence_penalty", "repeat_penalty"} {
		if _, ok := result.Options[key]; ok {
			t.Fatalf("omitted %s must not be injected so generation_config can apply", key)
		}
	}
}

func TestFromChatRequest_TopKAndRepetitionPenalty(t *testing.T) {
	k := 40
	rp := 1.1
	result, err := FromChatRequest(ChatCompletionRequest{
		Model: "test-model",
		Messages: []Message{
			{Role: "user", Content: "hi"},
		},
		TopK:              &k,
		RepetitionPenalty: &rp,
	})
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := result.Options["top_k"].(int); !ok || v != 40 {
		t.Fatalf("top_k = %v", result.Options["top_k"])
	}
	if v, ok := result.Options["repeat_penalty"].(float64); !ok || v != 1.1 {
		t.Fatalf("repeat_penalty = %v", result.Options["repeat_penalty"])
	}
}

func TestFromChatRequest_LogitBias(t *testing.T) {
	result, err := FromChatRequest(ChatCompletionRequest{
		Model: "test-model",
		Messages: []Message{
			{Role: "user", Content: "hi"},
		},
		LogitBias: map[string]float64{"13": -100},
	})
	if err != nil {
		t.Fatal(err)
	}
	got, ok := result.Options["logit_bias"].(map[int32]float32)
	if !ok || got[13] != -100 {
		t.Fatalf("logit_bias = %v", result.Options["logit_bias"])
	}
}

func TestFromCompleteRequest_ExplicitZeroTopP(t *testing.T) {
	zero := float32(0)
	result, err := FromCompleteRequest(CompletionRequest{
		Model:  "test-model",
		Prompt: "Hello",
		TopP:   &zero,
	})
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := result.Options["top_p"].(float64); !ok || v != 0 {
		t.Fatalf("top_p = %v want 0", result.Options["top_p"])
	}
}

func TestFromCompleteRequest_TopKAndRepetitionPenalty(t *testing.T) {
	k := 20
	rp := float32(1.05)
	result, err := FromCompleteRequest(CompletionRequest{
		Model:             "test-model",
		Prompt:            "Hello",
		TopK:              &k,
		RepetitionPenalty: &rp,
	})
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := result.Options["top_k"].(int); !ok || v != 20 {
		t.Fatalf("top_k = %v", result.Options["top_k"])
	}
	if v, ok := result.Options["repeat_penalty"].(float64); !ok || v < 1.049 || v > 1.051 {
		t.Fatalf("repeat_penalty = %v", result.Options["repeat_penalty"])
	}
}

func TestFromChatRequest_MaxCompletionTokensAlias(t *testing.T) {
	n := 77
	result, err := FromChatRequest(ChatCompletionRequest{
		Model:               "test-model",
		Messages:            []Message{{Role: "user", Content: "hi"}},
		MaxCompletionTokens: &n,
	})
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := result.Options["num_predict"].(int); !ok || v != 77 {
		t.Fatalf("num_predict = %v", result.Options["num_predict"])
	}
	if !strings.Contains(result.Messages[0].Content, api.OutputBudgetGuidance) {
		t.Fatalf("tight max_completion_tokens should hint, got %q", result.Messages[0].Content)
	}
}

func TestFromChatRequest_MaxTokensWinsOverCompletionTokens(t *testing.T) {
	maxTok, maxComp := 10, 99
	result, err := FromChatRequest(ChatCompletionRequest{
		Model:               "test-model",
		Messages:            []Message{{Role: "user", Content: "hi"}},
		MaxTokens:           &maxTok,
		MaxCompletionTokens: &maxComp,
	})
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := result.Options["num_predict"].(int); !ok || v != 10 {
		t.Fatalf("num_predict = %v want 10", result.Options["num_predict"])
	}
}

func TestFromChatRequest_OutputBudgetRoom(t *testing.T) {
	n := api.OutputBudgetTightThreshold
	result, err := FromChatRequest(ChatCompletionRequest{
		Model:     "test-model",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: &n,
	})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(result.Messages[0].Content, api.OutputBudgetGuidance) {
		t.Fatalf("budget at threshold must not hint, got %q", result.Messages[0].Content)
	}
}

func TestFromChatRequest_MinPTypicalPRepeatPenalty(t *testing.T) {
	minP, typ, rp := 0.05, 0.9, 1.15
	k := 0
	result, err := FromChatRequest(ChatCompletionRequest{
		Model:         "test-model",
		Messages:      []Message{{Role: "user", Content: "hi"}},
		MinP:          &minP,
		TypicalP:      &typ,
		RepeatPenalty: &rp,
		TopK:          &k,
	})
	if err != nil {
		t.Fatal(err)
	}
	if v, ok := result.Options["min_p"].(float64); !ok || v != 0.05 {
		t.Fatalf("min_p = %v", result.Options["min_p"])
	}
	if v, ok := result.Options["typical_p"].(float64); !ok || v != 0.9 {
		t.Fatalf("typical_p = %v", result.Options["typical_p"])
	}
	if v, ok := result.Options["repeat_penalty"].(float64); !ok || v != 1.15 {
		t.Fatalf("repeat_penalty = %v", result.Options["repeat_penalty"])
	}
	if v, ok := result.Options["top_k"].(int); !ok || v != 0 {
		t.Fatalf("top_k = %v want explicit 0", result.Options["top_k"])
	}
}

func TestFromChatRequest_EnablePLD(t *testing.T) {
	off := false
	result, err := FromChatRequest(ChatCompletionRequest{
		Model:     "test-model",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		EnablePLD: &off,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.EnablePLD == nil || *result.EnablePLD {
		t.Fatalf("EnablePLD = %v", result.EnablePLD)
	}
	if v, ok := result.Options["enable_pld"].(bool); !ok || v {
		t.Fatalf("options enable_pld = %v", result.Options["enable_pld"])
	}
}

func TestFromChatRequest_ParallelToolCallsFalse(t *testing.T) {
	off := false
	result, err := FromChatRequest(ChatCompletionRequest{
		Model:             "test-model",
		Messages:          []Message{{Role: "user", Content: "hi"}},
		ParallelToolCalls: &off,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.ParallelToolCalls == nil || *result.ParallelToolCalls {
		t.Fatalf("ParallelToolCalls = %v", result.ParallelToolCalls)
	}
}

func TestChatFinishReasonLengthWinsOverTools(t *testing.T) {
	got := chatFinishReason("length", true)
	if got == nil || *got != "length" {
		t.Fatalf("got %v", got)
	}
	got = chatFinishReason("stop", true)
	if got == nil || *got != finishReasonToolCalls {
		t.Fatalf("got %v", got)
	}
	if chatFinishReason("", true) != nil {
		t.Fatal("empty done_reason stays nil on stream deltas")
	}
}

func TestToUsage(t *testing.T) {
	resp := api.ChatResponse{
		Metrics: api.Metrics{
			PromptEvalCount: 10,
			EvalCount:       20,
		},
	}

	usage := ToUsage(resp)

	if usage.PromptTokens != 10 {
		t.Errorf("expected PromptTokens 10, got %d", usage.PromptTokens)
	}

	if usage.CompletionTokens != 20 {
		t.Errorf("expected CompletionTokens 20, got %d", usage.CompletionTokens)
	}

	if usage.TotalTokens != 30 {
		t.Errorf("expected TotalTokens 30, got %d", usage.TotalTokens)
	}
	if usage.Compression != nil {
		t.Fatal("expected nil compression_meta")
	}
}

func TestToUsage_compressionMeta(t *testing.T) {
	resp := api.ChatResponse{
		Metrics:     api.Metrics{PromptEvalCount: 10, EvalCount: 2},
		Compression: &api.ChatCompressionMeta{Mode: "placeholder", ElideFrom: 4},
	}
	usage := ToUsage(resp)
	if usage.Compression == nil || usage.Compression.Mode != "placeholder" || usage.Compression.ElideFrom != 4 {
		t.Fatalf("%+v", usage.Compression)
	}
}

func TestToUsage_multimodalPromptTokensDetails(t *testing.T) {
	resp := api.ChatResponse{
		Metrics: api.Metrics{
			PromptEvalCount: 100,
			EvalCount:       5,
			ImageTokens:     768,
			VideoTokens:     2304,
		},
	}
	usage := ToUsage(resp)
	if usage.PromptTokensDetails == nil {
		t.Fatal("expected prompt_tokens_details")
	}
	if usage.PromptTokensDetails.ImageTokens == nil || *usage.PromptTokensDetails.ImageTokens != 768 {
		t.Fatalf("image_tokens=%v, want 768", usage.PromptTokensDetails.ImageTokens)
	}
	if usage.PromptTokensDetails.VideoTokens == nil || *usage.PromptTokensDetails.VideoTokens != 2304 {
		t.Fatalf("video_tokens=%v, want 2304", usage.PromptTokensDetails.VideoTokens)
	}
	if usage.PromptTokensDetails.AudioTokens != nil {
		t.Fatal("expected no audio_tokens")
	}
}

func TestToUsage_cachedPromptTokens(t *testing.T) {
	resp := api.ChatResponse{
		Metrics: api.Metrics{
			PromptEvalCount:    200,
			EvalCount:          10,
			CachedPromptTokens: 150,
		},
	}
	usage := ToUsage(resp)
	if usage.PromptTokensDetails == nil || usage.PromptTokensDetails.CachedTokens == nil {
		t.Fatal("expected cached_tokens")
	}
	if *usage.PromptTokensDetails.CachedTokens != 150 {
		t.Fatalf("cached_tokens=%d, want 150", *usage.PromptTokensDetails.CachedTokens)
	}
}

func TestToUsage_alwaysCachedTokens(t *testing.T) {
	usage := ToUsage(api.ChatResponse{Metrics: api.Metrics{PromptEvalCount: 10, EvalCount: 2}})
	if usage.PromptTokensDetails == nil || usage.PromptTokensDetails.CachedTokens == nil {
		t.Fatal("mlx-serve always emits cached_tokens")
	}
	if *usage.PromptTokensDetails.CachedTokens != 0 {
		t.Fatalf("cached_tokens=%d want 0", *usage.PromptTokensDetails.CachedTokens)
	}
}

func TestToUsage_createdCacheTokens(t *testing.T) {
	resp := api.ChatResponse{
		Metrics: api.Metrics{
			PromptEvalCount:     200,
			CachedPromptTokens:  50,
			CacheCreationTokens: 150,
		},
	}
	usage := ToUsage(resp)
	if usage.PromptTokensDetails == nil || usage.PromptTokensDetails.CreatedCacheTokens == nil {
		t.Fatal("expected created_cache_tokens")
	}
	if *usage.PromptTokensDetails.CreatedCacheTokens != 150 {
		t.Fatalf("created_cache_tokens=%d, want 150", *usage.PromptTokensDetails.CreatedCacheTokens)
	}
}

func TestSglExtFromMetrics(t *testing.T) {
	ext := SglExtFromMetrics(api.Metrics{
		CachedPromptTokens:         100,
		CachedTokensHost:           40,
		CachedTokensStorage:        10,
		CachedTokensStorageBackend: "file",
	})
	if ext == nil || ext.CachedTokensDetails == nil {
		t.Fatal("expected sglext")
	}
	if ext.CachedTokensDetails.Device != 100 || ext.CachedTokensDetails.Host != 40 {
		t.Fatalf("device/host=%d/%d", ext.CachedTokensDetails.Device, ext.CachedTokensDetails.Host)
	}
	if ext.CachedTokensDetails.Storage == nil || *ext.CachedTokensDetails.Storage != 10 {
		t.Fatal("expected storage")
	}
	if ext.CachedTokensDetails.StorageBackend == nil || *ext.CachedTokensDetails.StorageBackend != "file" {
		t.Fatal("expected storage_backend")
	}
	b, err := json.Marshal(ext)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"cached_tokens_details"`) || !strings.Contains(string(b), `"device":100`) {
		t.Fatalf("json=%s", b)
	}
}

func TestSglExtFromMetrics_nilWhenEmpty(t *testing.T) {
	if SglExtFromMetrics(api.Metrics{}) != nil {
		t.Fatal("expected nil sglext for empty metrics")
	}
}

func TestToUsageGenerate_multimodalPromptTokensDetails(t *testing.T) {
	resp := api.GenerateResponse{
		Metrics: api.Metrics{
			PromptEvalCount: 50,
			EvalCount:       10,
			VideoTokens:     1536,
		},
	}
	usage := ToUsageGenerate(resp)
	if usage.PromptTokensDetails == nil || usage.PromptTokensDetails.VideoTokens == nil || *usage.PromptTokensDetails.VideoTokens != 1536 {
		t.Fatalf("video_tokens=%v, want 1536", usage.PromptTokensDetails)
	}
}

func TestNewError(t *testing.T) {
	tests := []struct {
		code int
		want string
	}{
		{400, "invalid_request_error"},
		{404, "not_found_error"},
		{500, "api_error"},
	}

	for _, tt := range tests {
		result := NewError(tt.code, "test message")
		if result.Error.Type != tt.want {
			t.Errorf("NewError(%d) type = %q, want %q", tt.code, result.Error.Type, tt.want)
		}
		if result.Error.Message != "test message" {
			t.Errorf("NewError(%d) message = %q, want %q", tt.code, result.Error.Message, "test message")
		}
	}
}

func TestToToolCallsPreservesIDs(t *testing.T) {
	original := []api.ToolCall{
		{
			ID: "call_abc123",
			Function: api.ToolCallFunction{
				Index: 2,
				Name:  "get_weather",
				Arguments: testArgs(map[string]any{
					"location": "Seattle",
				}),
			},
		},
		{
			ID: "call_def456",
			Function: api.ToolCallFunction{
				Index: 7,
				Name:  "get_time",
				Arguments: testArgs(map[string]any{
					"timezone": "UTC",
				}),
			},
		},
	}

	toolCalls := make([]api.ToolCall, len(original))
	copy(toolCalls, original)
	got := ToToolCalls(toolCalls)

	expected := []ToolCall{
		{
			ID:    "call_abc123",
			Type:  "function",
			Index: 2,
			Function: struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			}{
				Name:      "get_weather",
				Arguments: `{"location":"Seattle"}`,
			},
		},
		{
			ID:    "call_def456",
			Type:  "function",
			Index: 7,
			Function: struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			}{
				Name:      "get_time",
				Arguments: `{"timezone":"UTC"}`,
			},
		},
	}

	if diff := cmp.Diff(expected, got); diff != "" {
		t.Errorf("tool calls mismatch (-want +got):\n%s", diff)
	}

	if diff := cmp.Diff(original, toolCalls, argsComparer); diff != "" {
		t.Errorf("input tool calls mutated (-want +got):\n%s", diff)
	}
}

func TestToToolCallsFillsEmptyIDs(t *testing.T) {
	got := ToToolCalls([]api.ToolCall{{
		Function: api.ToolCallFunction{Name: "ping"},
	}})
	if len(got) != 1 || got[0].ID != "call_0" {
		t.Fatalf("got %+v", got)
	}
}

func TestToToolCallsEmptyArgumentsAreObject(t *testing.T) {
	got := ToToolCalls([]api.ToolCall{{
		ID:       "call_1",
		Function: api.ToolCallFunction{Name: "ping"},
	}})
	if len(got) != 1 || got[0].Function.Arguments != "{}" {
		t.Fatalf("got %+v", got)
	}
}

func TestFromCompletionToolCallEmptyArguments(t *testing.T) {
	got, err := FromCompletionToolCall([]ToolCall{{
		ID:   "call_1",
		Type: "function",
		Function: struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		}{Name: "ping", Arguments: ""},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Function.Name != "ping" || got[0].Function.Arguments.Len() != 0 {
		t.Fatalf("%+v", got[0].Function)
	}
}

func TestFromChatRequest_WithLogprobs(t *testing.T) {
	trueVal := true

	req := ChatCompletionRequest{
		Model: "test-model",
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
		Logprobs:    &trueVal,
		TopLogprobs: 5,
	}

	result, err := FromChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !result.Logprobs {
		t.Error("expected Logprobs to be true")
	}

	if result.TopLogprobs != 5 {
		t.Errorf("expected TopLogprobs to be 5, got %d", result.TopLogprobs)
	}
}

func TestFromChatRequest_LogprobsDefault(t *testing.T) {
	req := ChatCompletionRequest{
		Model: "test-model",
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
	}

	result, err := FromChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Logprobs {
		t.Error("expected Logprobs to be false by default")
	}

	if result.TopLogprobs != 0 {
		t.Errorf("expected TopLogprobs to be 0 by default, got %d", result.TopLogprobs)
	}
}

func TestFromCompleteRequest_WithLogprobs(t *testing.T) {
	logprobsVal := 5

	req := CompletionRequest{
		Model:    "test-model",
		Prompt:   "Hello",
		Logprobs: &logprobsVal,
	}

	result, err := FromCompleteRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !result.Logprobs {
		t.Error("expected Logprobs to be true")
	}

	if result.TopLogprobs != 5 {
		t.Errorf("expected TopLogprobs to be 5, got %d", result.TopLogprobs)
	}
}

func TestFromCompleteRequest_LogprobsZero(t *testing.T) {
	zero := 0
	result, err := FromCompleteRequest(CompletionRequest{Model: "m", Prompt: "Hello", Logprobs: &zero})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Logprobs {
		t.Fatal("logprobs:0 should still enable chosen-token logprobs")
	}
	if result.TopLogprobs != 0 {
		t.Fatalf("TopLogprobs=%d", result.TopLogprobs)
	}
}

func TestFromCompleteRequest_LogprobsOutOfRange(t *testing.T) {
	six := 6
	if _, err := FromCompleteRequest(CompletionRequest{Model: "m", Prompt: "Hello", Logprobs: &six}); err == nil {
		t.Fatal("expected error for logprobs=6")
	}
	neg := -1
	if _, err := FromCompleteRequest(CompletionRequest{Model: "m", Prompt: "Hello", Logprobs: &neg}); err == nil {
		t.Fatal("expected error for logprobs=-1")
	}
}

func TestToListCompletionUsesModelIdentity(t *testing.T) {
	modified := time.Unix(1234567890, 0).UTC()

	result := ToListCompletion(api.ListResponse{
		Models: []api.ListModelResponse{
			{
				Name:       "legacy-name:latest",
				Model:      "namespace/exposed-model:latest",
				ModifiedAt: modified,
			},
			{
				Name:       "fallback-name:latest",
				ModifiedAt: modified.Add(time.Second),
			},
		},
	})

	if result.Object != "list" {
		t.Fatalf("object = %q, want list", result.Object)
	}
	if len(result.Data) != 2 {
		t.Fatalf("models = %d, want 2", len(result.Data))
	}

	if result.Data[0].Id != "namespace/exposed-model:latest" {
		t.Fatalf("id = %q, want model field", result.Data[0].Id)
	}
	if result.Data[0].OwnedBy != "namespace" {
		t.Fatalf("owned_by = %q, want namespace", result.Data[0].OwnedBy)
	}
	if result.Data[0].Created != modified.Unix() {
		t.Fatalf("created = %d, want %d", result.Data[0].Created, modified.Unix())
	}

	if result.Data[1].Id != "fallback-name:latest" {
		t.Fatalf("fallback id = %q, want name field", result.Data[1].Id)
	}
	if result.Data[1].OwnedBy != "library" {
		t.Fatalf("fallback owned_by = %q, want library", result.Data[1].OwnedBy)
	}
}

func TestToListCompletionAdvertisesContextLength(t *testing.T) {
	modified := time.Unix(1234567890, 0).UTC()
	result := ToListCompletion(api.ListResponse{
		Models: []api.ListModelResponse{{
			Name:           "gemma4:e4b",
			Model:          "gemma4:e4b",
			ModifiedAt:     modified,
			Details:        api.ModelDetails{ContextLength: 131072},
			HostMaxContext: 65536,
		}},
	})
	if result.Data[0].ContextLength != 131072 || result.Data[0].MaxModelLen != 131072 {
		t.Fatalf("top-level ctx = %d/%d", result.Data[0].ContextLength, result.Data[0].MaxModelLen)
	}
	if result.Data[0].ModelMaxTokens != 131072 {
		t.Fatalf("model_max_tokens = %d", result.Data[0].ModelMaxTokens)
	}
	if result.Data[0].Meta == nil || result.Data[0].Meta.ContextLength != 131072 {
		t.Fatalf("meta ctx = %+v", result.Data[0].Meta)
	}
	if result.Data[0].SupportsMTP {
		t.Fatal("supports_mtp should be omitted/false")
	}
}

func TestToListCompletionAdvertisesSupportsMTP(t *testing.T) {
	result := ToListCompletion(api.ListResponse{
		Models: []api.ListModelResponse{{
			Name:        "qwen3:4b",
			Model:       "qwen3:4b",
			SupportsMTP: true,
			Details:     api.ModelDetails{ContextLength: 4096},
		}},
	})
	if !result.Data[0].SupportsMTP {
		t.Fatal("supports_mtp")
	}
	if result.Data[0].Meta == nil || !result.Data[0].Meta.SupportsMTP {
		t.Fatalf("meta=%+v", result.Data[0].Meta)
	}
}

func TestToListCompletionAdvertisesCapabilities(t *testing.T) {
	result := ToListCompletion(api.ListResponse{
		Models: []api.ListModelResponse{{
			Name:  "qwen3:4b",
			Model: "qwen3:4b",
			Details: api.ModelDetails{
				Family:           "qwen3",
				ArchitectureType: "qwen3",
				ContextLength:    40960,
			},
			Capabilities: []model.Capability{
				model.CapabilityCompletion,
				model.CapabilityTools,
				model.CapabilityThinking,
				model.CapabilityVision,
			},
		}},
	})
	row := result.Data[0]
	want := []string{"chat", "tool_use", "streaming", "vision", "reasoning", "json_schema"}
	if len(row.Capabilities) != len(want) {
		t.Fatalf("capabilities=%v", row.Capabilities)
	}
	for i, s := range want {
		if row.Capabilities[i] != s {
			t.Fatalf("capabilities=%v", row.Capabilities)
		}
	}
	if len(row.InputModalities) != 2 || row.InputModalities[0] != "text" || row.InputModalities[1] != "image" {
		t.Fatalf("input_modalities=%v", row.InputModalities)
	}
	if row.Meta == nil || row.Meta.Architecture != "qwen3" {
		t.Fatalf("meta=%+v", row.Meta)
	}
}

func TestToModelAdvertisesContextFromModelInfo(t *testing.T) {
	got := ToModel(api.ShowResponse{
		ModifiedAt: time.Unix(1, 0),
		ModelInfo:  map[string]any{"qwen3.context_length": float64(40960)},
		Details:    api.ModelDetails{Family: "qwen3"},
		Capabilities: []model.Capability{
			model.CapabilityCompletion,
			model.CapabilityEmbedding,
		},
	}, "qwen3:4b")
	if got.ContextLength != 40960 || got.MaxModelLen != 40960 {
		t.Fatalf("ctx=%d max=%d", got.ContextLength, got.MaxModelLen)
	}
	if got.ModelMaxTokens != 40960 {
		t.Fatalf("model_max_tokens=%d", got.ModelMaxTokens)
	}
	if got.Meta == nil || got.Meta.ContextLength != 40960 || got.Meta.Architecture != "qwen3" {
		t.Fatalf("meta=%+v", got.Meta)
	}
	if got.Capabilities[len(got.Capabilities)-1] != "embeddings" {
		t.Fatalf("capabilities=%v", got.Capabilities)
	}
}

func TestToChatCompletion_WithLogprobs(t *testing.T) {
	createdAt := time.Unix(1234567890, 0)
	resp := api.ChatResponse{
		Model:     "test-model",
		CreatedAt: createdAt,
		Message:   api.Message{Role: "assistant", Content: "Hello there"},
		Logprobs: []api.Logprob{
			{
				TokenLogprob: api.TokenLogprob{
					Token:   "Hello",
					Logprob: -0.5,
				},
				TopLogprobs: []api.TokenLogprob{
					{Token: "Hello", Logprob: -0.5},
					{Token: "Hi", Logprob: -1.2},
				},
			},
			{
				TokenLogprob: api.TokenLogprob{
					Token:   " there",
					Logprob: -0.3,
				},
				TopLogprobs: []api.TokenLogprob{
					{Token: " there", Logprob: -0.3},
					{Token: " world", Logprob: -1.5},
				},
			},
		},
		Done: true,
		Metrics: api.Metrics{
			PromptEvalCount: 5,
			EvalCount:       2,
		},
	}

	id := "test-id"

	result := ToChatCompletion(id, resp)

	if result.Id != id {
		t.Errorf("expected Id %q, got %q", id, result.Id)
	}

	if result.Created != 1234567890 {
		t.Errorf("expected Created %d, got %d", int64(1234567890), result.Created)
	}

	if len(result.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(result.Choices))
	}

	choice := result.Choices[0]
	if choice.Message.Content != "Hello there" {
		t.Errorf("expected content %q, got %q", "Hello there", choice.Message.Content)
	}

	if choice.Logprobs == nil {
		t.Fatal("expected Logprobs to be present")
	}

	if len(choice.Logprobs.Content) != 2 {
		t.Fatalf("expected 2 logprobs, got %d", len(choice.Logprobs.Content))
	}

	// Verify first logprob
	if choice.Logprobs.Content[0].Token != "Hello" {
		t.Errorf("expected first token %q, got %q", "Hello", choice.Logprobs.Content[0].Token)
	}
	if choice.Logprobs.Content[0].Logprob != -0.5 {
		t.Errorf("expected first logprob -0.5, got %f", choice.Logprobs.Content[0].Logprob)
	}
	if len(choice.Logprobs.Content[0].TopLogprobs) != 2 {
		t.Errorf("expected 2 top_logprobs, got %d", len(choice.Logprobs.Content[0].TopLogprobs))
	}

	// Verify second logprob
	if choice.Logprobs.Content[1].Token != " there" {
		t.Errorf("expected second token %q, got %q", " there", choice.Logprobs.Content[1].Token)
	}
}

func TestToChatCompletion_WithoutLogprobs(t *testing.T) {
	createdAt := time.Unix(1234567890, 0)
	resp := api.ChatResponse{
		Model:     "test-model",
		CreatedAt: createdAt,
		Message:   api.Message{Role: "assistant", Content: "Hello"},
		Done:      true,
		Metrics: api.Metrics{
			PromptEvalCount: 5,
			EvalCount:       1,
		},
	}

	id := "test-id"

	result := ToChatCompletion(id, resp)

	if len(result.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(result.Choices))
	}

	// When no logprobs, Logprobs should be nil
	if result.Choices[0].Logprobs != nil {
		t.Error("expected Logprobs to be nil when not requested")
	}
}

func TestToCompletion_WithLogprobs(t *testing.T) {
	resp := api.GenerateResponse{
		Model:    "test-model",
		Response: "Hi",
		Logprobs: []api.Logprob{{
			TokenLogprob: api.TokenLogprob{Token: "Hi", Logprob: -0.1},
		}},
	}
	got := ToCompletion("id", resp)
	if got.Choices[0].Logprobs == nil || len(got.Choices[0].Logprobs.Tokens) != 1 || got.Choices[0].Logprobs.Tokens[0] != "Hi" {
		t.Fatalf("got %+v", got.Choices[0].Logprobs)
	}
	if got.Choices[0].Logprobs.TextOffset[0] != 0 {
		t.Fatalf("offset %+v", got.Choices[0].Logprobs.TextOffset)
	}
	chunk := ToCompleteChunk("id", resp)
	if chunk.Choices[0].Logprobs == nil || chunk.Choices[0].Logprobs.Tokens[0] != "Hi" {
		t.Fatalf("chunk %+v", chunk.Choices[0].Logprobs)
	}
}

func TestKeepaliveChunk(t *testing.T) {
	chunk := KeepaliveChunk("chatcmpl-test", "test-model")
	if chunk.Object != "chat.completion.chunk" {
		t.Fatalf("object = %q want chat.completion.chunk", chunk.Object)
	}
	if len(chunk.Choices) != 1 {
		t.Fatalf("choices = %d want 1", len(chunk.Choices))
	}
	if chunk.Choices[0].FinishReason != nil {
		t.Fatalf("finish_reason = %v want nil", chunk.Choices[0].FinishReason)
	}
	if chunk.Choices[0].Delta.Role != "assistant" {
		t.Fatalf("delta role = %q want assistant", chunk.Choices[0].Delta.Role)
	}
}

func TestToChunks_SplitsThinkingAndContent(t *testing.T) {
	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Thinking: "step-by-step",
			Content:  "final answer",
		},
		Done:       true,
		DoneReason: "stop",
	}

	chunks := ToChunks("test-id", resp, false)
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	reasoning := chunks[0].Choices[0]
	if reasoning.Delta.Reasoning != "step-by-step" {
		t.Fatalf("expected reasoning chunk to contain thinking, got %q", reasoning.Delta.Reasoning)
	}
	if reasoning.Delta.Content != "" {
		t.Fatalf("expected reasoning chunk content to be empty, got %v", reasoning.Delta.Content)
	}
	if len(reasoning.Delta.ToolCalls) != 0 {
		t.Fatalf("expected reasoning chunk tool calls to be empty, got %d", len(reasoning.Delta.ToolCalls))
	}
	if reasoning.FinishReason != nil {
		t.Fatalf("expected reasoning chunk finish reason to be nil, got %q", *reasoning.FinishReason)
	}

	content := chunks[1].Choices[0]
	if content.Delta.Reasoning != "" {
		t.Fatalf("expected content chunk reasoning to be empty, got %q", content.Delta.Reasoning)
	}
	if content.Delta.Content != "final answer" {
		t.Fatalf("expected content chunk content %q, got %v", "final answer", content.Delta.Content)
	}
	if content.FinishReason == nil || *content.FinishReason != "stop" {
		t.Fatalf("expected content chunk finish reason %q, got %v", "stop", content.FinishReason)
	}
}

func TestToChunks_SplitsThinkingAndToolCalls(t *testing.T) {
	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Thinking: "need a tool",
			ToolCalls: []api.ToolCall{
				{
					ID: "call_123",
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "Seattle",
						}),
					},
				},
			},
		},
		Done:       true,
		DoneReason: "stop",
	}

	chunks := ToChunks("test-id", resp, false)
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	reasoning := chunks[0].Choices[0]
	if reasoning.Delta.Reasoning != "need a tool" {
		t.Fatalf("expected reasoning chunk to contain thinking, got %q", reasoning.Delta.Reasoning)
	}
	if len(reasoning.Delta.ToolCalls) != 0 {
		t.Fatalf("expected reasoning chunk tool calls to be empty, got %d", len(reasoning.Delta.ToolCalls))
	}
	if reasoning.FinishReason != nil {
		t.Fatalf("expected reasoning chunk finish reason to be nil, got %q", *reasoning.FinishReason)
	}

	toolCallChunk := chunks[1].Choices[0]
	if toolCallChunk.Delta.Reasoning != "" {
		t.Fatalf("expected tool-call chunk reasoning to be empty, got %q", toolCallChunk.Delta.Reasoning)
	}
	if len(toolCallChunk.Delta.ToolCalls) != 1 {
		t.Fatalf("expected one tool call in second chunk, got %d", len(toolCallChunk.Delta.ToolCalls))
	}
	if toolCallChunk.Delta.ToolCalls[0].ID != "call_123" {
		t.Fatalf("expected tool call id %q, got %q", "call_123", toolCallChunk.Delta.ToolCalls[0].ID)
	}
	if toolCallChunk.FinishReason == nil || *toolCallChunk.FinishReason != finishReasonToolCalls {
		t.Fatalf("expected tool-call chunk finish reason %q, got %v", finishReasonToolCalls, toolCallChunk.FinishReason)
	}
}

func TestToChunks_SingleChunkForNonMixedResponses(t *testing.T) {
	toolCalls := []api.ToolCall{
		{
			ID: "call_456",
			Function: api.ToolCallFunction{
				Index: 0,
				Name:  "get_time",
				Arguments: testArgs(map[string]any{
					"timezone": "UTC",
				}),
			},
		},
	}

	tests := []struct {
		name    string
		message api.Message
	}{
		{
			name:    "thinking-only",
			message: api.Message{Thinking: "pondering"},
		},
		{
			name:    "content-only",
			message: api.Message{Content: "hello"},
		},
		{
			name:    "toolcalls-only",
			message: api.Message{ToolCalls: toolCalls},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := api.ChatResponse{
				Model:   "test-model",
				Message: tt.message,
			}

			chunks := ToChunks("test-id", resp, false)
			if len(chunks) != 1 {
				t.Fatalf("expected 1 chunk, got %d", len(chunks))
			}
		})
	}
}

func TestToChunks_SplitsThinkingAndToolCallsWhenNotDone(t *testing.T) {
	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Thinking: "need a tool",
			ToolCalls: []api.ToolCall{
				{
					ID: "call_789",
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "San Francisco",
						}),
					},
				},
			},
		},
		Done: false,
	}

	chunks := ToChunks("test-id", resp, false)
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	reasoning := chunks[0].Choices[0]
	if reasoning.Delta.Reasoning != "need a tool" {
		t.Fatalf("expected reasoning chunk to contain thinking, got %q", reasoning.Delta.Reasoning)
	}
	if reasoning.FinishReason != nil {
		t.Fatalf("expected reasoning chunk finish reason nil, got %v", reasoning.FinishReason)
	}

	toolCallChunk := chunks[1].Choices[0]
	if len(toolCallChunk.Delta.ToolCalls) != 1 {
		t.Fatalf("expected one tool call in second chunk, got %d", len(toolCallChunk.Delta.ToolCalls))
	}
	if toolCallChunk.Delta.ToolCalls[0].ID != "call_789" {
		t.Fatalf("expected tool call id %q, got %q", "call_789", toolCallChunk.Delta.ToolCalls[0].ID)
	}
	if toolCallChunk.FinishReason != nil {
		t.Fatalf("expected tool-call chunk finish reason nil when not done, got %v", toolCallChunk.FinishReason)
	}
}

func TestToChunks_SplitsThinkingAndContentWhenNotDone(t *testing.T) {
	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Thinking: "thinking",
			Content:  "partial content",
		},
		Done: false,
	}

	chunks := ToChunks("test-id", resp, false)
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	reasoning := chunks[0].Choices[0]
	if reasoning.Delta.Reasoning != "thinking" {
		t.Fatalf("expected reasoning chunk to contain thinking, got %q", reasoning.Delta.Reasoning)
	}
	if reasoning.FinishReason != nil {
		t.Fatalf("expected reasoning chunk finish reason nil, got %v", reasoning.FinishReason)
	}

	content := chunks[1].Choices[0]
	if content.Delta.Content != "partial content" {
		t.Fatalf("expected content chunk content %q, got %v", "partial content", content.Delta.Content)
	}
	if content.FinishReason != nil {
		t.Fatalf("expected content chunk finish reason nil when not done, got %v", content.FinishReason)
	}
}

func TestToChunks_SplitSendsLogprobsOnContentChunk(t *testing.T) {
	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Thinking: "thinking",
			Content:  "content",
		},
		Logprobs: []api.Logprob{
			{
				TokenLogprob: api.TokenLogprob{
					Token:   "tok",
					Logprob: -0.25,
				},
			},
		},
		Done:       true,
		DoneReason: "stop",
	}

	chunks := ToChunks("test-id", resp, false)
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	first := chunks[0].Choices[0]
	if first.Logprobs != nil {
		t.Fatalf("expected reasoning chunk logprobs nil, got %+v", first.Logprobs)
	}

	second := chunks[1].Choices[0]
	if second.Logprobs == nil {
		t.Fatal("expected content chunk to include logprobs")
	}
	if len(second.Logprobs.Content) != 1 || second.Logprobs.Content[0].Token != "tok" {
		t.Fatalf("unexpected content chunk logprobs: %+v", second.Logprobs.Content)
	}
}

func TestToChunks_ThinkingOnlyDropsLogprobs(t *testing.T) {
	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Thinking: "reasoning",
		},
		Logprobs: []api.Logprob{{
			TokenLogprob: api.TokenLogprob{Token: "tok", Logprob: -0.25},
		}},
	}
	chunks := ToChunks("test-id", resp, false)
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks", len(chunks))
	}
	if chunks[0].Choices[0].Logprobs != nil {
		t.Fatalf("reasoning-only logprobs = %+v", chunks[0].Choices[0].Logprobs)
	}
	nonstream := ToChatCompletion("id", resp)
	if nonstream.Choices[0].Logprobs != nil {
		t.Fatalf("non-stream reasoning-only logprobs = %+v", nonstream.Choices[0].Logprobs)
	}
}

func TestToChunk_LegacyMixedThinkingAndContentSingleChunk(t *testing.T) {
	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Thinking: "reasoning",
			Content:  "answer",
		},
		Done:       true,
		DoneReason: "stop",
	}

	chunk := ToChunk("test-id", resp, false)
	if len(chunk.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(chunk.Choices))
	}

	delta := chunk.Choices[0].Delta
	if delta.Reasoning != "reasoning" {
		t.Fatalf("expected reasoning %q, got %q", "reasoning", delta.Reasoning)
	}
	if delta.Content != "answer" {
		t.Fatalf("expected content %q, got %v", "answer", delta.Content)
	}
}

func TestFromChatRequest_TopLogprobsRange(t *testing.T) {
	tests := []struct {
		name        string
		topLogprobs int
		expectValid bool
	}{
		{name: "valid: 0", topLogprobs: 0, expectValid: true},
		{name: "valid: 1", topLogprobs: 1, expectValid: true},
		{name: "valid: 10", topLogprobs: 10, expectValid: true},
		{name: "valid: 20", topLogprobs: 20, expectValid: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			trueVal := true
			req := ChatCompletionRequest{
				Model: "test-model",
				Messages: []Message{
					{Role: "user", Content: "Hello"},
				},
				Logprobs:    &trueVal,
				TopLogprobs: tt.topLogprobs,
			}

			result, err := FromChatRequest(req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if result.TopLogprobs != tt.topLogprobs {
				t.Errorf("expected TopLogprobs %d, got %d", tt.topLogprobs, result.TopLogprobs)
			}
		})
	}
}

func TestFromImageEditRequest_Basic(t *testing.T) {
	req := ImageEditRequest{
		Model:  "test-model",
		Prompt: "make it blue",
		Image:  prefix + image,
	}

	result, err := FromImageEditRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Model != "test-model" {
		t.Errorf("expected model 'test-model', got %q", result.Model)
	}

	if result.Prompt != "make it blue" {
		t.Errorf("expected prompt 'make it blue', got %q", result.Prompt)
	}

	if len(result.Images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(result.Images))
	}
}

func TestFromImageEditRequest_WithSize(t *testing.T) {
	req := ImageEditRequest{
		Model:  "test-model",
		Prompt: "make it blue",
		Image:  prefix + image,
		Size:   "512x768",
	}

	result, err := FromImageEditRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Width != 512 {
		t.Errorf("expected width 512, got %d", result.Width)
	}

	if result.Height != 768 {
		t.Errorf("expected height 768, got %d", result.Height)
	}
}

func TestFromImageEditRequest_WithSeed(t *testing.T) {
	seed := int64(12345)
	req := ImageEditRequest{
		Model:  "test-model",
		Prompt: "make it blue",
		Image:  prefix + image,
		Seed:   &seed,
	}

	result, err := FromImageEditRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Options == nil {
		t.Fatal("expected options to be set")
	}

	if result.Options["seed"] != seed {
		t.Errorf("expected seed %d, got %v", seed, result.Options["seed"])
	}
}

func TestFromImageEditRequest_InvalidImage(t *testing.T) {
	req := ImageEditRequest{
		Model:  "test-model",
		Prompt: "make it blue",
		Image:  "not-valid-base64",
	}

	_, err := FromImageEditRequest(req)
	if err == nil {
		t.Error("expected error for invalid image")
	}
}

func TestRejectUnsupportedImageOpenAI(t *testing.T) {
	n := 2
	_, err := FromImageEditRequest(ImageEditRequest{Model: "m", Prompt: "p", Image: prefix + image, N: &n})
	if err == nil || !strings.Contains(err.Error(), "n=2") {
		t.Fatalf("n: %v", err)
	}
	_, err = FromImageEditRequest(ImageEditRequest{Model: "m", Prompt: "p", Image: prefix + image, Mask: "data:image/png;base64,xx"})
	if err == nil || !strings.Contains(err.Error(), "mask") {
		t.Fatalf("mask: %v", err)
	}
	st := true
	_, err = FromImageEditRequest(ImageEditRequest{Model: "m", Prompt: "p", Image: prefix + image, Stream: &st})
	if err == nil || !strings.Contains(err.Error(), "stream") {
		t.Fatalf("stream: %v", err)
	}
	_, err = FromImageGenerationRequest(ImageGenerationRequest{Model: "m", Prompt: "p", ResponseFormat: "url"})
	if err == nil || !strings.Contains(err.Error(), "response_format") {
		t.Fatalf("url: %v", err)
	}
	if _, err := FromImageGenerationRequest(ImageGenerationRequest{Model: "m", Prompt: "p", N: 1}); err != nil {
		t.Fatal(err)
	}
}
