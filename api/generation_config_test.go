package api

import "testing"

func TestSamplingMapFromGenerationConfig(t *testing.T) {
	got := SamplingMapFromGenerationConfig([]byte(`{
		"temperature": 0.6,
		"top_p": 0.95,
		"top_k": 20,
		"repetition_penalty": 1.05,
		"do_sample": true
	}`))
	if got["temperature"] != 0.6 {
		t.Fatalf("temperature=%v", got["temperature"])
	}
	if got["top_p"] != 0.95 {
		t.Fatalf("top_p=%v", got["top_p"])
	}
	if got["top_k"] != 20 {
		t.Fatalf("top_k=%v", got["top_k"])
	}
	if got["repeat_penalty"] != 1.05 {
		t.Fatalf("repeat_penalty=%v", got["repeat_penalty"])
	}

	greedy := SamplingMapFromGenerationConfig([]byte(`{"temperature": 0.7, "do_sample": false}`))
	if greedy["temperature"] != 0.0 {
		t.Fatalf("do_sample false should force temperature 0, got %v", greedy["temperature"])
	}

	if SamplingMapFromGenerationConfig([]byte(`{"eos_token_id": 1}`)) != nil {
		t.Fatal("token-only config should not produce sampling keys")
	}
	if SamplingMapFromGenerationConfig(nil) != nil {
		t.Fatal("empty")
	}
	id := SamplingMapFromGenerationConfig([]byte(`{"temperature": 1.0, "top_p": 1.0, "do_sample": true}`))
	if id != nil {
		t.Fatalf("HF identity 1.0/1.0 should be omitted, got %v", id)
	}
}
