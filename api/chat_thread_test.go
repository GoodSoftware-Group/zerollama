package api

import "testing"

func TestChatThreadStickyElideFrom(t *testing.T) {
	th := &ChatThread{Model: "qwen3.5:9b", Messages: []Message{{Role: "user", Content: "hi"}}}
	req := th.NextRequest()
	if req.Compression != nil {
		t.Fatal("no sticky yet")
	}
	th.Observe(ChatResponse{Compression: &ChatCompressionMeta{Mode: "placeholder", ElideFrom: 4}})
	req = th.NextRequest()
	if req.Compression == nil || req.Compression.ElideFrom == nil || *req.Compression.ElideFrom != 4 {
		t.Fatalf("sticky %+v", req.Compression)
	}
	th.ClearHistory()
	if th.CompressionMeta() != nil || len(th.Messages) != 0 {
		t.Fatal("clear should drop sticky and messages")
	}
}

func TestChatThreadExplicitElideFromWins(t *testing.T) {
	from := 1
	th := &ChatThread{
		Model:       "m",
		Compression: &ChatCompressionConfig{ElideFrom: &from},
	}
	th.Observe(ChatResponse{Compression: &ChatCompressionMeta{Mode: "placeholder", ElideFrom: 9}})
	req := th.NextRequest()
	if req.Compression == nil || req.Compression.ElideFrom == nil || *req.Compression.ElideFrom != 1 {
		t.Fatalf("explicit should win: %+v", req.Compression)
	}
}
