package cmd

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestListContextSummary(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		m    api.ListModelResponse
		want string
	}{
		{"empty", api.ListModelResponse{}, "--"},
		{"train only", api.ListModelResponse{Details: api.ModelDetails{ContextLength: 81920}}, "–80k"},
		{"host fits", api.ListModelResponse{Details: api.ModelDetails{ContextLength: 81920}, HostMaxContext: 81920}, "80k"},
		{"host above train", api.ListModelResponse{Details: api.ModelDetails{ContextLength: 8192}, HostMaxContext: 16384}, "8k"},
		{"range", api.ListModelResponse{Details: api.ModelDetails{ContextLength: 131072}, HostMaxContext: 16384}, "16k–128k"},
		{"host only", api.ListModelResponse{HostMaxContext: 4096}, "4k"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := listContextSummary(tc.m); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}
