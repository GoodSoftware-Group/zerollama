package api

import (
	"io"
	"os"
	"strings"
	"testing"
	"time"
)

func TestMetricsSummaryCachedPromptTokens(t *testing.T) {
	read, write, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	original := os.Stderr
	os.Stderr = write
	t.Cleanup(func() { os.Stderr = original })

	(&Metrics{
		PromptEvalCount:    10,
		CachedPromptTokens: 4,
		PromptEvalDuration: time.Second,
	}).Summary()
	write.Close()
	os.Stderr = original

	output, err := io.ReadAll(read)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"prompt eval count:    10 token(s)",
		"prompt eval cached:   4 token(s)",
		"prompt eval rate:     6.00 tokens/s",
	} {
		if !strings.Contains(string(output), want) {
			t.Errorf("summary missing %q:\n%s", want, output)
		}
	}
}
