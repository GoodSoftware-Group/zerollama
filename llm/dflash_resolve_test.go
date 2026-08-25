package llm

import "testing"

func TestResolveDFlashSpecTypeFromRealHelp(t *testing.T) {
	t.Cleanup(resetLlamaServerHelpCache)
	bin := "/Users/user1/Sites/inference/zerollama/vendor/llama-cpp-c84b3020/build/bin/llama-server"
	if !isUsableLlamaServerBin(bin) {
		t.Skip("c84 llama-server missing")
	}
	got := resolveDFlashSpecType(bin)
	if got != "dflash" {
		t.Fatalf("resolveDFlashSpecType = %q, want dflash (help=%q)", got, truncate(llamaServerHelp(bin), 200))
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
