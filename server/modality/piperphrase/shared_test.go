package piperphrase

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSharedReusesOneProcess(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "starts")
	script := writeFakePiper(t, dir, logPath, false)
	var shared Shared
	defer shared.Close()
	args := []string{"--json-input"}
	first, err := shared.Speak(context.Background(), script, args, "Hello.", filepath.Join(dir, "a.wav"))
	if err != nil {
		t.Fatal(err)
	}
	second, err := shared.Speak(context.Background(), script, args, "Next.", filepath.Join(dir, "b.wav"))
	if err != nil {
		t.Fatal(err)
	}
	changed, err := shared.Speak(context.Background(), script, []string{"--json-input", "--length_scale", "1"}, "Last.", filepath.Join(dir, "c.wav"))
	if err != nil {
		t.Fatal(err)
	}
	starts, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(starts), "start") != 2 || string(first) != "RIFFHello." || string(second) != "RIFFNext." || string(changed) != "RIFFLast." {
		t.Fatalf("starts %q first %q second %q changed %q", starts, first, second, changed)
	}
}
