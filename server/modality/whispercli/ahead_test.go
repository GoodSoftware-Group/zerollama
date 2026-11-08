package whispercli

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestAheadWaitsForTheClip(t *testing.T) {
	dir := t.TempDir()
	flag := filepath.Join(dir, "up")
	out := filepath.Join(dir, "out")
	script := filepath.Join(dir, "whisper")
	body := "#!/usr/bin/env python3\nimport sys\nopen(sys.argv[1], 'w').write('up')\ndata = sys.stdin.buffer.read()\nopen(sys.argv[2] + '.txt', 'w').write('heard\\n')\nopen(sys.argv[2] + '.bin', 'wb').write(data)\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	proc, err := StartAhead(ctx, script, []string{flag, out}, out)
	if err != nil {
		t.Fatal(err)
	}
	defer proc.Close()
	deadline := time.Now().Add(2 * time.Second)
	for {
		if _, err := os.Stat(flag); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("process did not start")
		}
		time.Sleep(10 * time.Millisecond)
	}
	text, err := proc.Finish(context.Background(), []byte("RIFF-clip"))
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(out + ".bin")
	if err != nil {
		t.Fatal(err)
	}
	if text != "heard" || string(got) != "RIFF-clip" {
		t.Fatalf("text %q audio %q", text, got)
	}
}
