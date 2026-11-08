package piperphrase

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestProcReusesOneProcess(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "starts")
	script := writeFakePiper(t, dir, logPath, false)
	proc, err := Start(script, []string{"--json-input"})
	if err != nil {
		t.Fatal(err)
	}
	defer proc.Close()
	first, err := proc.Speak(context.Background(), "Hello.", filepath.Join(dir, "a.wav"))
	if err != nil {
		t.Fatal(err)
	}
	second, err := proc.Speak(context.Background(), "Next.", filepath.Join(dir, "b.wav"))
	if err != nil {
		t.Fatal(err)
	}
	starts, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(starts), "start") != 1 || string(first) != "RIFFHello." || string(second) != "RIFFNext." {
		t.Fatalf("starts %q first %q second %q", starts, first, second)
	}
}

func TestSpeakPCMStreamsBeforeTheUtteranceEnds(t *testing.T) {
	dir := t.TempDir()
	script := filepath.Join(dir, "piper")
	body := `#!/usr/bin/env python3
import json, sys, time
for line in sys.stdin:
    msg = json.loads(line)
    sys.stdout.buffer.write(msg["text"].encode()[:1] + b"\x00")
    sys.stdout.buffer.flush()
    time.sleep(0.6)
    rest = msg["text"].encode()[1:]
    if len(rest) % 2 != 0:
        rest += b"\x00"
    sys.stdout.buffer.write(rest)
    sys.stdout.buffer.flush()
    sys.stderr.write("Real-time factor: 1 (infer=0 sec, audio=0 sec)\n")
    sys.stderr.flush()
`
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	proc, err := Start(script, []string{"--json-input", "--output_raw"})
	if err != nil {
		t.Fatal(err)
	}
	defer proc.Close()
	start := time.Now()
	var first time.Time
	var got []byte
	err = proc.SpeakPCM(context.Background(), "Hi", func(chunk []byte) error {
		if first.IsZero() {
			first = time.Now()
		}
		got = append(got, chunk...)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if first.IsZero() || first.Sub(start) > 400*time.Millisecond {
		t.Fatalf("first chunk at %s", first.Sub(start))
	}
	if string(got) != "H\x00i\x00" {
		t.Fatalf("pcm %q", got)
	}
	got = nil
	if err := proc.SpeakPCM(context.Background(), "Yo", func(chunk []byte) error {
		got = append(got, chunk...)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if string(got) != "Y\x00o\x00" {
		t.Fatalf("second pcm %q", got)
	}
}

func TestProcCancelStopsTheProcess(t *testing.T) {
	dir := t.TempDir()
	script := writeFakePiper(t, dir, filepath.Join(dir, "starts"), true)
	proc, err := Start(script, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer proc.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	_, err = proc.Speak(ctx, "Hello.", filepath.Join(dir, "a.wav"))
	if err == nil {
		t.Fatal("expected cancel")
	}
}

func writeFakePiper(t *testing.T, dir, logPath string, stall bool) string {
	t.Helper()
	script := filepath.Join(dir, "piper")
	stallLine := ""
	if stall {
		stallLine = "time.sleep(30)\n"
	}
	body := "#!/usr/bin/env python3\nimport json, sys, time\nopen(" + pyQuote(logPath) + ", 'a').write('start\\n')\nfor line in sys.stdin:\n    " + stallLine + "msg = json.loads(line)\n    open(msg['output_file'], 'wb').write(b'RIFF' + msg['text'].encode())\n    sys.stdout.write(msg['output_file'] + '\\n')\n    sys.stdout.flush()\n"
	if stall {
		body = "#!/usr/bin/env python3\nimport json, sys, time\nopen(" + pyQuote(logPath) + ", 'a').write('start\\n')\nfor line in sys.stdin:\n    time.sleep(30)\n    msg = json.loads(line)\n    open(msg['output_file'], 'wb').write(b'RIFF' + msg['text'].encode())\n    sys.stdout.write(msg['output_file'] + '\\n')\n    sys.stdout.flush()\n"
	}
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return script
}

func pyQuote(path string) string {
	return "'" + strings.ReplaceAll(path, "'", "\\'") + "'"
}
