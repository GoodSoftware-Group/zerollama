package piperphrase

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
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

// Switching flag sets must not kill a process another request is still reading.
func TestSharedKeepsInFlightProcessOnFlagChange(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "starts")
	script := filepath.Join(dir, "piper")
	body := "#!/usr/bin/env python3\nimport json, sys, time\nopen(" + pyQuote(logPath) + ", 'a').write('start\\n')\nfor line in sys.stdin:\n    msg = json.loads(line)\n    time.sleep(0.25)\n    open(msg['output_file'], 'wb').write(b'RIFF' + msg['text'].encode())\n    sys.stdout.write(msg['output_file'] + '\\n')\n    sys.stdout.flush()\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	var shared Shared
	defer shared.Close()

	var wg sync.WaitGroup
	errCh := make(chan error, 2)
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, err := shared.Speak(context.Background(), script, []string{"--json-input"}, "slow", filepath.Join(dir, "a.wav"))
		errCh <- err
	}()
	time.Sleep(50 * time.Millisecond)
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, err := shared.Speak(context.Background(), script, []string{"--json-input", "--length_scale", "1"}, "fast", filepath.Join(dir, "b.wav"))
		errCh <- err
	}()
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			t.Fatalf("in-flight speak killed: %v", err)
		}
	}
	starts, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(starts), "start") != 2 {
		t.Fatalf("starts %q", starts)
	}
}
