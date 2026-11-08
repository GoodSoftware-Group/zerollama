package kokoro

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestPathsRequireBothFiles(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "m.onnx")
	voices := filepath.Join(dir, "v.bin")
	_ = os.WriteFile(model, []byte("x"), 0o644)
	if m, v := Paths(model, ""); m != "" || v != "" {
		t.Fatalf("incomplete %q %q", m, v)
	}
	_ = os.WriteFile(voices, []byte("x"), 0o644)
	m, v := Paths(model, voices)
	if m != model || v != voices {
		t.Fatalf("got %q %q", m, v)
	}
}

func fakeKokoroRepo(t *testing.T) (model, voices string) {
	t.Helper()
	dir := t.TempDir()
	model = filepath.Join(dir, "m.onnx")
	voices = filepath.Join(dir, "v.bin")
	if err := os.WriteFile(model, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(voices, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	repo := filepath.Join(dir, "repo")
	pkg := filepath.Join(repo, "runtime", "runtime", "tts")
	if err := os.MkdirAll(pkg, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{
		filepath.Join(repo, "runtime", "runtime", "__init__.py"),
		filepath.Join(pkg, "__init__.py"),
	} {
		if err := os.WriteFile(name, []byte(""), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	src := `import argparse, base64, json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--voices", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--default-voice", default="af_heart")
    args = ap.parse_args()
    wav = b"RIFF....WAVEfmt "
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            b = b'{"status":"ok","rate":24000}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode())
            voice = body.get("voice") or ""
            log = os.environ.get("KOKORO_VOICE_LOG")
            if log:
                with open(log, "a") as f:
                    f.write((voice or "") + "\t" + str(body.get("lang") or "") + "\n")
            if voice == "nope":
                b = b'{"error":"voice nope not found"}'
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers(); self.wfile.write(b)
                return
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Connection", "close")
                self.end_headers()
                pcm = base64.b64encode(b"\x00\x01\x00\x02").decode()
                self.wfile.write(json.dumps({"rate":24000,"pcm":pcm}).encode()+b"\n")
                self.wfile.write(json.dumps({"done":True}).encode()+b"\n")
                return
            out = json.dumps({"rate":24000,"wav":base64.b64encode(wav).decode()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers(); self.wfile.write(out)
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()

if __name__ == "__main__":
    main()
`
	if err := os.WriteFile(filepath.Join(pkg, "kokoro_serve.py"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ZEROLLAMA_REPO", repo)
	return model, voices
}

func TestServerSpeakAndStream(t *testing.T) {
	model, voices := fakeKokoroRepo(t)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	proc, err := StartServer(ctx, model, voices, "af_heart")
	if err != nil {
		t.Fatal(err)
	}
	defer proc.Close()
	if err := proc.Wait(ctx); err != nil {
		t.Fatalf("wait: %v stderr=%s", err, proc.stderr())
	}
	wav, rate, err := proc.Speak(ctx, "hello", "af_heart", "", "", "", 1)
	if err != nil {
		t.Fatal(err)
	}
	if rate != 24000 || !strings.HasPrefix(string(wav), "RIFF") {
		t.Fatalf("rate=%d wav=%q", rate, wav[:min(8, len(wav))])
	}
	var chunks int
	if err := proc.SpeakStream(ctx, "hello", "af_heart", "", "", "", 1, func(r int, pcm []byte) error {
		chunks++
		if r != 24000 || len(pcm) == 0 {
			t.Fatalf("chunk rate=%d len=%d", r, len(pcm))
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if chunks != 1 {
		t.Fatalf("chunks=%d", chunks)
	}
}

func TestValidateSpeed(t *testing.T) {
	for _, speed := range []float64{0, 0.49, 2.01, math.NaN(), math.Inf(1)} {
		if err := validateSpeed(speed); !errors.Is(err, ErrBadRequest) {
			t.Fatalf("speed %v: %v", speed, err)
		}
	}
	for _, speed := range []float64{0.5, 1, 2} {
		if err := validateSpeed(speed); err != nil {
			t.Fatal(err)
		}
	}
}

func TestSpeakUsesRequestDefaultVoice(t *testing.T) {
	model, voices := fakeKokoroRepo(t)
	logPath := filepath.Join(t.TempDir(), "voices.log")
	t.Setenv("KOKORO_VOICE_LOG", logPath)
	s := &Shared{}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	defer func() {
		s.mu.Lock()
		proc := s.proc
		s.proc = nil
		s.mu.Unlock()
		if proc != nil {
			proc.Close()
		}
	}()
	if _, _, err := s.Speak(ctx, model, voices, "hi", "", "", "", "", 1, "af_bella"); err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.Speak(ctx, model, voices, "hi", "", "", "", "", 1, "af_sky"); err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.Speak(ctx, model, voices, "hi", "af_heart", "", "", "", 1, "af_sky"); err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.Speak(ctx, model, voices, "hi", "af_heart", "en-GB", "", "", 1, "af_sky"); err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.Speak(ctx, model, voices, "hi", "bf_emma", "", "", "", 1, "af_heart"); err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.Speak(ctx, model, voices, "hi", "zf_xiaobei", "", "", "", 1, "af_heart"); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("non-English voice: %v", err)
	}
	if _, _, err := s.Speak(ctx, model, voices, "hi", "af_heart", "zh", "", "", 1, "af_heart"); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("non-English lang: %v", err)
	}
	if _, _, err := s.Speak(ctx, model, voices, "hi", "af_heart", "", "", "joyful", 1, "af_heart"); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("emotion: %v", err)
	}
	if _, _, err := s.Speak(ctx, model, voices, "hi", "af_heart", "", "", "excited", 1.9, "af_heart"); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("emotion speed: %v", err)
	}
	got, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	want := "af_bella\ten-us\naf_sky\ten-us\naf_heart\ten-us\naf_heart\ten-gb\nbf_emma\ten-gb\n"
	if string(got) != want {
		t.Fatalf("voices %q", got)
	}
}

func TestSpeakUnknownVoice(t *testing.T) {
	model, voices := fakeKokoroRepo(t)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	proc, err := StartServer(ctx, model, voices, "af_heart")
	if err != nil {
		t.Fatal(err)
	}
	defer proc.Close()
	if err := proc.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	if _, _, err := proc.Speak(ctx, "hi", "nope", "", "", "", 1); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("speak: %v", err)
	}
	err = proc.SpeakStream(ctx, "hi", "nope", "", "", "", 1, func(int, []byte) error {
		t.Fatal("emitted audio for a rejected voice")
		return nil
	})
	if !errors.Is(err, ErrBadRequest) {
		t.Fatalf("stream: %v", err)
	}
}

func TestSharedRestartsAfterCrash(t *testing.T) {
	model, voices := fakeKokoroRepo(t)
	s := &Shared{}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	defer func() {
		s.mu.Lock()
		proc := s.proc
		s.proc = nil
		s.mu.Unlock()
		if proc != nil {
			proc.Close()
		}
	}()
	if _, _, err := s.Speak(ctx, model, voices, "hi", "af_heart", "", "", "", 1, "af_heart"); err != nil {
		t.Fatal(err)
	}
	s.mu.Lock()
	proc := s.proc
	s.mu.Unlock()
	proc.mu.Lock()
	cmd := proc.cmd
	proc.mu.Unlock()
	if cmd == nil || cmd.Process == nil {
		t.Fatal("no process")
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(3 * time.Second)
	for proc.Ready() {
		if time.Now().After(deadline) {
			t.Fatal("process still ready after kill")
		}
		time.Sleep(20 * time.Millisecond)
	}
	wav, rate, err := s.Speak(ctx, model, voices, "hi", "af_heart", "", "", "", 1, "af_heart")
	if err != nil {
		t.Fatal(err)
	}
	if rate != 24000 || !strings.HasPrefix(string(wav), "RIFF") {
		t.Fatalf("rate=%d wav=%q", rate, wav)
	}
}

func TestSpeakPayloadKeepsSpeedAndControls(t *testing.T) {
	raw, err := json.Marshal(speakPayload("Hi.", "af_heart", "en-us", `[[".","0.2"]]`, "excited", 1, false))
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		Speed   float64         `json:"speed"`
		Emotion string          `json:"emotion"`
		Pauses  json.RawMessage `json:"phoneme_silence"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}
	if body.Speed != 1 || body.Emotion != "excited" || string(body.Pauses) != `[[".","0.2"]]` {
		t.Fatalf("%s", raw)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
