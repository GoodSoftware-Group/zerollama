package modality

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/server/modality/kokoro"
	"github.com/ollama/ollama/server/modality/piperphrase"
	"github.com/ollama/ollama/types/model"
)

func TestSpeechKokoroRejectsSpeedAndEmpty(t *testing.T) {
	zero := 0.0
	_, _, _, err := SpeechKokoro(context.Background(), model.ConfigV2{}, "hi", "af_heart", "", &zero, "", "")
	if !IsKokoroRequest(err) || !strings.Contains(err.Error(), "0.5") {
		t.Fatalf("speed: %v", err)
	}
	_, _, _, _, err = SpeechKokoroStreamReader(context.Background(), model.ConfigV2{}, "hi", "af_heart", "", &zero, "", "")
	if !IsKokoroRequest(err) {
		t.Fatalf("stream speed: %v", err)
	}
	_, _, _, err = SpeechKokoro(context.Background(), model.ConfigV2{}, "  ", "af_heart", "", nil, "", "")
	if !IsKokoroRequest(err) || !strings.Contains(err.Error(), "input") {
		t.Fatalf("empty: %v", err)
	}
	_, _, _, err = SpeechKokoro(context.Background(), model.ConfigV2{}, "hi", "af_heart", "", nil, "", "joyful")
	if !IsKokoroRequest(err) || !strings.Contains(err.Error(), "emotion") {
		t.Fatalf("emotion: %v", err)
	}
	fast := 1.8
	_, _, _, err = SpeechKokoro(context.Background(), model.ConfigV2{}, "hi", "af_heart", "", &fast, "", "excited")
	if !IsKokoroRequest(err) || !strings.Contains(err.Error(), "0.5") {
		t.Fatalf("emotion speed: %v", err)
	}
}

func TestSpeechKokoroRejectsNonEnglish(t *testing.T) {
	_, _, _, err := SpeechKokoro(context.Background(), model.ConfigV2{}, "hi", "zf_xiaobei", "", nil, "", "")
	if !IsKokoroRequest(err) || !strings.Contains(err.Error(), "zf_xiaobei") {
		t.Fatalf("voice: %v", err)
	}
	_, _, _, err = SpeechKokoro(context.Background(), model.ConfigV2{}, "hi", "af_heart", "zh", nil, "", "")
	if !IsKokoroRequest(err) || !strings.Contains(err.Error(), "en-us") {
		t.Fatalf("lang: %v", err)
	}
	_, _, _, _, err = SpeechKokoroStreamReader(context.Background(), model.ConfigV2{}, "hi", "bf_emma", "ja", nil, "", "")
	if !IsKokoroRequest(err) {
		t.Fatalf("stream: %v", err)
	}
}

func TestSpeechKokoroStreamReaderSurfacesVoiceError(t *testing.T) {
	modelPath, voices := writeSpeechFake(t)
	defer kokoro.CloseShared()
	cfg := model.ConfigV2{BackendPaths: map[string]string{
		"kokoro_model":  modelPath,
		"kokoro_voices": voices,
	}}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	_, _, r, _, err := SpeechKokoroStreamReader(ctx, cfg, "hello", "nope", "", nil, "", "")
	if r != nil {
		r.Close()
		t.Fatal("reader returned before the voice error")
	}
	if !IsKokoroRequest(err) || !strings.Contains(err.Error(), "nope") {
		t.Fatalf("got %v", err)
	}
}

func TestSpeechKokoroCacheHit(t *testing.T) {
	dir := t.TempDir()
	cfg := model.ConfigV2{BackendPaths: map[string]string{
		"kokoro_model":  filepath.Join(dir, "kokoro.onnx"),
		"kokoro_voices": filepath.Join(dir, "voices.bin"),
	}}
	pcm := []byte{0x01, 0x00, 0x02, 0x00}
	wav := piperphrase.PCM16WAV(kokoro.SampleRate, pcm)
	text := "Okay."
	piperphrase.Store(piperphrase.Key{
		Model:  cfg.BackendPaths["kokoro_model"],
		Config: cfg.BackendPaths["kokoro_voices"],
		Voice:  "af_heart",
		Scale:  "1.0000",
		Lang:   "en-us",
		Text:   text,
	}, wav)
	got, ct, cached, err := SpeechKokoro(context.Background(), cfg, " "+text+" ", "", "", nil, "", " ")
	if err != nil || !cached || ct != "audio/wav" || string(got) != string(wav) {
		t.Fatalf("hit cached=%v ct=%s err=%v", cached, ct, err)
	}
	excited := append([]byte(nil), wav...)
	piperphrase.Store(piperphrase.Key{
		Model:   cfg.BackendPaths["kokoro_model"],
		Config:  cfg.BackendPaths["kokoro_voices"],
		Voice:   "af_heart",
		Scale:   "1.0000",
		Lang:    "en-us",
		Emotion: "excited",
		Text:    text,
	}, excited)
	_, _, cached, err = SpeechKokoro(context.Background(), cfg, text, "af_heart", "", nil, "", " Excited ")
	if err != nil || !cached {
		t.Fatalf("emotion hit cached=%v err=%v", cached, err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ct, rate, r, cached, err := SpeechKokoroStreamReader(ctx, cfg, text, "af_heart", "en-us", nil, "", "")
	if err != nil || !cached || rate != kokoro.SampleRate || !strings.Contains(ct, "audio/pcm") {
		t.Fatalf("stream hit cached=%v ct=%s rate=%d err=%v", cached, ct, rate, err)
	}
	defer r.Close()
	body, err := io.ReadAll(r)
	if err != nil || string(body) != string(pcm) {
		t.Fatalf("pcm %q err=%v", body, err)
	}
}

func writeSpeechFake(t *testing.T) (modelPath, voices string) {
	t.Helper()
	dir := t.TempDir()
	modelPath = filepath.Join(dir, "m.onnx")
	voices = filepath.Join(dir, "v.bin")
	if err := os.WriteFile(modelPath, []byte("x"), 0o644); err != nil {
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
	src := `import argparse, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--voices", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--default-voice", default="af_heart")
    args = ap.parse_args()
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            b = b'{"status":"ok","rate":24000}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode())
            b = json.dumps({"error": "voice %s not found" % (body.get("voice") or "")}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()

if __name__ == "__main__":
    main()
`
	if err := os.WriteFile(filepath.Join(pkg, "kokoro_serve.py"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ZEROLLAMA_REPO", repo)
	return modelPath, voices
}
