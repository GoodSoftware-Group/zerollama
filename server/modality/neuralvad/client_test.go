package neuralvad

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestStartServerMissingFile(t *testing.T) {
	_, err := StartServer(context.Background(), filepath.Join(t.TempDir(), "missing.onnx"))
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestServerScoresTwoSessions(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "silero.onnx")
	if err := os.WriteFile(model, []byte("fake"), 0o644); err != nil {
		t.Fatal(err)
	}
	repo := filepath.Join(dir, "repo")
	pkg := filepath.Join(repo, "runtime", "runtime", "asr")
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
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    n = {"i": 0}
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            b = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            if self.path == "/reset":
                b = b'{"status":"ok"}'
            else:
                n["i"] += 1
                b = json.dumps({"last": n["i"]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()

if __name__ == "__main__":
    main()
`
	if err := os.WriteFile(filepath.Join(pkg, "vad_serve.py"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ZEROLLAMA_REPO", repo)
	t.Setenv("OLLAMA_NEURAL_VAD", model)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	proc, err := StartServer(ctx, model)
	if err != nil {
		t.Fatal(err)
	}
	defer proc.Close()
	if err := proc.Wait(ctx); err != nil {
		t.Fatalf("wait: %v stderr=%s", err, proc.stderr())
	}
	port := strings.TrimPrefix(proc.baseURL, "http://127.0.0.1:")
	if p, _ := strconv.Atoi(port); p == 11434 || p == 8081 || p == 8080 {
		t.Fatalf("production port %d", p)
	}
	a, err := proc.Score(ctx, "a", []byte{0, 0}, 16000, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	b, err := proc.Score(ctx, "b", []byte{0, 0}, 16000, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if a != 1 || b != 2 {
		t.Fatalf("a=%d b=%d", a, b)
	}
	if err := proc.Reset(ctx, "a"); err != nil {
		t.Fatal(err)
	}
}
