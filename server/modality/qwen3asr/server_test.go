package qwen3asr

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestPostTranscribeSendsLanguage(t *testing.T) {
	var gotFile, gotLang, gotCT string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/transcribe" {
			http.NotFound(w, r)
			return
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Error(err)
			http.Error(w, err.Error(), 400)
			return
		}
		gotLang = r.FormValue("language")
		f, hdr, err := r.FormFile("file")
		if err != nil {
			t.Error(err)
			http.Error(w, err.Error(), 400)
			return
		}
		gotCT = hdr.Header.Get("Content-Type")
		raw, _ := io.ReadAll(f)
		gotFile = string(raw)
		_ = json.NewEncoder(w).Encode(map[string]string{"text": "  hi  "})
	}))
	defer srv.Close()

	text, err := PostTranscribe(context.Background(), srv.Client(), srv.URL, []byte("RIFF"), "English")
	if err != nil {
		t.Fatal(err)
	}
	if text != "hi" || gotFile != "RIFF" || gotLang != "English" {
		t.Fatalf("text=%q file=%q lang=%q", text, gotFile, gotLang)
	}
	if gotCT != "application/octet-stream" {
		t.Fatalf("file content-type %q", gotCT)
	}
}

func TestPostTranscribeOmitsEmptyLanguage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseMultipartForm(1 << 20)
		if _, ok := r.MultipartForm.Value["language"]; ok {
			t.Fatal("language was set")
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"text": "ok"})
	}))
	defer srv.Close()
	if _, err := PostTranscribe(context.Background(), srv.Client(), srv.URL, []byte("x"), ""); err != nil {
		t.Fatal(err)
	}
}

func TestPostTranscribeBusy(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"busy"}`))
	}))
	defer srv.Close()
	_, err := PostTranscribe(context.Background(), srv.Client(), srv.URL, []byte("x"), "")
	if !errors.Is(err, ErrBusy) {
		t.Fatalf("err=%v", err)
	}
}

func TestParseHealthBusy(t *testing.T) {
	ok, busy := parseHealth([]byte(`{"status":"ok"}`))
	if !ok || busy != 0 {
		t.Fatalf("ok=%v busy=%s", ok, busy)
	}
	ok, busy = parseHealth([]byte(`{"status":"ok","busy_ms":1500}`))
	if !ok || busy != 1500*time.Millisecond {
		t.Fatalf("ok=%v busy=%s", ok, busy)
	}
	if ok, _ := parseHealth([]byte(`{"status":"loading"}`)); ok {
		t.Fatal("loading counted as ready")
	}
}

func TestLimitWriterKeepsCopyAlive(t *testing.T) {
	var buf bytes.Buffer
	var mu sync.Mutex
	w := &limitWriter{dst: &buf, mu: &mu, max: 4}
	n, err := io.Copy(w, strings.NewReader(strings.Repeat("abcdefghij", 20)))
	if err != nil {
		t.Fatal(err)
	}
	if n != 200 {
		t.Fatalf("copied %d", n)
	}
	if buf.String() != "abcd" {
		t.Fatalf("kept %q", buf.String())
	}
}

func TestPostTranscribeUnavailable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"no model"}`))
	}))
	defer srv.Close()
	_, err := PostTranscribe(context.Background(), srv.Client(), srv.URL, []byte("x"), "")
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("err=%v", err)
	}
}

func TestStartServerMissingDir(t *testing.T) {
	_, err := StartServer(context.Background(), filepath.Join(t.TempDir(), "missing"))
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("err=%v", err)
	}
}

func TestServerPostsTwoClipsOnOneProcess(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "state.txt")
	weights := filepath.Join(dir, "weights")
	if err := os.Mkdir(weights, 0o755); err != nil {
		t.Fatal(err)
	}
	// PYTHONPATH=$ZEROLLAMA_REPO/runtime → import runtime.asr.qwen3_serve
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
	src := `import argparse, json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    with open(os.environ["FAKE_STATE"], "a") as f:
        f.write("start\n")
    n = {"i": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            b = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        def do_POST(self):
            if self.path != "/transcribe":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            n["i"] += 1
            b = json.dumps({"text": "c%d" % n["i"]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    ThreadingHTTPServer((args.host, args.port), H).serve_forever()

if __name__ == "__main__":
    main()
`
	if err := os.WriteFile(filepath.Join(pkg, "qwen3_serve.py"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ZEROLLAMA_REPO", repo)
	t.Setenv("FAKE_STATE", state)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	proc, err := StartServer(ctx, weights)
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
	t1, err := proc.Transcribe(ctx, []byte("a"), "English")
	if err != nil {
		t.Fatal(err)
	}
	t2, err := proc.Transcribe(ctx, []byte("b"), "")
	if err != nil {
		t.Fatal(err)
	}
	if t1 != "c1" || t2 != "c2" {
		t.Fatalf("got %q %q", t1, t2)
	}
	raw, _ := os.ReadFile(state)
	if strings.Count(string(raw), "start") != 1 {
		t.Fatalf("starts=%q", raw)
	}
}

func TestSharedFallsBackWhenDirMissing(t *testing.T) {
	_, err := Transcribe(context.Background(), filepath.Join(t.TempDir(), "nope"), []byte("x"), "en")
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("err=%v", err)
	}
}
