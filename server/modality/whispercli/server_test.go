package whispercli

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestPostInferenceSendsLiveFields(t *testing.T) {
	var gotFile, gotLang, gotBeam, gotBest, gotTS, gotFmt string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/inference" {
			http.NotFound(w, r)
			return
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Error(err)
			http.Error(w, err.Error(), 400)
			return
		}
		gotLang = r.FormValue("language")
		gotBeam = r.FormValue("beam_size")
		gotBest = r.FormValue("best_of")
		gotTS = r.FormValue("no_timestamps")
		gotFmt = r.FormValue("response_format")
		f, _, err := r.FormFile("file")
		if err != nil {
			t.Error(err)
			http.Error(w, err.Error(), 400)
			return
		}
		raw, _ := io.ReadAll(f)
		gotFile = string(raw)
		_ = json.NewEncoder(w).Encode(map[string]string{"text": "  hello  "})
	}))
	defer srv.Close()

	text, err := PostInference(context.Background(), srv.Client(), srv.URL, []byte("RIFF-wav"), "fr")
	if err != nil {
		t.Fatal(err)
	}
	if text != "hello" {
		t.Fatalf("text %q", text)
	}
	if gotFile != "RIFF-wav" || gotLang != "fr" || gotBeam != "1" || gotBest != "1" || gotTS != "true" || gotFmt != "json" {
		t.Fatalf("file=%q lang=%q beam=%q best=%q ts=%q fmt=%q", gotFile, gotLang, gotBeam, gotBest, gotTS, gotFmt)
	}
}

func TestPostInferenceOmitsEmptyLanguage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseMultipartForm(1 << 20)
		if _, ok := r.MultipartForm.Value["language"]; ok {
			t.Fatal("language was set")
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"text": "ok"})
	}))
	defer srv.Close()
	if _, err := PostInference(context.Background(), srv.Client(), srv.URL, []byte("x"), ""); err != nil {
		t.Fatal(err)
	}
}

func TestPostInferenceRejectsOldBinary(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"failed to read audio data"}`))
	}))
	defer srv.Close()
	_, err := PostInference(context.Background(), srv.Client(), srv.URL, []byte("RIFF"), "en")
	if !errors.Is(err, ErrBadAudio) {
		t.Fatalf("err=%v", err)
	}
}

func TestPostInferenceRejectsInvalidRequestBody(t *testing.T) {
	// whisper-server's error_handler replaces the JSON with this plain text.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("Invalid request"))
	}))
	defer srv.Close()
	_, err := PostInference(context.Background(), srv.Client(), srv.URL, []byte("RIFF"), "en")
	if !errors.Is(err, ErrBadAudio) {
		t.Fatalf("err=%v", err)
	}
}

func TestServerPostsTwoClipsOnOneProcess(t *testing.T) {
	dir := t.TempDir()
	script := filepath.Join(dir, "whisper-server")
	body := `#!/usr/bin/env python3
import json, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

port = None
for i, a in enumerate(sys.argv):
    if a == "--port":
        port = int(sys.argv[i + 1])
n = 0
lock = threading.Lock()

class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        self.send_response(404)
        self.end_headers()
    def do_POST(self):
        global n
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        with lock:
            n += 1
            cur = n
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"text": f"clip{cur}"}).encode())

HTTPServer(("127.0.0.1", port), H).serve_forever()
`
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(dir, "model.bin")
	if err := os.WriteFile(model, []byte("m"), 0o644); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	proc, err := StartServer(ctx, script, model)
	if err != nil {
		t.Fatal(err)
	}
	defer proc.Close()
	if err := proc.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	if !proc.Ready() {
		t.Fatal("not ready")
	}

	a, err := proc.Transcribe(ctx, []byte("RIFF-a"), "en")
	if err != nil {
		t.Fatal(err)
	}
	b, err := proc.Transcribe(ctx, []byte("RIFF-b"), "fr")
	if err != nil {
		t.Fatal(err)
	}
	if a != "clip1" || b != "clip2" {
		t.Fatalf("got %q %q", a, b)
	}
}

func TestServerRetiresOnBadAudio(t *testing.T) {
	dir := t.TempDir()
	script := filepath.Join(dir, "whisper-server")
	body := `#!/usr/bin/env python3
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = None
for i, a in enumerate(sys.argv):
    if a == "--port":
        port = int(sys.argv[i + 1])

class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        self.send_response(400)
        self.end_headers()
        self.wfile.write(b'Invalid request')

HTTPServer(("127.0.0.1", port), H).serve_forever()
`
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(dir, "model.bin")
	if err := os.WriteFile(model, []byte("m"), 0o644); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	proc, err := StartServer(ctx, script, model)
	if err != nil {
		t.Fatal(err)
	}
	defer proc.Close()
	if err := proc.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	_, err = proc.Transcribe(ctx, []byte("RIFF"), "en")
	if !errors.Is(err, ErrBadAudio) {
		t.Fatalf("err=%v", err)
	}
	if !proc.Retired() {
		t.Fatal("expected retired")
	}
}

func TestServerRestartsAfterCrash(t *testing.T) {
	dir := t.TempDir()
	script := filepath.Join(dir, "whisper-server")
	countPath := filepath.Join(dir, "starts")
	body := `#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
with open(` + strconv.Quote(countPath) + `, "a") as f:
    f.write("1\n")
port=None
for i,a in enumerate(sys.argv):
    if a=="--port":
        port=int(sys.argv[i+1])
n=0
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')
    def do_POST(self):
        global n
        length=int(self.headers.get("Content-Length","0"))
        self.rfile.read(length)
        n+=1
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"text":"c%d"%n}).encode())
HTTPServer(("127.0.0.1", port), H).serve_forever()
`
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(dir, "m.bin")
	_ = os.WriteFile(model, []byte("m"), 0o644)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	proc, err := StartServer(ctx, script, model)
	if err != nil {
		t.Fatal(err)
	}
	defer proc.Close()
	if err := proc.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	first, err := proc.Transcribe(ctx, []byte("a"), "")
	if err != nil || first != "c1" {
		t.Fatalf("first=%q err=%v", first, err)
	}
	proc.mu.Lock()
	cmd := proc.cmd
	done := proc.done
	proc.mu.Unlock()
	if cmd == nil || cmd.Process == nil {
		t.Fatal("missing process")
	}
	_ = cmd.Process.Kill()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("process did not exit")
	}
	if proc.Ready() {
		t.Fatal("still ready after crash")
	}
	if err := proc.Restart(ctx); err != nil {
		t.Fatal(err)
	}
	if err := proc.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	second, err := proc.Transcribe(ctx, []byte("b"), "")
	if err != nil || second != "c1" {
		t.Fatalf("second=%q err=%v", second, err)
	}
	raw, err := os.ReadFile(countPath)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(string(raw), "1"); got != 2 {
		t.Fatalf("starts=%d raw=%q", got, raw)
	}
}

func TestHealthRejectsLooseBody(t *testing.T) {
	if healthOK([]byte(`{"ok":true}`)) {
		t.Fatal("accepted loose body")
	}
	if !healthOK([]byte(`{"status":"ok"}`)) {
		t.Fatal("rejected status ok")
	}
}

func TestLookServerIgnoresBareCwdName(t *testing.T) {
	dir := t.TempDir()
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(cwd)
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile("whisper-server-not-on-path", []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := LookServer("whisper-server-not-on-path"); got != "" {
		t.Fatalf("cwd bare name resolved to %q", got)
	}
}

func TestLookServerFindsScript(t *testing.T) {
	dir := t.TempDir()
	script := filepath.Join(dir, "whisper-server")
	if err := os.WriteFile(script, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := LookServer(script); got != script {
		t.Fatalf("got %q", got)
	}
	if got := LookServer(filepath.Join(dir, "missing")); got != "" {
		t.Fatalf("got %q", got)
	}
}

func TestStartServerUsesFreePort(t *testing.T) {
	dir := t.TempDir()
	script := filepath.Join(dir, "whisper-server")
	flag := filepath.Join(dir, "port")
	body := `#!/usr/bin/env python3
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port=None
for i,a in enumerate(sys.argv):
    if a=="--port":
        port=int(sys.argv[i+1])
open(` + strconv.Quote(flag) + `, "w").write(str(port))
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')
HTTPServer(("127.0.0.1", port), H).serve_forever()
`
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(dir, "m.bin")
	_ = os.WriteFile(model, []byte("m"), 0o644)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	proc, err := StartServer(ctx, script, model)
	if err != nil {
		t.Fatal(err)
	}
	defer proc.Close()
	if err := proc.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(flag)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(strings.TrimSpace(string(raw)))
	if err != nil {
		t.Fatal(err)
	}
	for _, banned := range []int{11434, 8081, 8080} {
		if port == banned {
			t.Fatalf("bound production port %d", port)
		}
	}
	host, p, err := net.SplitHostPort(strings.TrimPrefix(proc.baseURL, "http://"))
	if err != nil || host != "127.0.0.1" {
		t.Fatalf("baseURL %s", proc.baseURL)
	}
	if p != strconv.Itoa(port) {
		t.Fatalf("port file %d url %s", port, p)
	}
}

func TestServerStartsOnceForTwoClips(t *testing.T) {
	dir := t.TempDir()
	script := filepath.Join(dir, "whisper-server")
	countPath := filepath.Join(dir, "starts")
	body := `#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
with open(` + strconv.Quote(countPath) + `, "a") as f:
    f.write("1\n")
port=None
for i,a in enumerate(sys.argv):
    if a=="--port":
        port=int(sys.argv[i+1])
n=0
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')
    def do_POST(self):
        global n
        length=int(self.headers.get("Content-Length","0"))
        self.rfile.read(length)
        n+=1
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"text":"c%d"%n}).encode())
HTTPServer(("127.0.0.1", port), H).serve_forever()
`
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(dir, "m.bin")
	_ = os.WriteFile(model, []byte("m"), 0o644)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	proc, err := StartServer(ctx, script, model)
	if err != nil {
		t.Fatal(err)
	}
	defer proc.Close()
	if err := proc.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := proc.Transcribe(ctx, []byte("a"), ""); err != nil {
		t.Fatal(err)
	}
	if _, err := proc.Transcribe(ctx, []byte("b"), ""); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(countPath)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(string(raw), "1"); got != 1 {
		t.Fatalf("starts=%d raw=%q", got, raw)
	}
}
