package storaged

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/server/remotestore"
)

func TestHandlePullStreamSuccess(t *testing.T) {
	auth, err := remotestore.NewAuth("pull-test-secret")
	if err != nil {
		t.Fatal(err)
	}
	s := New(t.TempDir(), auth)
	var gotModel string
	s.Pull = func(ctx context.Context, modelName string, opts PullOptions, fn func(ProgressEvent)) error {
		gotModel = modelName
		fn(ProgressEvent{Status: "pulling manifest"})
		fn(ProgressEvent{Status: "downloading", Digest: "sha256:abc", Total: 100, Completed: 50})
		fn(ProgressEvent{Status: "downloading", Digest: "sha256:abc", Total: 100, Completed: 100})
		return nil
	}

	body := `{"model":"llama3.2:3b"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/pull", strings.NewReader(body))
	if err := auth.SignRequest(req, []byte(body)); err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	if rr.Code != 200 {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	if gotModel != "llama3.2:3b" {
		t.Fatalf("model=%q", gotModel)
	}
	dec := json.NewDecoder(rr.Body)
	var events []ProgressEvent
	for {
		var ev ProgressEvent
		if err := dec.Decode(&ev); err == io.EOF {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		events = append(events, ev)
	}
	if len(events) < 2 {
		t.Fatalf("events=%d %#v", len(events), events)
	}
	if events[len(events)-1].Status != "success" {
		t.Fatalf("last=%#v", events[len(events)-1])
	}
}

func TestHandlePullErrorLine(t *testing.T) {
	auth, err := remotestore.NewAuth("pull-test-secret")
	if err != nil {
		t.Fatal(err)
	}
	s := New(t.TempDir(), auth)
	s.Pull = func(ctx context.Context, modelName string, opts PullOptions, fn func(ProgressEvent)) error {
		return errors.New("registry 404")
	}
	body := `{"model":"missing:latest"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/pull", strings.NewReader(body))
	if err := auth.SignRequest(req, []byte(body)); err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	if rr.Code != 200 {
		t.Fatalf("status=%d", rr.Code)
	}
	var last ProgressEvent
	dec := json.NewDecoder(rr.Body)
	for {
		var ev ProgressEvent
		if err := dec.Decode(&ev); err == io.EOF {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		last = ev
	}
	if last.Status != "error" || last.Error == "" {
		t.Fatalf("last=%#v", last)
	}
}

func TestHandlePullNotImplemented(t *testing.T) {
	auth, err := remotestore.NewAuth("pull-test-secret")
	if err != nil {
		t.Fatal(err)
	}
	s := New(t.TempDir(), auth) // Pull unset
	body := `{"model":"x"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/pull", strings.NewReader(body))
	if err := auth.SignRequest(req, []byte(body)); err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	if rr.Code != http.StatusNotImplemented {
		t.Fatalf("status=%d", rr.Code)
	}
}

func TestHandlePullSync(t *testing.T) {
	auth, err := remotestore.NewAuth("pull-test-secret")
	if err != nil {
		t.Fatal(err)
	}
	s := New(t.TempDir(), auth)
	s.Pull = func(ctx context.Context, modelName string, opts PullOptions, fn func(ProgressEvent)) error {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(10 * time.Millisecond):
			return nil
		}
	}
	body := `{"model":"tiny:latest","stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/v1/pull", strings.NewReader(body))
	if err := auth.SignRequest(req, []byte(body)); err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	if rr.Code != 200 {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var ev ProgressEvent
	if err := json.Unmarshal(rr.Body.Bytes(), &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Status != "success" {
		t.Fatalf("%#v", ev)
	}
}

func TestCapabilityPullFlag(t *testing.T) {
	auth, err := remotestore.NewAuth("pull-test-secret")
	if err != nil {
		t.Fatal(err)
	}
	s := New(t.TempDir(), auth)
	s.Pull = func(context.Context, string, PullOptions, func(ProgressEvent)) error { return nil }
	req := httptest.NewRequest(http.MethodGet, remotestore.CapabilityPath, nil)
	if err := auth.SignRequest(req, nil); err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	var cap remotestore.Capability
	if err := json.Unmarshal(rr.Body.Bytes(), &cap); err != nil {
		t.Fatal(err)
	}
	if !cap.Pull {
		t.Fatal("expected pull=true")
	}
}
