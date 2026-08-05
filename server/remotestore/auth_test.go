package remotestore

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestAuthSignVerifyRoundTrip(t *testing.T) {
	t.Parallel()
	a, err := NewAuth("test-secret")
	if err != nil {
		t.Fatal(err)
	}
	fixed := time.Date(2026, 8, 4, 12, 0, 0, 0, time.UTC)
	a.Now = func() time.Time { return fixed }

	req := httptest.NewRequest(http.MethodGet, "http://example/v1/blob/sha256-abc", nil)
	if err := a.SignRequest(req, nil); err != nil {
		t.Fatal(err)
	}
	if err := a.VerifyRequest(req, nil); err != nil {
		t.Fatalf("verify: %v", err)
	}
}

func TestAuthRejectsTampered(t *testing.T) {
	t.Parallel()
	a, err := NewAuth("test-secret")
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodGet, "http://example/v1/blob/x", nil)
	if err := a.SignRequest(req, nil); err != nil {
		t.Fatal(err)
	}
	req.Header.Set(AuthHeader, req.Header.Get(AuthHeader)+"dead")
	if err := a.VerifyRequest(req, nil); err != ErrAuthInvalid {
		t.Fatalf("want ErrAuthInvalid, got %v", err)
	}
}

func TestAuthRejectsExpired(t *testing.T) {
	t.Parallel()
	a, err := NewAuth("test-secret")
	if err != nil {
		t.Fatal(err)
	}
	past := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	a.Now = func() time.Time { return past }
	req := httptest.NewRequest(http.MethodHead, "http://example/v1/capability", nil)
	if err := a.SignRequest(req, nil); err != nil {
		t.Fatal(err)
	}
	a.Now = time.Now
	if err := a.VerifyRequest(req, nil); err != ErrAuthExpired {
		t.Fatalf("want ErrAuthExpired, got %v", err)
	}
}

func TestAuthRejectsMissing(t *testing.T) {
	t.Parallel()
	a, err := NewAuth("s")
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodGet, "http://example/", nil)
	if err := a.VerifyRequest(req, nil); err != ErrAuthMissing {
		t.Fatalf("want ErrAuthMissing, got %v", err)
	}
}

func TestAuthBodyMatters(t *testing.T) {
	t.Parallel()
	a, err := NewAuth("s")
	if err != nil {
		t.Fatal(err)
	}
	body := []byte("hello")
	req := httptest.NewRequest(http.MethodPut, "http://example/v1/blob/sha256-x", nil)
	if err := a.SignRequest(req, body); err != nil {
		t.Fatal(err)
	}
	if err := a.VerifyRequest(req, body); err != nil {
		t.Fatal(err)
	}
	if err := a.VerifyRequest(req, []byte("other")); err != ErrAuthInvalid {
		t.Fatalf("want ErrAuthInvalid for different body, got %v", err)
	}
}

func TestNewAuthNoSecret(t *testing.T) {
	t.Setenv("ZEROLLAMA_STORAGE_SECRET", "")
	t.Setenv("ZEROLLAMA_STORAGE_SECRET_FILE", "")
	if _, err := NewAuth(""); err != ErrAuthNoSecret {
		t.Fatalf("want ErrAuthNoSecret, got %v", err)
	}
}

func TestAuthMiddlewareRejectsUnsignedPUT(t *testing.T) {
	t.Parallel()
	a, err := NewAuth("mw-secret")
	if err != nil {
		t.Fatal(err)
	}
	called := false
	h := a.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}))
	req := httptest.NewRequest(http.MethodPut, "http://example/v1/blob/x", bytes.NewReader([]byte("body")))
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("want 401, got %d", rr.Code)
	}
	if called {
		t.Fatal("handler should not run without auth")
	}
}

func TestAuthMiddlewareAllowsSignedPUT(t *testing.T) {
	t.Parallel()
	a, err := NewAuth("mw-secret")
	if err != nil {
		t.Fatal(err)
	}
	body := []byte("payload")
	var gotBody []byte
	h := a.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusCreated)
	}))
	req := httptest.NewRequest(http.MethodPut, "http://example/v1/manifest/h/n/m/t", bytes.NewReader(body))
	if err := a.SignRequest(req, body); err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusCreated {
		t.Fatalf("want 201, got %d %s", rr.Code, rr.Body.String())
	}
	if string(gotBody) != string(body) {
		t.Fatalf("body not restored: %q", gotBody)
	}
}
