package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestDoctorDurationNS(t *testing.T) {
	d, ok := doctorDurationNS(float64(1_500_000_000))
	if !ok || d != 1500*time.Millisecond {
		t.Fatalf("got %v ok=%v", d, ok)
	}
	if _, ok := doctorDurationNS(float64(0)); ok {
		t.Fatal("zero should fail")
	}
	if _, ok := doctorDurationNS(nil); ok {
		t.Fatal("nil should fail")
	}
}

func TestDoctorCheckLatencyReconciliation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(20 * time.Millisecond)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message":        map[string]any{"content": "ok"},
			"done":           true,
			"total_duration": float64(15_000_000), // 15ms server-side
		})
	}))
	t.Cleanup(srv.Close)

	c := doctorCheckLatencyReconciliation(srv.URL, doctorLoadedModel{Name: "m"})
	if c.Status != "ok" || !strings.Contains(c.Detail, "gap=") {
		t.Fatalf("%+v", c)
	}

	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2100 * time.Millisecond)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message":        map[string]any{"content": "ok"},
			"done":           true,
			"total_duration": float64(1_000_000), // 1ms server figure
		})
	}))
	t.Cleanup(slow.Close)
	c = doctorCheckLatencyReconciliation(slow.URL, doctorLoadedModel{Name: "m"})
	if c.Status != "warn" || !strings.Contains(c.Detail, "48") {
		t.Fatalf("want trap-48 gap warn: %+v", c)
	}

	missing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{"content": "ok"},
			"done":    true,
		})
	}))
	t.Cleanup(missing.Close)
	c = doctorCheckLatencyReconciliation(missing.URL, doctorLoadedModel{Name: "m"})
	if c.Status != "warn" || !strings.Contains(c.Detail, "total_duration") {
		t.Fatalf("want missing duration warn: %+v", c)
	}
}
