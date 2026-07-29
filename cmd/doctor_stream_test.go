package cmd

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDoctorCountStreamDeltaKeys(t *testing.T) {
	body := strings.NewReader("" +
		"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"Oslo\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n\n" +
		"data: [DONE]\n\n")
	counts, err := doctorCountStreamDeltaKeys(body)
	if err != nil {
		t.Fatal(err)
	}
	if counts["content"] != 1 || counts["role"] != 1 {
		t.Fatalf("got %v", counts)
	}
}

func TestDoctorCheckStreamContent(t *testing.T) {
	ok := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"Oslo\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	t.Cleanup(ok.Close)
	c := doctorCheckStreamContent(ok.URL, doctorLoadedModel{Name: "m", SupportsThinking: true})
	if c.Status != "ok" || !strings.Contains(c.Detail, "content") {
		t.Fatalf("%+v", c)
	}

	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"reasoning\":\"Oslo\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	t.Cleanup(bad.Close)
	c = doctorCheckStreamContent(bad.URL, doctorLoadedModel{Name: "m", SupportsThinking: true})
	if c.Status != "warn" || !strings.Contains(c.Detail, "23") {
		t.Fatalf("want trap-23 warn: %+v", c)
	}
}
