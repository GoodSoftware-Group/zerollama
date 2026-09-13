package cmd

import (
	"strings"
	"testing"
)

func TestDoctorCheckNgramStructured_Off(t *testing.T) {
	t.Setenv("ZEROLLAMA_SPEC_TYPE", "")
	t.Setenv("ZEROLLAMA_LLAMA_SPEC_TYPE", "")
	t.Setenv("ZEROLLAMA_ELIZA_NGRAM", "0")
	c := doctorCheckNgramStructured()
	if c.Status != "ok" || !strings.Contains(c.Name, "138") {
		t.Fatalf("%+v", c)
	}
}

func TestDoctorCheckNgramStructured_On(t *testing.T) {
	t.Setenv("ZEROLLAMA_SPEC_TYPE", "ngram-simple")
	t.Setenv("ZEROLLAMA_ELIZA_NGRAM", "0")
	c := doctorCheckNgramStructured()
	if c.Status != "warn" || !strings.Contains(c.Detail, "138") {
		t.Fatalf("%+v", c)
	}
}

func TestDoctorCheckHTTPConcurrency(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_PARALLEL_SLOTS", "4")
	c := doctorCheckHTTPConcurrency()
	if c.Status != "ok" || !strings.Contains(c.Detail, "slots≈4") {
		t.Fatalf("%+v", c)
	}
}
