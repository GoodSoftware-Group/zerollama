package cmd

import (
	"runtime"
	"strings"
	"testing"
)

func TestDoctorCheckSpeculativeUMA_HighProduct(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_PARALLEL_SLOTS", "32")
	t.Setenv("ZEROLLAMA_DRAFT_MAX", "15")
	t.Setenv("ZEROLLAMA_ANE_DRAFT", "1")
	t.Setenv("LLAMA_DRAFT_MODEL", "")
	t.Setenv("ZEROLLAMA_SPEC_TYPE", "")
	c := doctorCheckSpeculativeUMA()
	if !strings.Contains(c.Detail, "product=480") {
		t.Fatalf("detail=%s", c.Detail)
	}
	if runtime.GOOS == "darwin" && c.Status != "warn" {
		t.Fatalf("want warn on Darwin high product: %+v", c)
	}
	if runtime.GOOS != "darwin" && c.Status != "ok" {
		t.Fatalf("non-UMA hosts stay ok: %+v", c)
	}
}

func TestDoctorCheckSpeculativeUMA_Quiet(t *testing.T) {
	t.Setenv("ZEROLLAMA_LLAMA_PARALLEL_SLOTS", "1")
	t.Setenv("ZEROLLAMA_DRAFT_MAX", "4")
	t.Setenv("ZEROLLAMA_ANE_DRAFT", "0")
	t.Setenv("LLAMA_DRAFT_MODEL", "")
	t.Setenv("ZEROLLAMA_SPEC_TYPE", "")
	c := doctorCheckSpeculativeUMA()
	if c.Status != "ok" {
		t.Fatalf("%+v", c)
	}
	if !strings.Contains(c.Name, "98") {
		t.Fatalf("name=%s", c.Name)
	}
}

func TestDoctorEnvInt(t *testing.T) {
	t.Setenv("ZEROLLAMA_DRAFT_MAX", "16")
	if doctorEnvInt("ZEROLLAMA_DRAFT_MAX", 0) != 16 {
		t.Fatal("parse")
	}
}
