package qwen3next

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/ml/nn"
)

func TestValidateRecurrentLayerRequiresSSMDT(t *testing.T) {
	m := &Model{
		Layers: []Layer{{
			Operator: &GatedDeltaNet{
				SSMQKV:     &nn.Linear{},
				SSMQKVGate: &nn.Linear{},
				SSMBeta:    &nn.Linear{},
				SSMAlpha:   &nn.Linear{},
			},
		}},
		Options: &Options{
			isRecurrent: []bool{true},
		},
	}

	err := m.Validate()
	if err == nil {
		t.Fatal("Validate() expected error, got nil")
	}
	if !strings.Contains(err.Error(), "missing ssm_dt") {
		t.Fatalf("unexpected error = %v", err)
	}
}

func TestValidateRecurrentSSMInAccepted(t *testing.T) {
	// When SSMIn is set, Validate must not reject the layer for missing
	// attn_qkv/attn_gate. It should fail later on missing ssm_dt.
	m := &Model{
		Layers: []Layer{{
			Operator: &GatedDeltaNet{
				SSMIn:    &nn.Linear{},
				SSMBeta:  &nn.Linear{},
				SSMAlpha: &nn.Linear{},
			},
		}},
		Options: &Options{
			isRecurrent: []bool{true},
		},
	}

	err := m.Validate()
	if err == nil {
		t.Fatal("Validate() expected error, got nil")
	}
	if strings.Contains(err.Error(), "missing attn_qkv/attn_gate") {
		t.Fatalf("Validate() should not fail on attn_qkv/attn_gate when SSMIn is set, got: %v", err)
	}
}

func TestEmptyMTPClearedOnValidate(t *testing.T) {
	m := &Model{
		MTP:    newMTPHead(1, false),
		Layers: []Layer{{Operator: &FullAttention{}}},
		Options: &Options{
			isRecurrent: []bool{false},
		},
	}
	if err := m.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	if m.MTP != nil {
		t.Fatal("empty MTP head should be dropped")
	}
}

func TestMTPHeadLoadedRequiresFCAndQuery(t *testing.T) {
	h := newMTPHead(1, true)
	if h.loaded() {
		t.Fatal("unbound MTP should not report loaded")
	}
	h.FC = &nn.Linear{}
	h.ENorm = &nn.RMSNorm{}
	h.HNorm = &nn.RMSNorm{}
	if h.loaded() {
		t.Fatal("MTP without weights should not report loaded")
	}
}

func TestCausalTrunk(t *testing.T) {
	attn := &Model{Options: &Options{isRecurrent: []bool{false, false}}}
	if !attn.CausalTrunk() {
		t.Fatal("dense full-attn trunk should be causal-only")
	}
	hybrid := &Model{Options: &Options{isRecurrent: []bool{true, false}}}
	if hybrid.CausalTrunk() {
		t.Fatal("GDN layer must disable causal-only accept")
	}
}

func TestValidateNonRecurrentSkipsLinearChecks(t *testing.T) {
	m := &Model{
		Layers: []Layer{{Operator: &FullAttention{}}},
		Options: &Options{
			isRecurrent: []bool{false},
		},
	}

	if err := m.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
}
