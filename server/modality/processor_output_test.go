package modality

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestPreflightProcessorOutputs_requiresPaddedIDs(t *testing.T) {
	req := &api.ChatRequest{Messages: []api.Message{{
		ProcessorOutputs: []api.ProcessorOutput{{
			PixelValues:  []float32{1, 2},
			ImageGridTHW: []int{1, 2, 2},
		}},
	}}}
	if err := PreflightProcessorOutputs(req); err == nil {
		t.Fatal("expected error")
	}
}

func TestAppendProcessorOutputsToLLM(t *testing.T) {
	msg := api.Message{
		ProcessorOutputs: []api.ProcessorOutput{{
			PixelValues:  []float32{1, 2, 3},
			ImageGridTHW: []int{1, 4, 4},
		}},
	}
	out := AppendProcessorOutputsToLLM(msg, nil)
	if len(out) != 1 || !out[0].HasProcessorOutput() || len(out[0].GridTHW) != 3 {
		t.Fatalf("got %+v", out)
	}
}
