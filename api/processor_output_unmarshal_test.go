package api

import (
	"encoding/json"
	"testing"
)

func TestProcessorOutputUnmarshal_2dPixelValues(t *testing.T) {
	raw := `{"format":"processor_output","pixel_values":[[1,2],[3,4]],"image_grid_thw":[1,24,32]}`
	var po ProcessorOutput
	if err := json.Unmarshal([]byte(raw), &po); err != nil {
		t.Fatal(err)
	}
	if len(po.PixelValues) != 4 {
		t.Fatalf("pixel_values=%d vals=%v", len(po.PixelValues), po.PixelValues)
	}
}

func TestMessageUnmarshal_processorOutputInImages(t *testing.T) {
	raw := `{"role":"user","padded_input_ids":[1,2,3],"images":[{"format":"processor_output","pixel_values":[[1,2],[3,4]],"image_grid_thw":[1,24,32]}]}`
	var msg Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatal(err)
	}
	if len(msg.ProcessorOutputs) != 1 || len(msg.Images) != 0 {
		t.Fatalf("processor=%d images=%d", len(msg.ProcessorOutputs), len(msg.Images))
	}
	if len(msg.ProcessorOutputs[0].PixelValues) != 4 {
		t.Fatalf("pixel_values=%d vals=%v", len(msg.ProcessorOutputs[0].PixelValues), msg.ProcessorOutputs[0].PixelValues)
	}
}
