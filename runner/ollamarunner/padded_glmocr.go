package ollamarunner

import (
	"fmt"

	"github.com/ollama/ollama/tokenizer"
)

func (s *Server) glmocrVisionTokens() (Lfm2VisionTokens, error) {
	tok, ok := s.model.(tokenizer.Tokenizer)
	if !ok {
		return Lfm2VisionTokens{}, fmt.Errorf("glmocr padded inject requires a tokenizer")
	}
	// GGUF defaults from model/models/glmocr/model.go
	slots := Lfm2VisionTokens{Image: 59280, Start: 59256, End: 59257}
	if id, err := s.encodePlaceholderToken(tok, "<|image_start|>"); err == nil && id != 0 {
		slots.Start = id
	}
	if id, err := s.encodePlaceholderToken(tok, "<|image_end|>"); err == nil && id != 0 {
		slots.End = id
	}
	if id, err := s.encodePlaceholderToken(tok, "<|image|>"); err == nil && id != 0 {
		slots.Image = id
	}
	slots.UseBlock = slots.Start != 0 && slots.End != 0
	return slots, nil
}
