package ollamarunner

import (
	"fmt"

	"github.com/ollama/ollama/tokenizer"
)

// DeepseekOcrVisionTokens holds DeepSeek-OCR image placeholder token id.
type DeepseekOcrVisionTokens struct {
	Image int
}

func (s *Server) deepseekOcrVisionTokens() (DeepseekOcrVisionTokens, error) {
	tok, ok := s.model.(tokenizer.Tokenizer)
	if !ok {
		return DeepseekOcrVisionTokens{}, fmt.Errorf("deepseekocr padded inject requires a tokenizer")
	}
	slots := DeepseekOcrVisionTokens{Image: 128815}
	if id, err := s.encodePlaceholderToken(tok, "<image>"); err == nil && id != 0 {
		slots.Image = id
	}
	return slots, nil
}
