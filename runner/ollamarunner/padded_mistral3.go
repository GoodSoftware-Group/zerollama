package ollamarunner

import (
	"fmt"

	"github.com/ollama/ollama/tokenizer"
)

// Mistral3VisionTokens holds Pixtral/Mistral3 vision token ids for padded inject.
type Mistral3VisionTokens struct {
	Img   int
	Break int
	End   int
}

func (s *Server) mistral3VisionTokens() (Mistral3VisionTokens, error) {
	tok, ok := s.model.(tokenizer.Tokenizer)
	if !ok {
		return Mistral3VisionTokens{}, fmt.Errorf("mistral3 padded inject requires a tokenizer")
	}
	slots := Mistral3VisionTokens{Img: 10, Break: 12, End: 13}
	if id, err := s.encodePlaceholderToken(tok, "[IMG]"); err == nil && id != 0 {
		slots.Img = id
	}
	if id, err := s.encodePlaceholderToken(tok, "[IMG_BREAK]"); err == nil && id != 0 {
		slots.Break = id
	}
	if id, err := s.encodePlaceholderToken(tok, "[IMG_END]"); err == nil && id != 0 {
		slots.End = id
	}
	return slots, nil
}

func isMistral3ImageInjectStart(tokens []int, i int, slots Mistral3VisionTokens) bool {
	if i >= len(tokens) || tokens[i] != slots.Img {
		return false
	}
	if i > 0 && (tokens[i-1] == slots.Img || tokens[i-1] == slots.Break) {
		return false
	}
	return mistral3ImageBlockEndIndex(tokens, i, slots) > i
}

func mistral3ImageBlockEndIndex(tokens []int, start int, slots Mistral3VisionTokens) int {
	for j := start; j < len(tokens); j++ {
		if tokens[j] == slots.End {
			return j
		}
	}
	return start
}
