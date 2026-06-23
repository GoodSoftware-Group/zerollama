package ollamarunner

import (
	"fmt"

	"github.com/ollama/ollama/tokenizer"
)

// Lfm2VisionTokens holds runtime-resolved LFM2 vision token ids for padded inject.
type Lfm2VisionTokens struct {
	Image    int
	Start    int
	End      int
	UseBlock bool
}

func (s *Server) lfm2VisionTokens() (Lfm2VisionTokens, error) {
	tok, ok := s.model.(tokenizer.Tokenizer)
	if !ok {
		return Lfm2VisionTokens{}, fmt.Errorf("lfm2 padded inject requires a tokenizer")
	}
	slots := Lfm2VisionTokens{Image: 396}
	if id, err := s.encodePlaceholderToken(tok, "<image>"); err == nil && id != 0 {
		slots.Image = id
	}
	start, err := s.encodePlaceholderToken(tok, "<|image_start|>")
	if err != nil {
		return Lfm2VisionTokens{}, err
	}
	end, err := s.encodePlaceholderToken(tok, "<|image_end|>")
	if err != nil {
		return Lfm2VisionTokens{}, err
	}
	slots.Start = start
	slots.End = end
	slots.UseBlock = start != 0 && end != 0
	return slots, nil
}

func (s *Server) encodePlaceholderToken(tok tokenizer.Tokenizer, placeholder string) (int, error) {
	toks, err := tok.Encode(placeholder, false)
	if err != nil {
		return 0, err
	}
	if len(toks) == 0 {
		return 0, nil
	}
	return int(toks[len(toks)-1]), nil
}

func isLfm2ImageBlockStart(tokens []int, i int, slots Lfm2VisionTokens) bool {
	if !slots.UseBlock || i >= len(tokens) || tokens[i] != slots.Start {
		return false
	}
	return lfm2ImageBlockEndIndex(tokens, i, slots.End) > i
}

func lfm2ImageBlockEndIndex(tokens []int, start int, endToken int) int {
	if start >= len(tokens) || endToken == 0 {
		return start
	}
	for j := start + 1; j < len(tokens); j++ {
		if tokens[j] == endToken {
			return j
		}
	}
	return start
}

func isFirstImageTokenInRun(tokens []int, i int, imageToken int) bool {
	return i == 0 || tokens[i-1] != imageToken
}

func skipImageTokenRun(tokens []int, start int, imageToken int) int {
	for i := start; i < len(tokens); i++ {
		if tokens[i] != imageToken {
			return i - 1
		}
	}
	return len(tokens) - 1
}
