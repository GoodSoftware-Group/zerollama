package ollamarunner

import (
	"fmt"

	"github.com/ollama/ollama/tokenizer"
)

const (
	gemma4ImagePlaceholder = "<|image|>"
	gemma4VideoPlaceholder = "<|video|>"
	gemma4AudioPlaceholder = "<|audio|>"
)

func (s *Server) gemma4SoftTokens() (Gemma4SoftTokens, error) {
	tok, ok := s.model.(tokenizer.Tokenizer)
	if !ok {
		return Gemma4SoftTokens{}, fmt.Errorf("gemma4 soft tokens require a tokenizer")
	}
	image, err := gemma4SlotToken(tok, gemma4ImagePlaceholder)
	if err != nil {
		return Gemma4SoftTokens{}, err
	}
	video, err := gemma4SlotToken(tok, gemma4VideoPlaceholder)
	if err != nil {
		return Gemma4SoftTokens{}, err
	}
	audio, err := gemma4SlotToken(tok, gemma4AudioPlaceholder)
	if err != nil {
		return Gemma4SoftTokens{}, err
	}
	return Gemma4SoftTokens{Image: image, Video: video, Audio: audio}, nil
}

func gemma4SlotToken(tok tokenizer.Tokenizer, placeholder string) (int, error) {
	ids, err := tok.Encode(placeholder, false)
	if err != nil {
		return 0, fmt.Errorf("tokenize gemma4 placeholder %q: %w", placeholder, err)
	}
	if len(ids) == 0 {
		return 0, fmt.Errorf("gemma4 placeholder %q tokenized to empty ids", placeholder)
	}
	return int(ids[len(ids)-1]), nil
}
