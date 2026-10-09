package autoroute

import "regexp"

// Output modalities a chat turn may really be asking for (docs/auto-router.md §9, AR8).
const (
	ModalityText   = "text"
	ModalityImage  = "image"
	ModalitySpeech = "speech"
	ModalityVideo  = "video"
	ModalityMusic  = "music"
)

// Keyword rules for GuessOutputModality. WHY a generate-verb near an artefact noun rather
// than nouns alone: "what's in this photo?", "summarise this clip" and "transcribe this
// voice note" mention media without asking to produce it. Order matters — video before
// image ("animate this logo"), music before speech ("write and perform a jingle").
var modalityRules = []struct {
	modality string
	re       *regexp.Regexp
}{
	{ModalityVideo, regexp.MustCompile(`(?i)\banimate\b|\b(make|generate|create|render|produce|turn)\b(?:\W+\w+){0,6}?\W+(clip|video|animation|movie|gif)s?\b`)},
	{ModalityMusic, regexp.MustCompile(`(?i)\b(compose|perform|produce|make|generate|create)\b(?:\W+\w+){0,6}?\W+(song|jingle|beat|melody|tune|track|soundtrack|music)s?\b`)},
	{ModalitySpeech, regexp.MustCompile(`(?i)\b(out loud|aloud|narrate|text[- ]to[- ]speech)\b|\bin an? \w+(?:\W+\w+)? voice\b|\bsay\s+['"‘“]|\bread (?:it|this|that)(?: back)? to me\b|\bspeak (?:it|this|that)\b`)},
	// Bare drawing verbs count only as a request (clause start, "can you …", "please …"):
	// "you may draw one fruit from the box" is not asking for a picture.
	{ModalityImage, regexp.MustCompile(`(?i)(?:^|[.!?]\s+|\b(?:can|could|would) you\s+|\bplease\s+)(draw|sketch|paint|illustrate)\b|\b(make|generate|create|design|render|edit)\b(?:\W+\w+){0,6}?\W+(picture|image|photo|logo|diagram|poster|icon|illustration|drawing|wallpaper|floor plan)s?\b`)},
}

// GuessOutputModality is the no-model fallback for cross-modal intent: a keyword guess at
// whether a chat turn wants an image, speech, video or music instead of text. It is only
// ever advisory (a suggestion header), never a reroute. Jev answers supersede it.
func GuessOutputModality(text string) (string, bool) {
	for _, r := range modalityRules {
		if r.re.MatchString(text) {
			return r.modality, true
		}
	}
	return "", false
}
