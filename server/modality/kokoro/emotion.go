package kokoro

import "strings"

// emotionScale is Kokoro's delivery control. The ONNX graph has no emotion
// head, so these words scale the requested speed. Unknown words are rejected.
var emotionScale = map[string]float64{
	"calm":    0.85,
	"sad":     0.80,
	"serious": 0.90,
	"neutral": 1.00,
	"happy":   1.10,
	"excited": 1.20,
	"angry":   1.25,
}

// CanonicalEmotion returns the empty string when emotion is omitted.
// A known word is lowercased. Anything else is a request error.
func CanonicalEmotion(emotion string) (string, error) {
	s := strings.ToLower(strings.TrimSpace(emotion))
	if s == "" {
		return "", nil
	}
	if _, ok := emotionScale[s]; !ok {
		return "", &BadRequestError{Msg: "kokoro emotion must be calm, sad, serious, neutral, happy, excited, or angry"}
	}
	return s, nil
}

// ApplyEmotion multiplies speed by the emotion scale.
// The product has to stay inside [0.5, 2]. An empty emotion leaves speed as it is.
func ApplyEmotion(speed float64, emotion string) (float64, string, error) {
	canon, err := CanonicalEmotion(emotion)
	if err != nil {
		return 0, "", err
	}
	if canon == "" {
		return speed, "", nil
	}
	out := speed * emotionScale[canon]
	if out < 0.5 || out > 2 {
		return 0, "", &BadRequestError{Msg: "kokoro emotion pushes speed outside 0.5 to 2"}
	}
	return out, canon, nil
}
