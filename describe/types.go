// Package describe is the shared catalog for POST /api/describe and
// `zerollama describe`. Callers keep the catalog; the server never opens a
// directory path.
package describe

import "time"

const (
	ActionProcessed  = "processed"
	ActionKept       = "kept"
	ActionNeedsImage = "needs_image"
	ActionError      = "error"

	DefaultMinScore = 0.20
	MaxImagesPerCall = 8
)

// LabelPhrase is one built-in tag: short JSON label plus the text embedded.
type LabelPhrase struct {
	Label  string
	Phrase string
}

// BuiltinLabels are coarse phrases SigLIP can compare without a user list.
var BuiltinLabels = []LabelPhrase{
	{Label: "photograph", Phrase: "a photograph"},
	{Label: "screenshot", Phrase: "a screenshot of a computer screen"},
	{Label: "document", Phrase: "a photo of a document"},
	{Label: "receipt", Phrase: "a photo of a receipt"},
	{Label: "person", Phrase: "a photo of a person"},
	{Label: "animal", Phrase: "a photo of an animal"},
	{Label: "food", Phrase: "a photo of food"},
	{Label: "outdoor", Phrase: "an outdoor scene"},
	{Label: "indoor", Phrase: "an indoor scene"},
}

// FileFacts are always stored; they are not a mode.
type FileFacts struct {
	Bytes  int64     `json:"bytes"`
	Width  int       `json:"width,omitempty"`
	Height int       `json:"height,omitempty"`
	Mtime  time.Time `json:"mtime,omitempty"`
}

// TagScore is one label that cleared the score cutoff.
type TagScore struct {
	Label string  `json:"label"`
	Score float64 `json:"score"`
}

// Detection is one box from POST /v1/detect.
type Detection struct {
	Label string    `json:"label"`
	Score float64   `json:"score"`
	Box   []float64 `json:"box"`
}

// ObjectsResult is the detect category on one entry.
type ObjectsResult struct {
	Status     string       `json:"status"`
	Detections []Detection  `json:"detections"`
	Reason     string       `json:"reason,omitempty"`
}

// Entry is one picture in a catalog. Id is opaque (relative path or DB id).
// Tags is a pointer so an empty list (matched nothing) is distinct from
// tags never requested.
type Entry struct {
	ID           string         `json:"id"`
	SHA256       string         `json:"sha256,omitempty"`
	File         *FileFacts     `json:"file,omitempty"`
	Tags         *[]TagScore    `json:"tags,omitempty"`
	Caption      string         `json:"caption,omitempty"`
	CaptionError string         `json:"caption_error,omitempty"`
	Text         []string       `json:"text,omitempty"`
	Objects      *ObjectsResult `json:"objects,omitempty"`
	Error        string         `json:"error,omitempty"`
	TagModel     string         `json:"tag_model,omitempty"`
	CaptionModel string         `json:"caption_model,omitempty"`
	Labels       []string       `json:"labels,omitempty"`
	MinScore     float64        `json:"min_score,omitempty"`
}

// SetTags stores tags, including an empty match list.
func (e *Entry) SetTags(tags []TagScore) {
	if tags == nil {
		tags = []TagScore{}
	}
	e.Tags = &tags
}

// HasTags reports whether tags were computed for this entry.
func (e Entry) HasTags() bool { return e.Tags != nil }

// HasCaption reports whether a caption (or caption_error) was stored.
func (e Entry) HasCaption() bool {
	return e.Caption != "" || e.CaptionError != ""
}

// ImageInput is one picture in a describe request. Data is base64-decoded bytes
// on the wire as a string; the JSON field is "data".
type ImageInput struct {
	ID     string `json:"id"`
	SHA256 string `json:"sha256,omitempty"`
	Data   []byte `json:"data,omitempty"`
}

// Request is POST /api/describe.
type Request struct {
	Fields       []string  `json:"fields,omitempty"`
	Labels       []string  `json:"labels,omitempty"`
	MinScore     float64   `json:"min_score,omitempty"`
	TagModel     string    `json:"tag_model,omitempty"`
	CaptionModel string    `json:"caption_model,omitempty"`
	WantObjects  bool      `json:"objects,omitempty"`
	Catalog      []Entry   `json:"catalog,omitempty"`
	Images       []ImageInput `json:"images"`
}

// ResultEntry is one id in the response.
type ResultEntry struct {
	Entry
	Action string `json:"action"`
}

// Response is the body of POST /api/describe.
type Response struct {
	Images []ResultEntry `json:"images"`
}

// CatalogFile is the on-disk shape written by the CLI (`photos.json`).
type CatalogFile struct {
	Directory string   `json:"directory"`
	Fields    []string `json:"fields"`
	Labels    []string `json:"labels"`
	Skipped   []string `json:"skipped,omitempty"`
	Models    struct {
		Tags    string `json:"tags,omitempty"`
		Caption string `json:"caption,omitempty"`
	} `json:"models"`
	MinScore float64 `json:"min_score"`
	Images   []Entry `json:"images"`
}
