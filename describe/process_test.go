package describe

import (
	"context"
	"encoding/json"
	"testing"
)

type fakeBackend struct {
	texts  map[string][]float32
	images map[string][]float32
}

func (f *fakeBackend) EmbedTexts(_ context.Context, _ string, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		if v, ok := f.texts[t]; ok {
			out[i] = v
		} else {
			out[i] = []float32{0, 1}
		}
	}
	return out, nil
}

func (f *fakeBackend) EmbedImage(_ context.Context, _ string, data []byte) ([]float32, error) {
	key := string(data)
	if v, ok := f.images[key]; ok {
		return v, nil
	}
	return []float32{1, 0}, nil
}

func (f *fakeBackend) Caption(context.Context, string, []byte) (string, []string, string, error) {
	return "A dog.", []string{}, `{"caption":"A dog.","text":[]}`, nil
}

func (f *fakeBackend) Detect(context.Context, []byte, []string) (*ObjectsResult, error) {
	return &ObjectsResult{Status: "ok", Detections: []Detection{}}, nil
}

type wireOnlyBackend struct{ fakeBackend }

func (w *wireOnlyBackend) Detect(context.Context, []byte, []string) (*ObjectsResult, error) {
	return &ObjectsResult{Status: "wire_only", Reason: "no weights", Detections: []Detection{}}, nil
}

func TestProcessRequestKeptAndNeedsImage(t *testing.T) {
	tags := []TagScore{}
	sha := HashBytes([]byte("img"))
	catalog := []Entry{{
		ID:       "a",
		SHA256:   sha,
		Tags:     &tags,
		TagModel: "siglip",
		Labels:   []string{"dog"},
		MinScore: 0.2,
	}}
	backend := &fakeBackend{
		texts:  map[string][]float32{"dog": {1, 0}},
		images: map[string][]float32{"img": {1, 0}},
	}

	resp, err := ProcessRequest(context.Background(), Request{
		Fields:   []string{"tags"},
		Labels:   []string{"dog"},
		TagModel: "siglip",
		MinScore: 0.2,
		Catalog:  catalog,
		Images:   []ImageInput{{ID: "a", SHA256: sha}},
	}, backend)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Images) != 1 || resp.Images[0].Action != ActionKept {
		t.Fatalf("kept: %#v", resp.Images)
	}

	resp, err = ProcessRequest(context.Background(), Request{
		Fields:   []string{"tags"},
		Labels:   []string{"dog"},
		TagModel: "siglip",
		Catalog:  catalog,
		Images:   []ImageInput{{ID: "b", SHA256: "newhash"}},
	}, backend)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Images[0].Action != ActionNeedsImage {
		t.Fatalf("needs_image: %#v", resp.Images[0])
	}
}

func TestProcessRequestTags(t *testing.T) {
	backend := &fakeBackend{
		texts:  map[string][]float32{"dog": {1, 0}, "cat": {0, 1}},
		images: map[string][]float32{"pixels": {1, 0}},
	}
	resp, err := ProcessRequest(context.Background(), Request{
		Fields:   []string{"tags"},
		Labels:   []string{"dog", "cat"},
		TagModel: "siglip",
		MinScore: 0.5,
		Images:   []ImageInput{{ID: "a", Data: []byte("pixels")}},
	}, backend)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Images[0].Action != ActionProcessed {
		t.Fatalf("action %s", resp.Images[0].Action)
	}
	if resp.Images[0].Tags == nil || len(*resp.Images[0].Tags) != 1 || (*resp.Images[0].Tags)[0].Label != "dog" {
		t.Fatalf("tags %#v", resp.Images[0].Tags)
	}
}

func TestProcessMaxImages(t *testing.T) {
	imgs := make([]ImageInput, MaxImagesPerCall+1)
	for i := range imgs {
		imgs[i] = ImageInput{ID: string(rune('a' + i)), Data: []byte{byte(i)}}
	}
	_, err := ProcessRequest(context.Background(), Request{
		Fields:   []string{"tags"},
		TagModel: "siglip",
		Images:   imgs,
	}, &fakeBackend{})
	if err == nil {
		t.Fatal("expected max images error")
	}
}

func TestParseCaptionJSON(t *testing.T) {
	c, text, ok := ParseCaptionJSON(`{"caption":"Hi","text":["HELLO"]}`)
	if !ok || c != "Hi" || len(text) != 1 {
		t.Fatalf("%v %q %v", ok, c, text)
	}
	_, _, ok = ParseCaptionJSON(`not json`)
	if ok {
		t.Fatal("expected fail")
	}
	_ = json.Marshal
}

func TestProcessWireOnlyObjectsError(t *testing.T) {
	resp, err := ProcessRequest(context.Background(), Request{
		Fields:   []string{"objects"},
		Labels:   []string{"dog"},
		Images:   []ImageInput{{ID: "a", Data: []byte("pixels")}},
		WantObjects: true,
	}, &wireOnlyBackend{})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Images[0].Action != ActionError || resp.Images[0].Objects != nil {
		t.Fatalf("%#v", resp.Images[0])
	}
}

func TestProcessBuiltinPhrasesWhenLabelsEmpty(t *testing.T) {
	backend := &fakeBackend{
		texts:  map[string][]float32{},
		images: map[string][]float32{"pixels": {1, 0}},
	}
	// Seed phrase vectors so only "a photograph" matches.
	for _, lp := range BuiltinLabels {
		backend.texts[lp.Phrase] = []float32{0, 1}
	}
	backend.texts["a photograph"] = []float32{1, 0}

	resp, err := ProcessRequest(context.Background(), Request{
		Fields:   []string{"tags"},
		TagModel: "siglip",
		MinScore: 0.5,
		Images:   []ImageInput{{ID: "a", Data: []byte("pixels")}},
	}, backend)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Images[0].Tags == nil || len(*resp.Images[0].Tags) != 1 || (*resp.Images[0].Tags)[0].Label != "photograph" {
		t.Fatalf("want builtin phrase match: %#v", resp.Images[0].Tags)
	}
}
