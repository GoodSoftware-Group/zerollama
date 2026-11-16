package describe

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSaveLoadCatalog(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "photos.json")
	tags := []TagScore{{Label: "outdoor", Score: 0.27}}
	cat := &CatalogFile{
		Directory: "photos",
		Fields:    []string{"file", "tags"},
		Labels:    []string{"outdoor"},
		MinScore:  0.2,
		Images: []Entry{{
			ID:     "a.jpg",
			SHA256: "abc",
			Tags:   &tags,
			File:   &FileFacts{Bytes: 12, Width: 1, Height: 1},
		}},
	}
	cat.Models.Tags = "siglip"
	if err := SaveCatalog(path, cat); err != nil {
		t.Fatal(err)
	}
	got, err := LoadCatalog(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Images) != 1 || got.Images[0].ID != "a.jpg" || !got.Images[0].HasTags() {
		t.Fatalf("%#v", got)
	}
	// rewrite after each image
	got.Images = Upsert(got.Images, Entry{ID: "b.jpg", SHA256: "def", Error: "unsupported image format"})
	if err := SaveCatalog(path, got); err != nil {
		t.Fatal(err)
	}
	got2, err := LoadCatalog(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got2.Images) != 2 {
		t.Fatalf("len=%d", len(got2.Images))
	}
	keep := map[string]struct{}{"a.jpg": {}}
	got2.Images = DropMissing(got2.Images, keep)
	if len(got2.Images) != 1 {
		t.Fatalf("drop: %d", len(got2.Images))
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
}

func TestHashBytes(t *testing.T) {
	a := HashBytes([]byte("x"))
	b := HashBytes([]byte("x"))
	c := HashBytes([]byte("y"))
	if a != b || a == c || len(a) != 64 {
		t.Fatalf("%s %s %s", a, b, c)
	}
}
