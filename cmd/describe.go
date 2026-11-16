package cmd

import (
	"bytes"
	"context"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/spf13/cobra"
	_ "golang.org/x/image/webp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/describe"
	"github.com/ollama/ollama/envconfig"
)

var imageExts = map[string]bool{
	".jpg": true, ".jpeg": true, ".png": true, ".webp": true, ".gif": true,
	".heic": true, ".heif": true,
}

var rasterExts = map[string]bool{
	".jpg": true, ".jpeg": true, ".png": true, ".webp": true, ".gif": true,
}

// NewDescribeCommand walks a directory and writes a JSON catalog of images.
func NewDescribeCommand() *cobra.Command {
	var (
		outPath      string
		labelsFlag   string
		tagModel     string
		captionModel string
		minScore     float64
		wantObjects  bool
		fieldsFlag   string
	)

	cmd := &cobra.Command{
		Use:   "describe DIRECTORY",
		Short: "Tag pictures in a folder into one JSON catalog",
		Long: `Walk DIRECTORY and write a JSON catalog (default DIRECTORY.json beside it).

By default each picture is tagged with SigLIP against a short built-in word list.
Use --labels to supply your own words. --caption MODEL adds one sentence per
picture. --objects adds boxes (requires ZEROLLAMA_DETECT_URL).

The same command resumes: unchanged pictures are skipped, new files are added,
and paths that disappeared are dropped from the catalog. photos.json is rewritten
after every finished picture so a kill loses only the image in flight.

Tags run first across the folder, then captions, then objects, so two models
are not swapped on every file.

Talks to the server at OLLAMA_HOST. Does not start serve.`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runDescribe(cmd, args[0], describeFlags{
				OutPath:      outPath,
				Labels:       describe.ParseCommaLabels(labelsFlag),
				TagModel:     tagModel,
				CaptionModel: captionModel,
				MinScore:     minScore,
				WantObjects:  wantObjects,
				Fields:       fieldsFlag,
			})
		},
	}

	cmd.Flags().StringVarP(&outPath, "output", "o", "", "catalog JSON path (default: DIRECTORY.json beside the folder)")
	cmd.Flags().StringVar(&labelsFlag, "labels", "", "comma-separated tag words (default: built-in coarse list)")
	cmd.Flags().StringVar(&tagModel, "tag-model", "", "SigLIP embedding model (default: the only local siglip model)")
	cmd.Flags().StringVar(&captionModel, "caption", "", "vision model for one sentence per picture")
	cmd.Flags().Float64Var(&minScore, "min-score", describe.DefaultMinScore, "keep a tag only when cosine similarity is at least this")
	cmd.Flags().BoolVar(&wantObjects, "objects", false, "also run object detect (requires ZEROLLAMA_DETECT_URL)")
	cmd.Flags().StringVar(&fieldsFlag, "fields", "", "comma list: tags,caption,objects (default: tags; caption/objects also set by their flags)")
	return cmd
}

type describeFlags struct {
	OutPath      string
	Labels       []string
	TagModel     string
	CaptionModel string
	MinScore     float64
	WantObjects  bool
	Fields       string
}

func runDescribe(cmd *cobra.Command, dir string, flags describeFlags) error {
	dir, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	st, err := os.Stat(dir)
	if err != nil {
		return err
	}
	if !st.IsDir() {
		return fmt.Errorf("%s is not a directory", dir)
	}

	outPath := flags.OutPath
	if outPath == "" {
		outPath = dir + ".json"
	}
	outPath, err = filepath.Abs(outPath)
	if err != nil {
		return err
	}

	want := describe.ParseFields(parseFieldsCSV(flags.Fields))
	if flags.CaptionModel != "" {
		want.Caption = true
	}
	if flags.WantObjects {
		want.Objects = true
	}
	if !want.Tags && !want.Caption && !want.Objects {
		want.Tags = true
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}
	ctx := cmd.Context()

	tagModel := strings.TrimSpace(flags.TagModel)
	if want.Tags && tagModel == "" {
		tagModel, err = resolveSiglipModel(ctx, client)
		if err != nil {
			return err
		}
	}
	if want.Caption && strings.TrimSpace(flags.CaptionModel) == "" {
		return fmt.Errorf("--caption requires a vision model name")
	}
	if want.Objects && strings.TrimSpace(envconfig.DetectURL()) == "" {
		return fmt.Errorf("objects requested but ZEROLLAMA_DETECT_URL is unset — see docs/detect.md")
	}

	// Short labels for Decide / catalog. Empty flags.Labels keeps builtin phrases
	// on the server (do not send resolved short names as custom labels).
	labels, _ := describe.ResolveLabels(flags.Labels)
	settings := describe.Settings{
		TagModel:     tagModel,
		CaptionModel: strings.TrimSpace(flags.CaptionModel),
		Labels:       labels,
		MinScore:     flags.MinScore,
	}
	if settings.MinScore <= 0 {
		settings.MinScore = describe.DefaultMinScore
	}

	files, err := walkImages(dir, outPath)
	if err != nil {
		return err
	}

	cat, err := describe.LoadCatalog(outPath)
	if err != nil {
		return err
	}
	cat.Directory = filepath.Base(dir)
	fieldList := []string{"file"}
	if want.Tags {
		fieldList = append(fieldList, "tags")
	}
	if want.Caption {
		fieldList = append(fieldList, "caption")
	}
	if want.Objects {
		fieldList = append(fieldList, "objects")
	}
	cat.Fields = fieldList
	cat.Labels = labels
	cat.MinScore = settings.MinScore
	cat.Models.Tags = tagModel
	cat.Models.Caption = settings.CaptionModel
	cat.Skipped = nil

	index := describe.IndexByID(cat.Images)
	keep := make(map[string]struct{}, len(files))
	var kept, newN, changed, removed int
	for _, f := range files {
		keep[f.rel] = struct{}{}
		existing := index[f.rel]
		mtimeEqual := fileMtimeEqual(existing, f.info)
		if describe.LocalSkip(existing, f.info.Size(), mtimeEqual, want, settings) {
			kept++
			continue
		}
		if existing == nil {
			newN++
		} else {
			changed++
		}
	}
	for id := range index {
		if _, ok := keep[id]; !ok {
			removed++
		}
	}
	cat.Images = describe.DropMissing(cat.Images, keep)
	index = describe.IndexByID(cat.Images)

	fmt.Fprintf(os.Stderr, "%d pictures. Tags with %s against %s.\n",
		len(files), orDash(tagModel), strings.Join(labels, ", "))
	if want.Caption {
		fmt.Fprintf(os.Stderr, "Caption with %s (slow).\n", settings.CaptionModel)
	}
	if want.Objects {
		fmt.Fprintf(os.Stderr, "Objects via detect sidecar.\n")
	}
	fmt.Fprintf(os.Stderr, "kept %d, new %d, changed %d, removed %d\n", kept, newN, changed, removed)

	runPhase := func(phaseWant describe.Want, label string) error {
		var n int
		for _, f := range files {
			existing := index[f.rel]
			mtimeEqual := fileMtimeEqual(existing, f.info)
			if describe.LocalSkip(existing, f.info.Size(), mtimeEqual, phaseWant, settings) {
				continue
			}
			n++
			entry, err := describeOne(ctx, client, f, existing, phaseWant, settings, flags.Labels)
			if err != nil {
				fmt.Fprintf(os.Stderr, "[%s] %s  error: %v\n", label, f.rel, err)
				continue
			}
			cat.Images = describe.Upsert(cat.Images, entry)
			index = describe.IndexByID(cat.Images)
			if err := describe.SaveCatalog(outPath, cat); err != nil {
				return err
			}
			fmt.Fprintf(os.Stderr, "[%s] %d  %s  %s\n", label, n, f.rel, entrySummary(entry, label))
		}
		return nil
	}

	if want.Tags {
		if err := runPhase(describe.Want{Tags: true}, "tags"); err != nil {
			return err
		}
		if want.Caption || want.Objects {
			// Free SigLIP before the vision / detect phase.
			_, _ = client.Unload(ctx, &api.UnloadRequest{Model: tagModel})
		}
	}
	if want.Caption {
		if err := runPhase(describe.Want{Caption: true}, "caption"); err != nil {
			return err
		}
		if want.Objects {
			_, _ = client.Unload(ctx, &api.UnloadRequest{Model: settings.CaptionModel})
		}
	}
	if want.Objects {
		if err := runPhase(describe.Want{Objects: true}, "objects"); err != nil {
			return err
		}
	}

	if err := describe.SaveCatalog(outPath, cat); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "wrote %s\n", outPath)
	return nil
}

func fileMtimeEqual(existing *describe.Entry, info os.FileInfo) bool {
	if existing == nil || existing.File == nil {
		return false
	}
	fileMT := info.ModTime().UTC().Truncate(time.Second)
	if existing.File.Mtime.Equal(fileMT) {
		return true
	}
	return existing.File.Mtime.Truncate(time.Second).Equal(fileMT)
}

func entrySummary(entry describe.Entry, phase string) string {
	if entry.Error != "" {
		return entry.Error
	}
	switch phase {
	case "tags":
		if entry.Tags != nil && len(*entry.Tags) > 0 {
			names := make([]string, len(*entry.Tags))
			for j, t := range *entry.Tags {
				names[j] = t.Label
			}
			return strings.Join(names, ",")
		}
		return "(no tags)"
	case "caption":
		if entry.Caption != "" {
			return entry.Caption
		}
		return "(no caption)"
	case "objects":
		if entry.Objects != nil {
			return fmt.Sprintf("%d boxes", len(entry.Objects.Detections))
		}
		return "(no objects)"
	}
	return phase
}

type imageFile struct {
	rel  string
	path string
	info os.FileInfo
}

func walkImages(dir, outPath string) ([]imageFile, error) {
	var out []imageFile
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			base := filepath.Base(path)
			if base != "." && strings.HasPrefix(base, ".") {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasPrefix(filepath.Base(path), ".") {
			return nil
		}
		if sameFile(path, outPath) {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if !imageExts[ext] {
			return nil
		}
		rel, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		out = append(out, imageFile{rel: rel, path: path, info: info})
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

func sameFile(a, b string) bool {
	aa, err1 := filepath.Abs(a)
	bb, err2 := filepath.Abs(b)
	if err1 != nil || err2 != nil {
		return a == b
	}
	return aa == bb
}

func describeOne(ctx context.Context, client *api.Client, f imageFile, existing *describe.Entry, want describe.Want, settings describe.Settings, callerLabels []string) (describe.Entry, error) {
	ext := strings.ToLower(filepath.Ext(f.path))
	data, err := loadImageBytes(f.path, ext)
	if err != nil {
		e := describe.Entry{
			ID:    f.rel,
			Error: err.Error(),
			File: &describe.FileFacts{
				Bytes: f.info.Size(),
				Mtime: f.info.ModTime().UTC(),
			},
		}
		if existing != nil && existing.SHA256 != "" {
			e.SHA256 = existing.SHA256
		}
		return e, nil
	}

	sha := describe.HashBytes(data)
	fields := []string{}
	if want.Tags {
		fields = append(fields, "tags")
	}
	if want.Caption {
		fields = append(fields, "caption")
	}
	if want.Objects {
		fields = append(fields, "objects")
	}

	req := &describe.Request{
		Fields:       fields,
		Labels:       callerLabels, // empty → server uses builtin phrases
		MinScore:     settings.MinScore,
		TagModel:     settings.TagModel,
		CaptionModel: settings.CaptionModel,
		WantObjects:  want.Objects,
		Images: []describe.ImageInput{{
			ID:   f.rel,
			Data: data,
		}},
	}
	if existing != nil {
		req.Catalog = []describe.Entry{*existing}
	}

	resp, err := client.Describe(ctx, req)
	if err != nil {
		return describe.Entry{}, err
	}
	if len(resp.Images) == 0 {
		return describe.Entry{}, fmt.Errorf("empty describe response")
	}
	entry := resp.Images[0].Entry
	if resp.Images[0].Action == describe.ActionError && entry.Error == "" {
		entry.Error = "describe failed"
	}
	entry.ID = f.rel
	entry.SHA256 = sha
	if entry.File == nil {
		entry.File = &describe.FileFacts{}
	}
	entry.File.Bytes = f.info.Size()
	entry.File.Mtime = f.info.ModTime().UTC()
	if entry.File.Width == 0 || entry.File.Height == 0 {
		if cfg, _, err := image.DecodeConfig(bytes.NewReader(data)); err == nil {
			entry.File.Width = cfg.Width
			entry.File.Height = cfg.Height
		}
	}
	// Merge with existing fields from other phases (tags then caption).
	if existing != nil {
		if !want.Tags && existing.HasTags() {
			entry.Tags = existing.Tags
			entry.TagModel = existing.TagModel
			if len(entry.Labels) == 0 {
				entry.Labels = existing.Labels
			}
			if entry.MinScore == 0 {
				entry.MinScore = existing.MinScore
			}
		}
		if !want.Caption && existing.HasCaption() {
			entry.Caption = existing.Caption
			entry.CaptionError = existing.CaptionError
			entry.Text = existing.Text
			entry.CaptionModel = existing.CaptionModel
		}
		if !want.Objects && existing.Objects != nil {
			entry.Objects = existing.Objects
		}
	}
	return entry, nil
}

func loadImageBytes(path, ext string) ([]byte, error) {
	if rasterExts[ext] {
		return os.ReadFile(path)
	}
	if ext == ".heic" || ext == ".heif" {
		if runtime.GOOS != "darwin" {
			return nil, fmt.Errorf("unsupported image format")
		}
		return convertHEIC(path)
	}
	return nil, fmt.Errorf("unsupported image format")
}

func convertHEIC(path string) ([]byte, error) {
	if _, err := exec.LookPath("sips"); err != nil {
		return nil, fmt.Errorf("unsupported image format")
	}
	tmp, err := os.CreateTemp("", "zerollama-heic-*.jpg")
	if err != nil {
		return nil, err
	}
	tmpName := tmp.Name()
	_ = tmp.Close()
	defer os.Remove(tmpName)

	cmd := exec.Command("sips", "-s", "format", "jpeg", path, "--out", tmpName)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("unsupported image format")
	}
	return os.ReadFile(tmpName)
}

func resolveSiglipModel(ctx context.Context, client *api.Client) (string, error) {
	list, err := client.List(ctx)
	if err != nil {
		return "", fmt.Errorf("list models: %w\nHint: start the daemon — zerollama serve", err)
	}
	var matches []string
	for _, m := range list.Models {
		if isSiglipModel(m) {
			matches = append(matches, m.Name)
		}
	}
	switch len(matches) {
	case 1:
		return matches[0], nil
	case 0:
		return "", fmt.Errorf("no local SigLIP model found; pass --tag-model MODEL")
	default:
		return "", fmt.Errorf("several SigLIP models found (%s); pass --tag-model MODEL", strings.Join(matches, ", "))
	}
}

func isSiglipModel(m api.ListModelResponse) bool {
	name := strings.ToLower(m.Name)
	if strings.Contains(name, "siglip") {
		return true
	}
	fam := strings.ToLower(m.Details.Family)
	if strings.Contains(fam, "siglip") {
		return true
	}
	for _, f := range m.Details.Families {
		if strings.Contains(strings.ToLower(f), "siglip") {
			return true
		}
	}
	return false
}

func parseFieldsCSV(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func orDash(s string) string {
	if s == "" {
		return "(none)"
	}
	return s
}
