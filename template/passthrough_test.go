package template

import "testing"

func TestIsPassthrough(t *testing.T) {
	cases := []struct {
		name string
		tmpl string
		want bool
	}{
		{"empty", "", false},
		{"bare prompt", "{{ .Prompt }}", true},
		{"compact", "{{.Prompt}}", true},
		{"prompt response", "{{ .Prompt }}{{ .Response }}", true},
		{"system prompt", "{{ .System }}{{ .Prompt }}", true},
		{"chatml", "<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ .Response }}", false},
		{"deepseek markers", "<｜User｜>{{ .Prompt }}<｜Assistant｜>", false},
		{"messages range", "{{ range .Messages }}{{ .Content }}{{ end }}", false},
		{"literal leftover", "User: {{ .Prompt }}", false},
		{"jinja-ish no go", "{% for m in messages %}{{ m.content }}{% endfor %}", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsPassthrough(tc.tmpl); got != tc.want {
				t.Fatalf("IsPassthrough(%q)=%v want %v", tc.tmpl, got, tc.want)
			}
		})
	}
}
