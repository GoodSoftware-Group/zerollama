// Command duplex_hygiene_check fails when a spoken reply leaks role labels.
//
//	go run ./scripts/smoke/duplex_hygiene_check.go -text 'Hello there.'
//	go run ./scripts/smoke/duplex_hygiene_check.go -file reply.txt
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/ollama/ollama/server/voicepipe"
)

func main() {
	text := flag.String("text", "", "spoken reply text")
	file := flag.String("file", "", "path to spoken reply text")
	flag.Parse()
	raw := strings.TrimSpace(*text)
	if *file != "" {
		b, err := os.ReadFile(*file)
		if err != nil {
			fatal(err)
		}
		raw = string(b)
	}
	if err := voicepipe.SpokenReplyHygiene(raw); err != nil {
		fatal(fmt.Errorf("%w (%q)", err, strings.TrimSpace(raw)))
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
