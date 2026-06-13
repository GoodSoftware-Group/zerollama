package version

// Version is the default when the binary is built without -ldflags.
// Why ldflags in build_zerollama_mac.sh: release builds must report the tag
// operators passed (VERSION=…) via ./zerollama --version and /api/version.
var Version string = "0.0.1"
