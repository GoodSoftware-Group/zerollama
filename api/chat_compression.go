package api

// ApplyStickyChatCompression copies ElideFrom from the previous done meta onto
// the next request. Clients that only append turns must do this so a later
// num_ctx bump cannot restore an already-elided tool and split prefix KV.
//
// No-op when meta is missing, compression did not run (empty Mode), or the
// request already set ElideFrom.
func ApplyStickyChatCompression(req *ChatRequest, meta *ChatCompressionMeta) {
	if req == nil || meta == nil || meta.Mode == "" {
		return
	}
	if req.Compression != nil && req.Compression.ElideFrom != nil {
		return
	}
	from := meta.ElideFrom
	if req.Compression == nil {
		req.Compression = &ChatCompressionConfig{}
	}
	req.Compression.ElideFrom = &from
}
