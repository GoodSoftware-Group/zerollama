// Package remotestore implements remote model blob storage for zerollama.
//
// Why: inference hosts fill disk with hundreds of GB of models. A central
// content-addressed store (sha256 digests, same layout as $OLLAMA_MODELS) plus
// on-demand fetch into a local cache lets operators keep one canonical tree
// and pull only what each node runs.
//
// Auth and BulkTransport are payload-agnostic on purpose: later KV spillover
// or compute buffer moves should reuse HMAC + transport preference instead of
// inventing a second wire. Tensor catalog / tensorproto language is specified
// in v1 so stream/runtime paging can land without redesigning addressing.
package remotestore
