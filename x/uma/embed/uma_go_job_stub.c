/* Non-Go link of uma_glue.c: llama-server has no goUmaMlxJob export.
 * Go builds provide the real symbol via //export; do not link this stub into
 * cgo binaries.
 */
void goUmaMlxJob(void *ctx) { (void)ctx; }
