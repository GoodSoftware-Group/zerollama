#ifndef OLLAMA_RUNTIME_SHIM_H
#define OLLAMA_RUNTIME_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* repo_root: zerollama repo; runtime_parent: repo/runtime (for import runtime).
 * port: localhost port for embedded uvicorn. Reuses Py_Initialize if already up (training).
 */
int runtime_embed_start(
	const char *repo_root,
	const char *runtime_parent,
	int port,
	const char *bootstrap_py,
	char **err_out);

int runtime_embed_is_started(void);
void runtime_embed_free(char *s);

#ifdef __cplusplus
}
#endif

#endif
