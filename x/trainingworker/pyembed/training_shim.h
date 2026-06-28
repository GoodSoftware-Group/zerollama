/* Embedded Python training: C API for Go CGO.
 * Do not call Py_Finalize after torch has been imported — process exit cleans up.
 */
#ifndef OLLAMA_TRAINING_SHIM_H
#define OLLAMA_TRAINING_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Call once before Py_Initialize. Registers the native extension module. */
void training_preinit_native_module(void);

/* repo_root: directory containing training.py (appended to sys.path for import training).
 * err_out: malloc'd UTF-8 string on failure; caller must free().
 * Returns 0 on success.
 */
int training_init(const char *repo_root, const char *bootstrap_py, char **err_out);

/* 1 after successful training_init; 0 otherwise. */
int training_is_initialized(void);

/* 1 if training_init failed after Py_Initialize (must restart process to retry). */
int training_init_aborted(void);

/* After Go evicts inference VRAM, unblock Python load_model retry. */
void training_ack_vram_headroom(const char *job_id);

char *training_health(char **err_out);
char *training_submit_job(const char *kind, const char *payload_json, char **err_out);
char *training_job_status(const char *job_id, char **err_out);
char *training_list_jobs(char **err_out);
/* training_cancel_job: returns 1 if cancelled, 0 if not cancelled, -1 on error (err_out set). */
int training_cancel_job(const char *job_id, char **err_out);
int training_unload(char **err_out);
void training_shutdown(void);

void training_free(char *s);

#ifdef __cplusplus
}
#endif

#endif /* OLLAMA_TRAINING_SHIM_H */
