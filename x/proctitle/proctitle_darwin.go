//go:build darwin

package proctitle

/*
#include <crt_externs.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

// Rewrite argv[0] using the contiguous bytes from argv[0] up to (not into)
// environ[0]. Do not memset environ strings. Null remaining argv pointers so
// ps shows only the new title.
static void zerollama_setproctitle(const char *title) {
	if (title == NULL || title[0] == '\0') {
		return;
	}
	char tname[64];
	strncpy(tname, title, sizeof(tname) - 1);
	tname[sizeof(tname) - 1] = '\0';
	pthread_setname_np(tname);

	int argc = *_NSGetArgc();
	char **argv = *_NSGetArgv();
	char **environ = *_NSGetEnviron();
	if (argv == NULL || argc < 1 || argv[0] == NULL) {
		return;
	}

	char *begin = argv[0];
	char *limit = begin + strlen(begin);
	if (environ != NULL && environ[0] != NULL && environ[0] > begin) {
		limit = environ[0] - 1;
	} else if (argc > 1 && argv[argc - 1] != NULL) {
		limit = argv[argc - 1] + strlen(argv[argc - 1]);
	}
	if (limit <= begin) {
		return;
	}
	size_t cap = (size_t)(limit - begin);
	size_t n = strlen(title);
	if (n + 1 > cap) {
		n = cap - 1;
	}
	memset(begin, 0, cap);
	memcpy(begin, title, n);

	for (int i = 1; i < argc; i++) {
		argv[i] = NULL;
	}
}
*/
import "C"
import "unsafe"

func setOS(name string) {
	c := C.CString(name)
	defer C.free(unsafe.Pointer(c))
	C.zerollama_setproctitle(c)
}
