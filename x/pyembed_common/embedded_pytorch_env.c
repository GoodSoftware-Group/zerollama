/*
 * Embedded PyTorch library path setup (shared by training + runtime pyembed shims).
 */
#include "embedded_pytorch_env.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _WIN32
#define PATH_LIST_SEP ';'
static const char *lib_path_env(void) { return "PATH"; }
#else
#ifdef __APPLE__
#define PATH_LIST_SEP ':'
static const char *lib_path_env(void) { return "DYLD_LIBRARY_PATH"; }
#else
#define PATH_LIST_SEP ':'
static const char *lib_path_env(void) { return "LD_LIBRARY_PATH"; }
#endif
#endif

static int path_is_dir(const char *path) {
	struct stat st;
	return path && *path && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int copy_string(char *out, size_t out_sz, const char *src) {
	size_t n;

	if (!out || out_sz == 0 || !src)
		return 0;
	n = strlen(src);
	if (n + 1 > out_sz)
		return 0;
	memcpy(out, src, n + 1);
	return 1;
}

/* Return 1 if a/b/c fits in out (NUL-terminated); 0 if any part missing or too long. */
static int join_path3(char *out, size_t out_sz, const char *a, const char *b, const char *c) {
	size_t la, lb, lc, need;
	char *p;

	if (!out || out_sz == 0 || !a || !*a || !b || !*b || !c || !*c)
		return 0;
	la = strlen(a);
	lb = strlen(b);
	lc = strlen(c);
	need = la + 1 + lb + 1 + lc + 1;
	if (need > out_sz)
		return 0;
	p = out;
	memcpy(p, a, la);
	p += la;
	*p++ = '/';
	memcpy(p, b, lb);
	p += lb;
	*p++ = '/';
	memcpy(p, c, lc);
	p += lc;
	*p = '\0';
	return 1;
}

static int join_path2(char *out, size_t out_sz, const char *a, const char *b) {
	size_t la, lb, need;
	char *p;

	if (!out || out_sz == 0 || !a || !*a || !b || !*b)
		return 0;
	la = strlen(a);
	lb = strlen(b);
	need = la + 1 + lb + 1;
	if (need > out_sz)
		return 0;
	p = out;
	memcpy(p, a, la);
	p += la;
	*p++ = '/';
	memcpy(p, b, lb);
	p += lb;
	*p = '\0';
	return 1;
}

static int entry_shadows_cudnn(const char *entry) {
	if (!entry || !*entry)
		return 0;
	if (strstr(entry, "hostlibs") != NULL)
		return 1;
	for (const char *p = entry; *p; p++) {
		if ((p[0] == 'c' || p[0] == 'C') && (p[1] == 'u' || p[1] == 'U') &&
		    (p[2] == 'd' || p[2] == 'D') && (p[3] == 'n' || p[3] == 'N') &&
		    (p[4] == 'n' || p[4] == 'N'))
			return 1;
	}
	return 0;
}

static int try_torch_lib(const char *site_packages, char *out, size_t out_sz) {
	char candidate[4096];
	if (!site_packages || !*site_packages || !out || out_sz == 0)
		return 0;
	if (!join_path3(candidate, sizeof(candidate), site_packages, "torch", "lib"))
		return 0;
	if (!path_is_dir(candidate))
		return 0;
	return copy_string(out, out_sz, candidate);
}

static int path_matches_python_version(const char *path, int py_major, int py_minor) {
	char tag[32];

	if (!path || !*path)
		return 0;
	snprintf(tag, sizeof(tag), "python%d.%d", py_major, py_minor);
	if (strstr(path, tag) != NULL)
		return 1;
	/* Flat installs and repo roots without a pythonX.Y segment are allowed. */
	if (strstr(path, "/python") == NULL && strstr(path, "\\python") == NULL)
		return 1;
	return 0;
}

static int try_site_from_pythonpath_env(int py_major, int py_minor, char *out, size_t out_sz) {
	const char *pp = getenv("PYTHONPATH");
	char *copy;
	char *save;

	if (!pp || !*pp)
		return 0;
	copy = strdup(pp);
	if (!copy)
		return 0;
	save = NULL;
	for (char *tok = strtok_r(copy, ":", &save); tok; tok = strtok_r(NULL, ":", &save)) {
		if (!*tok)
			continue;
		if (!path_matches_python_version(tok, py_major, py_minor))
			continue;
		if (!path_is_dir(tok))
			continue;
		if (copy_string(out, out_sz, tok)) {
			free(copy);
			return 1;
		}
	}
	free(copy);
	return 0;
}

static int try_site_from_venv_root(
	const char *venv_root, int py_major, int py_minor, char *out, size_t out_sz) {
	char pydir[64];
	char libroot[4096];
	char site[4096];

	if (!venv_root || !*venv_root)
		return 0;
	snprintf(pydir, sizeof(pydir), "python%d.%d", py_major, py_minor);
	if (!join_path2(libroot, sizeof(libroot), venv_root, "lib"))
		return 0;
	if (!join_path3(site, sizeof(site), libroot, pydir, "site-packages"))
		return 0;
	if (!path_is_dir(site))
		return 0;
	return copy_string(out, out_sz, site);
}

static int repo_training_site_packages(
	const char *repo_root, int py_major, int py_minor, char *out, size_t out_sz) {
	char pydir[64];
	char libroot[4096];
	char site[4096];
	/* WHY .venv-training first: canonical operator path (uv); venv-training is legacy fallback only. */
	static const char *venv_names[] = {".venv-training", "venv-training", NULL};
	size_t i;

	if (!repo_root || !*repo_root || !out || out_sz == 0)
		return 0;
	snprintf(pydir, sizeof(pydir), "python%d.%d", py_major, py_minor);
	for (i = 0; venv_names[i] != NULL; i++) {
		if (!join_path3(libroot, sizeof(libroot), repo_root, venv_names[i], "lib"))
			continue;
		if (!join_path3(site, sizeof(site), libroot, pydir, "site-packages"))
			continue;
		if (!path_is_dir(site))
			continue;
		return copy_string(out, out_sz, site);
	}
	return 0;
}

static int resolve_training_site_packages(
	const char *repo_root, int py_major, int py_minor, char *out, size_t out_sz) {
	const char *site_env;
	const char *venv_env;

	out[0] = '\0';

	site_env = getenv("TRAINING_UV_SITE_PACKAGES");
	if (site_env && path_matches_python_version(site_env, py_major, py_minor) &&
	    path_is_dir(site_env) && copy_string(out, out_sz, site_env))
		return 1;

	venv_env = getenv("TRAINING_UV_VENV");
	if (try_site_from_venv_root(venv_env, py_major, py_minor, out, out_sz))
		return 1;

	if (repo_training_site_packages(repo_root, py_major, py_minor, out, out_sz))
		return 1;

	if (try_site_from_pythonpath_env(py_major, py_minor, out, out_sz))
		return 1;

	return 0;
}

static void scan_venv_lib(
	const char *venv_lib, int py_major, int py_minor, char *out, size_t out_sz) {
	char libdir[4096];
	DIR *d;

	if (!venv_lib || !*venv_lib || out[0])
		return;
	if (!copy_string(libdir, sizeof(libdir), venv_lib))
		return;
	d = opendir(libdir);
	if (!d)
		return;
	{
		struct dirent *ent;
		while ((ent = readdir(d)) != NULL) {
			char site[4096];
			if (strncmp(ent->d_name, "python", 6) != 0)
				continue;
			if (!path_matches_python_version(ent->d_name, py_major, py_minor))
				continue;
			if (!join_path3(site, sizeof(site), libdir, ent->d_name, "site-packages"))
				continue;
			if (try_torch_lib(site, out, out_sz))
				break;
		}
	}
	closedir(d);
}

static void find_torch_lib(
	const char *repo_root, int py_major, int py_minor, char *out, size_t out_sz) {
	const char *site_env;
	const char *pp;
	char *copy;
	char *save;
	char site[4096];

	out[0] = '\0';

	if (repo_training_site_packages(repo_root, py_major, py_minor, site, sizeof(site)) &&
	    try_torch_lib(site, out, out_sz))
		return;

	site_env = getenv("TRAINING_UV_SITE_PACKAGES");
	if (site_env && path_matches_python_version(site_env, py_major, py_minor) &&
	    try_torch_lib(site_env, out, out_sz))
		return;

	pp = getenv("PYTHONPATH");
	if (!pp || !*pp)
		goto venv;

	copy = strdup(pp);
	if (!copy)
		goto venv;
	save = NULL;
	for (char *tok = strtok_r(copy, ":", &save); tok; tok = strtok_r(NULL, ":", &save)) {
		if (*tok && path_matches_python_version(tok, py_major, py_minor) &&
		    try_torch_lib(tok, out, out_sz))
			break;
	}
	free(copy);

venv:
	if (out[0])
		return;

	{
		const char *venv = getenv("VIRTUAL_ENV");
		if (venv && *venv) {
			char libdir[4096];
			if (join_path2(libdir, sizeof(libdir), venv, "lib"))
				scan_venv_lib(libdir, py_major, py_minor, out, out_sz);
		}
	}
	if (out[0])
		return;

	if (repo_root && *repo_root) {
		static const char *venv_names[] = {".venv-training/lib", "venv-training/lib", NULL};
		size_t i;
		for (i = 0; venv_names[i] != NULL && !out[0]; i++) {
			char libdir[4096];
			if (join_path2(libdir, sizeof(libdir), repo_root, venv_names[i]))
				scan_venv_lib(libdir, py_major, py_minor, out, out_sz);
		}
	}
}

static char *filter_lib_path(const char *raw, const char *torch_lib) {
	char *copy;
	char *save;
	char *result;
	size_t cap = 1;

	if (!raw || !*raw)
		return calloc(1, 1);
	copy = strdup(raw);
	if (!copy)
		return NULL;
	result = calloc(1, 1);
	if (!result) {
		free(copy);
		return NULL;
	}
	save = NULL;
	for (char *tok = strtok_r(copy, ":", &save); tok; tok = strtok_r(NULL, ":", &save)) {
		size_t len;
		char *next;

		if (!*tok)
			continue;
		if (torch_lib && *torch_lib && strcmp(tok, torch_lib) == 0)
			continue;
		if (entry_shadows_cudnn(tok))
			continue;
		len = strlen(result) + strlen(tok) + 2;
		if (len > cap) {
			cap = len + 256;
			next = realloc(result, cap);
			if (!next) {
				free(result);
				free(copy);
				return NULL;
			}
			result = next;
		}
		if (*result)
			strcat(result, ":");
		strcat(result, tok);
	}
	free(copy);
	return result;
}

static char *join_lib_paths(const char *prefix, const char *rest) {
	size_t lp, lr, need;
	char *buf;

	if (!prefix || !*prefix)
		return rest && *rest ? strdup(rest) : calloc(1, 1);
	if (!rest || !*rest)
		return strdup(prefix);
	lp = strlen(prefix);
	lr = strlen(rest);
	need = lp + 1 + lr + 1;
	buf = malloc(need);
	if (!buf)
		return NULL;
	memcpy(buf, prefix, lp);
	buf[lp] = ':';
	memcpy(buf + lp + 1, rest, lr + 1);
	return buf;
}

void embedded_prepare_pytorch_ld_path_ex(const char *repo_root, int py_major, int py_minor) {
	const char *env_name = lib_path_env();
	const char *current = getenv(env_name);
	char torch_lib[4096];
	char *filtered;
	char *merged;

	find_torch_lib(repo_root, py_major, py_minor, torch_lib, sizeof(torch_lib));
	filtered = filter_lib_path(current ? current : "", torch_lib[0] ? torch_lib : NULL);
	if (!filtered)
		return;

	if (torch_lib[0]) {
		merged = join_lib_paths(torch_lib, filtered);
		free(filtered);
		if (!merged)
			return;
		setenv(env_name, merged, 1);
		fprintf(stderr,
			"ollama: embedded Python: prepended %s to %s (hostlibs/cudnn shadows removed)\n",
			torch_lib, env_name);
		free(merged);
		return;
	}

	if (*filtered)
		setenv(env_name, filtered, 1);
	else
		unsetenv(env_name);
	if (current && *current)
		fprintf(stderr,
			"ollama: embedded Python: stripped hostlibs/cudnn shadows from %s (torch/lib not found)\n",
			env_name);
	free(filtered);
}
