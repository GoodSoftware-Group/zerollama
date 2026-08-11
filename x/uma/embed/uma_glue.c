/*
 * Darwin Metal/ANE/AMX clients ↔ machine-wide uma_daemon (PACKAGING.md).
 *
 * Modes: ZEROLLAMA_UMA_SCHED=auto (default)|require|degraded|off|0|disabled
 * Coarse leases: LeaseBegin/End around load, prefill chunks, decode steps.
 * Multi-unit: HOLD_GPU / HOLD_ANE / HOLD_AMX (F0390) — independent tickets.
 * Persistent socket via uma_client fd.
 */
#include "uma_glue.h"

#include "uma/client.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

extern void goUmaMlxJob(void *ctx);

enum {
  UMA_MODE_OFF = 0,
  UMA_MODE_REQUIRE = 1,
  UMA_MODE_AUTO = 2,
  UMA_MODE_DEGRADED = 3,
};

enum {
  UMA_U_GPU = 0,
  UMA_U_ANE = 1,
  UMA_U_AMX = 2,
  UMA_U_COUNT = 3,
};

typedef struct {
  uint64_t ticket;
  int depth;
  uint64_t evals;
  double wait_ms;
  double hold_start_ms;
} UmaUnitLease;

static UmaClient *g_client;
static int g_mode;
static int g_log;
static int g_force_off;
static int g_last_failed;
static int g_grain_op; /* F0625: ZEROLLAMA_UMA_GRAIN=op */
static char g_last_err[256];

static UmaUnitLease g_units[UMA_U_COUNT];
static uint64_t g_seq;

static uint64_t g_stat_leases;
static uint64_t g_stat_evals;
static double g_stat_wait_ms;
static double g_stat_hold_ms;

static double now_ms(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    return 0;
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static void logf_uma(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fflush(stderr);
}

static void set_err(const char *msg) {
  g_last_failed = 1;
  snprintf(g_last_err, sizeof(g_last_err), "%s", msg ? msg : "uma_mlx error");
  logf_uma("uma_mlx: %s\n", g_last_err);
}

static void clear_err(void) {
  g_last_failed = 0;
  g_last_err[0] = 0;
}

static int parse_mode(void) {
  const char *e = getenv("ZEROLLAMA_UMA_SCHED");
  /* Default auto: gate when uma_daemon is up, else ungated MLX. */
  if (!e || !*e)
    return UMA_MODE_AUTO;
  if (e[0] == '0' && e[1] == '\0')
    return UMA_MODE_OFF;
  if (!strcmp(e, "false") || !strcmp(e, "False") || !strcmp(e, "no") ||
      !strcmp(e, "off") || !strcmp(e, "OFF") || !strcmp(e, "disabled") ||
      !strcmp(e, "disable") || !strcmp(e, "none") || !strcmp(e, "NONE"))
    return UMA_MODE_OFF;
  if (!strcmp(e, "auto") || !strcmp(e, "AUTO"))
    return UMA_MODE_AUTO;
  if (!strcmp(e, "degraded") || !strcmp(e, "DEGRADED"))
    return UMA_MODE_DEGRADED;
  /* 1 / true / on / require / anything else truthy */
  return UMA_MODE_REQUIRE;
}

/* F0625: default phase (coarse leases). op = per-Eval one-shot HOLD. */
static int parse_grain_op(void) {
  const char *e = getenv("ZEROLLAMA_UMA_GRAIN");
  if (!e || !*e)
    return 0;
  if (!strcasecmp(e, "op") || !strcasecmp(e, "eval") || !strcasecmp(e, "fine"))
    return 1;
  return 0; /* phase | coarse | unset */
}

const char *uma_mlx_grain(void) { return g_grain_op ? "op" : "phase"; }

static int env_truthy(const char *name) {
  const char *e = getenv(name);
  if (!e || !*e)
    return 0;
  if (e[0] == '0' && e[1] == '\0')
    return 0;
  if (!strcmp(e, "false") || !strcmp(e, "off") || !strcmp(e, "no"))
    return 0;
  return 1;
}

static const char *project_base(void) {
  const char *e = getenv("UMA_JOB_NAME");
  if (e && e[0])
    return e;
  e = getenv("UMA_PROJECT");
  if (e && e[0])
    return e;
  return "mlxrunner";
}

/*
 * Prefill vs decode project names (wishlist): broker UI / QUEUE show
 * mlxrunner-load|prefill|decode unless UMA_PROJECT_FLAT=1.
 */
static const char *project_name_for_phase(const char *phase) {
  static char buf[160];
  const char *base = project_base();
  if (env_truthy("UMA_PROJECT_FLAT")) {
    snprintf(buf, sizeof(buf), "%s", base);
    return buf;
  }
  if (phase && phase[0])
    snprintf(buf, sizeof(buf), "%s-%s", base, phase);
  else
    snprintf(buf, sizeof(buf), "%s", base);
  return buf;
}

static const char *project_name(void) { return project_base(); }

static int parse_unit(const char *unit) {
  if (!unit || !*unit)
    return UMA_U_GPU;
  if (!strcasecmp(unit, "gpu") || !strcasecmp(unit, "metal") ||
      !strcasecmp(unit, "HOLD_GPU"))
    return UMA_U_GPU;
  if (!strcasecmp(unit, "ane") || !strcasecmp(unit, "HOLD_ANE"))
    return UMA_U_ANE;
  if (!strcasecmp(unit, "amx") || !strcasecmp(unit, "HOLD_AMX"))
    return UMA_U_AMX;
  return -1;
}

static const char *hold_job(int u) {
  switch (u) {
  case UMA_U_ANE:
    return "HOLD_ANE";
  case UMA_U_AMX:
    return "HOLD_AMX";
  default:
    return "HOLD_GPU";
  }
}

static const char *unit_name(int u) {
  switch (u) {
  case UMA_U_ANE:
    return "ane";
  case UMA_U_AMX:
    return "amx";
  default:
    return "gpu";
  }
}

int uma_mlx_runtime_enabled(void) {
  if (g_force_off)
    return 0;
  return parse_mode() != UMA_MODE_OFF;
}

int uma_mlx_last_failed(void) { return g_last_failed; }

const char *uma_mlx_last_error(void) {
  return g_last_err[0] ? g_last_err : "";
}

static int broker_supports_hold_gpu(UmaClient *c) {
  char buf[4096];
  if (uma_client_cmd(c, "HELP", buf, sizeof(buf)) != 0)
    return 0;
  return strstr(buf, "HOLD_GPU") != NULL;
}

int uma_mlx_acquire(void) {
  clear_err();
  g_mode = parse_mode();
  g_grain_op = parse_grain_op(); /* refresh even if already connected */
  g_log = env_truthy("ZEROLLAMA_UMA_SCHED_LOG");
  if (g_mode == UMA_MODE_OFF)
    return 0;
  if (g_client)
    return 0;

  g_client = uma_client_connect(NULL);
  if (!g_client) {
    if (g_mode == UMA_MODE_AUTO) {
      logf_uma("uma_mlx: auto — broker not running, MLX ungated "
               "(socket %s)\n",
               UMA_DAEMON_SOCK_PATH);
      g_force_off = 1;
      return 0;
    }
    set_err("broker not running (start uma_daemon)");
    return -1;
  }
  if (uma_client_ping(g_client) != 0) {
    uma_client_close(g_client);
    g_client = NULL;
    if (g_mode == UMA_MODE_AUTO) {
      logf_uma("uma_mlx: auto — PING failed, MLX ungated\n");
      g_force_off = 1;
      return 0;
    }
    set_err("broker PING failed");
    return -1;
  }
  if (!broker_supports_hold_gpu(g_client)) {
    uma_client_close(g_client);
    g_client = NULL;
    if (g_mode == UMA_MODE_AUTO) {
      logf_uma("uma_mlx: auto — broker lacks HOLD_GPU, MLX ungated "
               "(upgrade uma_daemon)\n");
      g_force_off = 1;
      return 0;
    }
    set_err("broker lacks HOLD_GPU (upgrade uma_daemon)");
    return -1;
  }
  int proto = 0;
  (void)uma_client_proto(g_client, &proto);
  if (g_log)
    logf_uma("uma_mlx: connected mode=%d grain=%s uma_proto=%d project=%s "
             "(persistent)\n",
             g_mode, uma_mlx_grain(), proto, project_name());
  return 0;
}

static void release_unit_once(int u, uint64_t ticket);

void uma_mlx_release(void) {
  for (int u = 0; u < UMA_U_COUNT; u++) {
    while (g_units[u].depth > 0)
      uma_mlx_lease_end_unit(unit_name(u));
  }
  if (!g_client)
    return;
  if (g_stat_leases > 0)
    logf_uma("uma_mlx: stats leases=%llu evals=%llu wait_ms_total=%.1f "
             "hold_ms_total=%.1f\n",
             (unsigned long long)g_stat_leases,
             (unsigned long long)g_stat_evals, g_stat_wait_ms, g_stat_hold_ms);
  uma_client_close(g_client);
  g_client = NULL;
  if (g_log)
    logf_uma("uma_mlx: disconnected from broker\n");
}

void uma_mlx_stats(uint64_t *leases, uint64_t *evals, double *wait_ms_total,
                   double *hold_ms_total) {
  if (leases)
    *leases = g_stat_leases;
  if (evals)
    *evals = g_stat_evals;
  if (wait_ms_total)
    *wait_ms_total = g_stat_wait_ms;
  if (hold_ms_total)
    *hold_ms_total = g_stat_hold_ms;
}

int uma_mlx_active(void) {
  return g_client != NULL && !g_force_off && g_mode != UMA_MODE_OFF;
}

static int wait_holding(uint64_t ticket, double *wait_ms_out) {
  double t0 = now_ms();
  for (int i = 0; i < 60000; i++) {
    char buf[640];
    if (uma_client_job(g_client, ticket, buf, sizeof(buf)) != 0)
      return -1;
    if (strstr(buf, "phase=holding")) {
      if (wait_ms_out)
        *wait_ms_out = now_ms() - t0;
      return 0;
    }
    if (strstr(buf, "state=done") || strstr(buf, "state=err") ||
        strstr(buf, "ERR "))
      return -1;
    usleep(1000);
  }
  return -1;
}

static int hold_unit_once(int u, const char *phase, uint64_t *ticket_out,
                          double *wait_ms_out) {
  uint64_t ticket = 0;
  const char *job = hold_job(u);
  if (uma_client_submit(g_client, project_name_for_phase(phase), job, &ticket) !=
          0 ||
      ticket == 0)
    return -1;
  if (wait_holding(ticket, wait_ms_out) != 0) {
    (void)uma_client_release(g_client, ticket);
    char buf[256];
    (void)uma_client_wait(g_client, ticket, 5.0, buf, sizeof(buf));
    return -1;
  }
  *ticket_out = ticket;
  return 0;
}

static void release_unit_once(int u, uint64_t ticket) {
  if (!ticket)
    return;
  UmaUnitLease *L = &g_units[u];
  double hold_ms = 0;
  if (L->hold_start_ms > 0)
    hold_ms = now_ms() - L->hold_start_ms;
  (void)uma_client_release(g_client, ticket);
  char done[640];
  (void)uma_client_wait(g_client, ticket, 60.0, done, sizeof(done));
  g_seq++;
  g_stat_leases++;
  g_stat_evals += L->evals;
  g_stat_wait_ms += L->wait_ms;
  g_stat_hold_ms += hold_ms;
  if (g_log)
    logf_uma("uma_mlx: lease end unit=%s ticket=%llu seq=%llu evals=%llu "
             "wait_ms=%.1f hold_ms=%.1f cum_leases=%llu cum_wait_ms=%.1f "
             "cum_hold_ms=%.1f %s\n",
             unit_name(u), (unsigned long long)ticket, (unsigned long long)g_seq,
             (unsigned long long)L->evals, L->wait_ms, hold_ms,
             (unsigned long long)g_stat_leases, g_stat_wait_ms, g_stat_hold_ms,
             done);
  L->evals = 0;
  L->wait_ms = 0;
  L->hold_start_ms = 0;
}

static int lease_begin_u(int u, const char *phase) {
  clear_err();
  /* auto: if acquire saw a down broker, re-probe so a late UMAStatus.app works. */
  if (g_force_off && g_mode == UMA_MODE_AUTO) {
    g_force_off = 0;
    if (g_client) {
      uma_client_close(g_client);
      g_client = NULL;
    }
    if (uma_mlx_acquire() != 0)
      return 0; /* still ungated */
  }
  if (!uma_mlx_active())
    return 0;
  /* F0625 grain=op: skip coarse HOLD — each RunGPU/Eval is one-shot. */
  if (g_grain_op) {
    if (g_log)
      logf_uma("uma_mlx: grain=op — skip coarse lease begin unit=%s phase=%s\n",
               unit_name(u), phase ? phase : "?");
    return 0;
  }
  UmaUnitLease *L = &g_units[u];
  if (L->depth++ > 0)
    return 0;

  uint64_t ticket = 0;
  double wait_ms = 0;
  if (hold_unit_once(u, phase, &ticket, &wait_ms) != 0) {
    L->depth = 0;
    if (g_mode == UMA_MODE_DEGRADED) {
      logf_uma("uma_mlx: degraded — %s failed, ungated for this lease\n",
               hold_job(u));
      L->ticket = 0;
      return 0;
    }
    char msg[64];
    snprintf(msg, sizeof(msg), "%s failed", hold_job(u));
    set_err(msg);
    return -1;
  }
  L->ticket = ticket;
  L->evals = 0;
  L->wait_ms = wait_ms;
  L->hold_start_ms = now_ms();
  if (g_log)
    logf_uma("uma_mlx: lease begin unit=%s phase=%s project=%s ticket=%llu "
             "wait_ms=%.1f\n",
             unit_name(u), phase ? phase : "?", project_name_for_phase(phase),
             (unsigned long long)ticket, wait_ms);
  return 0;
}

static void lease_end_u(int u) {
  if (!uma_mlx_active())
    return;
  UmaUnitLease *L = &g_units[u];
  if (L->depth <= 0)
    return;
  if (--L->depth > 0)
    return;
  if (L->ticket) {
    release_unit_once(u, L->ticket);
    L->ticket = 0;
  }
}

static void run_u(int u) {
  clear_err();
  if (!uma_mlx_active()) {
    goUmaMlxJob(NULL);
    return;
  }

  UmaUnitLease *L = &g_units[u];
  /* Nested under LeaseBegin: already holding this unit. */
  if (L->depth > 0 && L->ticket) {
    goUmaMlxJob(NULL);
    L->evals++;
    return;
  }

  /* One-shot lease when no coarse LeaseBegin (or grain=op). */
  uint64_t ticket = 0;
  double wait_ms = 0;
  const char *phase = g_grain_op ? "eval" : unit_name(u);
  if (hold_unit_once(u, phase, &ticket, &wait_ms) != 0) {
    if (g_mode == UMA_MODE_DEGRADED) {
      logf_uma("uma_mlx: degraded — one-shot %s failed, ungated\n", hold_job(u));
      goUmaMlxJob(NULL);
      return;
    }
    char msg[64];
    snprintf(msg, sizeof(msg), "%s failed", hold_job(u));
    set_err(msg);
    return;
  }
  L->evals = 0;
  L->wait_ms = wait_ms;
  L->hold_start_ms = now_ms();
  goUmaMlxJob(NULL);
  L->evals++;
  release_unit_once(u, ticket);
}

int uma_mlx_lease_begin_unit(const char *unit, const char *phase) {
  int u = parse_unit(unit);
  if (u < 0) {
    set_err("unknown unit (want gpu|ane|amx)");
    return -1;
  }
  return lease_begin_u(u, phase);
}

void uma_mlx_lease_end_unit(const char *unit) {
  int u = parse_unit(unit);
  if (u < 0)
    return;
  lease_end_u(u);
}

int uma_mlx_lease_begin(const char *phase) {
  return lease_begin_u(UMA_U_GPU, phase);
}

void uma_mlx_lease_end(void) { lease_end_u(UMA_U_GPU); }

void uma_mlx_run_gpu(void) { run_u(UMA_U_GPU); }

void uma_mlx_run_unit(const char *unit) {
  int u = parse_unit(unit);
  if (u < 0) {
    set_err("unknown unit (want gpu|ane|amx)");
    return;
  }
  run_u(u);
}

/* --- GRAPH helpers (F0624) ------------------------------------------------ */

int uma_mlx_format_graph(char *out, size_t n, int ntok, const char *form,
                         const char *nodes) {
  return uma_client_format_graph(out, n, ntok, form, nodes);
}

int uma_mlx_format_graph_ex(char *out, size_t n, const char *level, int ntok,
                            const char *form, const char *nodes, int ngen,
                            int eos, const char *toks) {
  return uma_client_format_graph_ex(out, n, level, ntok, form, nodes, ngen, eos,
                                    toks);
}

int uma_mlx_submit(const char *project, const char *job, uint64_t *ticket_out) {
  clear_err();
  if (!g_client) {
    set_err("not connected (Acquire first)");
    return -1;
  }
  if (!job || !job[0]) {
    set_err("empty job");
    return -1;
  }
  const char *proj = (project && project[0]) ? project : project_name();
  uint64_t ticket = 0;
  if (uma_client_submit(g_client, proj, job, &ticket) != 0 || ticket == 0) {
    set_err("SUBMIT failed");
    return -1;
  }
  if (ticket_out)
    *ticket_out = ticket;
  return 0;
}

int uma_mlx_wait(uint64_t ticket, double timeout_s, char *buf, size_t n) {
  clear_err();
  if (!g_client) {
    set_err("not connected (Acquire first)");
    return -1;
  }
  if (!buf || n == 0) {
    set_err("wait buf required");
    return -1;
  }
  if (timeout_s <= 0)
    timeout_s = 60.0;
  if (uma_client_wait(g_client, ticket, timeout_s, buf, n) != 0) {
    char msg[280];
    snprintf(msg, sizeof(msg), "WAIT failed: %.200s", buf[0] ? buf : "(empty)");
    set_err(msg);
    return -1;
  }
  return 0;
}

int uma_mlx_graph(const char *project, const char *job, double timeout_s,
                  char *buf, size_t n) {
  uint64_t ticket = 0;
  if (uma_mlx_submit(project, job, &ticket) != 0)
    return -1;
  return uma_mlx_wait(ticket, timeout_s, buf, n);
}

/* --- BUF helpers (F0627) -------------------------------------------------- */

int uma_mlx_buf_alloc(const char *name, size_t nbytes) {
  clear_err();
  if (!g_client) {
    set_err("not connected (Acquire first)");
    return -1;
  }
  char resp[320];
  if (uma_client_buf_alloc(g_client, name, nbytes, resp, sizeof(resp)) != 0) {
    set_err(resp[0] ? resp : "BUF_ALLOC failed");
    return -1;
  }
  return 0;
}

int uma_mlx_buf_free(const char *name) {
  clear_err();
  if (!g_client) {
    set_err("not connected (Acquire first)");
    return -1;
  }
  char resp[320];
  (void)uma_client_buf_free(g_client, name, resp, sizeof(resp));
  return 0;
}

int uma_mlx_buf_put(const char *name, const void *data, size_t nbytes) {
  clear_err();
  if (!g_client) {
    set_err("not connected (Acquire first)");
    return -1;
  }
  char resp[320];
  if (uma_client_buf_put(g_client, name, data, nbytes, resp, sizeof(resp)) !=
      0) {
    set_err(resp[0] ? resp : "BUF_PUT failed");
    return -1;
  }
  return 0;
}

int uma_mlx_buf_get(const char *name, void *dst, size_t dst_n, size_t *got) {
  clear_err();
  if (!g_client) {
    set_err("not connected (Acquire first)");
    return -1;
  }
  char resp[320];
  size_t g = 0;
  if (uma_client_buf_get(g_client, name, dst, dst_n, &g, resp, sizeof(resp)) !=
      0) {
    set_err(resp[0] ? resp : "BUF_GET failed");
    return -1;
  }
  if (got)
    *got = g;
  return 0;
}

int uma_mlx_buf_export(const char *name, uint32_t *iosurface_id_out,
                       uint32_t *token_out) {
  clear_err();
  if (!g_client) {
    set_err("not connected (Acquire first)");
    return -1;
  }
  char resp[640];
  if (uma_client_buf_export(g_client, name, resp, sizeof(resp)) != 0) {
    set_err(resp[0] ? resp : "BUF_EXPORT failed");
    return -1;
  }
  uint32_t id = 0, tok = 0;
  if (uma_client_parse_iosurface_id(resp, &id) != 0 ||
      uma_client_parse_export_token(resp, &tok) != 0) {
    set_err("BUF_EXPORT missing iosurface_id/token");
    return -1;
  }
  if (iosurface_id_out)
    *iosurface_id_out = id;
  if (token_out)
    *token_out = tok;
  return 0;
}

int uma_mlx_buf_reclaim(const char *name, uint32_t token) {
  clear_err();
  if (!g_client) {
    set_err("not connected (Acquire first)");
    return -1;
  }
  char resp[320];
  if (uma_client_buf_reclaim(g_client, name, token, resp, sizeof(resp)) != 0) {
    set_err(resp[0] ? resp : "BUF_RECLAIM failed");
    return -1;
  }
  return 0;
}
