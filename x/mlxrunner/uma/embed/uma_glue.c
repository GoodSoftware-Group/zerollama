/*
 * mlxrunner UMA glue — machine-wide uma_daemon only (PACKAGING.md).
 *
 * Modes: ZEROLLAMA_UMA_SCHED=auto (default)|require|degraded|0
 * Coarse leases: LeaseBegin/End around load, prefill chunks, decode steps.
 * Persistent socket via uma_client fd.
 */
#include "uma_glue.h"

#include "uma/client.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

extern void goUmaMlxJob(void *ctx);

enum {
  UMA_MODE_OFF = 0,
  UMA_MODE_REQUIRE = 1,
  UMA_MODE_AUTO = 2,
  UMA_MODE_DEGRADED = 3,
};

static UmaClient *g_client;
static int g_mode;
static int g_log;
static int g_force_off;
static int g_last_failed;
static char g_last_err[256];

static uint64_t g_lease_ticket;
static int g_lease_depth;
static uint64_t g_seq;
static uint64_t g_lease_evals;
static double g_lease_wait_ms;
static double g_lease_hold_start_ms;

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
      !strcmp(e, "off") || !strcmp(e, "OFF"))
    return UMA_MODE_OFF;
  if (!strcmp(e, "auto") || !strcmp(e, "AUTO"))
    return UMA_MODE_AUTO;
  if (!strcmp(e, "degraded") || !strcmp(e, "DEGRADED"))
    return UMA_MODE_DEGRADED;
  /* 1 / true / on / require / anything else truthy */
  return UMA_MODE_REQUIRE;
}

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
  const char *suffix = NULL;
  if (phase && phase[0]) {
    if (!strcmp(phase, "load"))
      suffix = "load";
    else if (!strcmp(phase, "prefill") || !strcmp(phase, "prefill-seed"))
      suffix = "prefill";
    else if (!strcmp(phase, "decode") || !strcmp(phase, "decode-prime"))
      suffix = "decode";
  }
  if (!suffix)
    snprintf(buf, sizeof(buf), "%s", base);
  else
    snprintf(buf, sizeof(buf), "%s-%s", base, suffix);
  return buf;
}

static const char *project_name(void) { return project_name_for_phase(NULL); }

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
  char buf[1200];
  if (uma_client_cmd(c, "HELP", buf, sizeof(buf)) != 0)
    return 0;
  return strstr(buf, "HOLD_GPU") != NULL;
}

int uma_mlx_acquire(void) {
  clear_err();
  g_mode = parse_mode();
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
    logf_uma("uma_mlx: connected mode=%d uma_proto=%d project=%s (persistent)\n",
             g_mode, proto, project_name());
  return 0;
}

void uma_mlx_release(void) {
  while (g_lease_depth > 0)
    uma_mlx_lease_end();
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

static int hold_gpu_once(const char *phase, uint64_t *ticket_out,
                         double *wait_ms_out) {
  uint64_t ticket = 0;
  if (uma_client_submit(g_client, project_name_for_phase(phase), "HOLD_GPU",
                        &ticket) != 0 ||
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

static void release_gpu_once(uint64_t ticket) {
  if (!ticket)
    return;
  double hold_ms = 0;
  if (g_lease_hold_start_ms > 0)
    hold_ms = now_ms() - g_lease_hold_start_ms;
  (void)uma_client_release(g_client, ticket);
  char done[640];
  (void)uma_client_wait(g_client, ticket, 60.0, done, sizeof(done));
  g_seq++;
  g_stat_leases++;
  g_stat_evals += g_lease_evals;
  g_stat_wait_ms += g_lease_wait_ms;
  g_stat_hold_ms += hold_ms;
  if (g_log)
    logf_uma("uma_mlx: lease end ticket=%llu seq=%llu evals=%llu "
             "wait_ms=%.1f hold_ms=%.1f cum_leases=%llu cum_wait_ms=%.1f "
             "cum_hold_ms=%.1f %s\n",
             (unsigned long long)ticket, (unsigned long long)g_seq,
             (unsigned long long)g_lease_evals, g_lease_wait_ms, hold_ms,
             (unsigned long long)g_stat_leases, g_stat_wait_ms, g_stat_hold_ms,
             done);
  g_lease_evals = 0;
  g_lease_wait_ms = 0;
  g_lease_hold_start_ms = 0;
}

int uma_mlx_lease_begin(const char *phase) {
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
  if (g_lease_depth++ > 0)
    return 0;

  uint64_t ticket = 0;
  double wait_ms = 0;
  if (hold_gpu_once(phase, &ticket, &wait_ms) != 0) {
    g_lease_depth = 0;
    if (g_mode == UMA_MODE_DEGRADED) {
      logf_uma("uma_mlx: degraded — HOLD_GPU failed, ungated for this lease\n");
      g_lease_ticket = 0;
      return 0;
    }
    set_err("HOLD_GPU failed");
    return -1;
  }
  g_lease_ticket = ticket;
  g_lease_evals = 0;
  g_lease_wait_ms = wait_ms;
  g_lease_hold_start_ms = now_ms();
  if (g_log)
    logf_uma("uma_mlx: lease begin phase=%s project=%s ticket=%llu wait_ms=%.1f\n",
             phase ? phase : "?", project_name_for_phase(phase),
             (unsigned long long)ticket, wait_ms);
  return 0;
}

void uma_mlx_lease_end(void) {
  if (!uma_mlx_active())
    return;
  if (g_lease_depth <= 0)
    return;
  if (--g_lease_depth > 0)
    return;
  if (g_lease_ticket) {
    release_gpu_once(g_lease_ticket);
    g_lease_ticket = 0;
  }
}

void uma_mlx_run_gpu(void) {
  clear_err();
  if (!uma_mlx_active()) {
    goUmaMlxJob(NULL);
    return;
  }

  /* Nested under LeaseBegin: already holding GPU. */
  if (g_lease_depth > 0 && g_lease_ticket) {
    goUmaMlxJob(NULL);
    g_lease_evals++;
    return;
  }

  /* One-shot lease when caller forgot LeaseBegin. */
  uint64_t ticket = 0;
  double wait_ms = 0;
  if (hold_gpu_once("eval", &ticket, &wait_ms) != 0) {
    if (g_mode == UMA_MODE_DEGRADED) {
      logf_uma("uma_mlx: degraded — one-shot HOLD failed, ungated eval\n");
      goUmaMlxJob(NULL);
      return;
    }
    set_err("HOLD_GPU failed");
    return;
  }
  g_lease_evals = 0;
  g_lease_wait_ms = wait_ms;
  g_lease_hold_start_ms = now_ms();
  goUmaMlxJob(NULL);
  g_lease_evals++;
  release_gpu_once(ticket);
}
