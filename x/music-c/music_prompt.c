#define _DARWIN_C_SOURCE 1
#include "music_prompt.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *xstrdup(const char *s) {
  size_t n = strlen(s);
  char *p = (char *)malloc(n + 1);
  if (!p)
    return NULL;
  memcpy(p, s, n + 1);
  return p;
}

static int append_str(char **buf, size_t *len, size_t *cap, const char *s) {
  size_t n = strlen(s);
  if (*len + n + 1 > *cap) {
    size_t nc = (*cap < 64) ? 64 : *cap * 2;
    while (nc < *len + n + 1)
      nc *= 2;
    char *p = (char *)realloc(*buf, nc);
    if (!p)
      return 0;
    *buf = p;
    *cap = nc;
  }
  memcpy(*buf + *len, s, n + 1);
  *len += n;
  return 1;
}

static void lowercase_tag(char *s) {
  if (*s != '[')
    return;
  for (char *p = s + 1; *p && *p != ']'; p++)
    *p = (char)tolower((unsigned char)*p);
}

int music3_clean_caption(const char *caption, char **out) {
  if (!caption || !out)
    return 0;
  /* Special <|a b|> -> "a is b"; collapse blank lines. Markdown is best-effort. */
  size_t cap = strlen(caption) * 2 + 16;
  char *dst = (char *)malloc(cap);
  if (!dst)
    return 0;
  size_t o = 0;
  const char *p = caption;
  int nl = 0;
  while (*p) {
    if (p[0] == '<' && p[1] == '|') {
      const char *end = strstr(p, "|>");
      if (end) {
        char inner[256];
        size_t n = (size_t)(end - (p + 2));
        if (n >= sizeof(inner))
          n = sizeof(inner) - 1;
        memcpy(inner, p + 2, n);
        inner[n] = 0;
        while (n && (inner[n - 1] == ' ' || inner[n - 1] == '\t'))
          inner[--n] = 0;
        char *sp = strchr(inner, ' ');
        const char *repl;
        char tmp[320];
        if (sp) {
          *sp = 0;
          snprintf(tmp, sizeof(tmp), "%s is %s", inner, sp + 1);
          repl = tmp;
        } else {
          repl = inner;
        }
        size_t rn = strlen(repl);
        if (o + rn + 1 > cap) {
          cap = (o + rn + 1) * 2;
          char *np = (char *)realloc(dst, cap);
          if (!np) {
            free(dst);
            return 0;
          }
          dst = np;
        }
        memcpy(dst + o, repl, rn);
        o += rn;
        p = end + 2;
        nl = 0;
        continue;
      }
    }
    if (*p == '\n') {
      if (nl < 1) {
        dst[o++] = '\n';
        nl++;
      }
      p++;
      continue;
    }
    nl = 0;
    if (o + 2 > cap) {
      cap *= 2;
      char *np = (char *)realloc(dst, cap);
      if (!np) {
        free(dst);
        return 0;
      }
      dst = np;
    }
    dst[o++] = *p++;
  }
  dst[o] = 0;
  *out = dst;
  return 1;
}

int music3_normalize_lyrics(const char *lyrics, char **out) {
  if (!lyrics || !out)
    return 0;
  char *work = xstrdup(lyrics);
  if (!work)
    return 0;
  /* Drop same-line text after leading [tags]. */
  char *stripped = NULL;
  size_t slen = 0, scap = 0;
  if (!append_str(&stripped, &slen, &scap, "")) {
    free(work);
    return 0;
  }
  char *save = work;
  char *line = strsep(&save, "\n");
  int first = 1;
  while (line) {
    if (!first && !append_str(&stripped, &slen, &scap, "\n")) {
      free(work);
      free(stripped);
      return 0;
    }
    first = 0;
    const char *s = line;
    while (*s == ' ' || *s == '\t')
      s++;
    if (*s == '[') {
      const char *q = s;
      while (*q == '[') {
        const char *cls = strchr(q, ']');
        if (!cls)
          break;
        q = cls + 1;
        while (*q == ' ' || *q == '\t')
          q++;
      }
      size_t n = (size_t)(q - s);
      while (n && (s[n - 1] == ' ' || s[n - 1] == '\t'))
        n--;
      char tag[256];
      if (n >= sizeof(tag))
        n = sizeof(tag) - 1;
      memcpy(tag, s, n);
      tag[n] = 0;
      if (!append_str(&stripped, &slen, &scap, tag)) {
        free(work);
        free(stripped);
        return 0;
      }
    } else if (!append_str(&stripped, &slen, &scap, line)) {
      free(work);
      free(stripped);
      return 0;
    }
    line = strsep(&save, "\n");
  }
  free(work);

  /* "] " -> "]\n", " [" -> "\n[" */
  char *rew = NULL;
  size_t rlen = 0, rcap = 0;
  if (!append_str(&rew, &rlen, &rcap, "")) {
    free(stripped);
    return 0;
  }
  for (size_t i = 0; stripped[i]; i++) {
    if (stripped[i] == ']' && stripped[i + 1] == ' ') {
      if (!append_str(&rew, &rlen, &rcap, "]\n")) {
        free(stripped);
        free(rew);
        return 0;
      }
      i++;
      continue;
    }
    if (stripped[i] == ' ' && stripped[i + 1] == '[') {
      if (!append_str(&rew, &rlen, &rcap, "\n[")) {
        free(stripped);
        free(rew);
        return 0;
      }
      i++;
      continue;
    }
    if (stripped[i] == ' ' && stripped[i + 1] == '^' && stripped[i + 2] == ' ') {
      if (!append_str(&rew, &rlen, &rcap, "\n")) {
        free(stripped);
        free(rew);
        return 0;
      }
      i += 2;
      continue;
    }
    char ch[2] = {stripped[i], 0};
    if (!append_str(&rew, &rlen, &rcap, ch)) {
      free(stripped);
      free(rew);
      return 0;
    }
  }
  free(stripped);

  for (char *t = rew; *t; t++) {
    if (*t == '[')
      lowercase_tag(t);
  }

  char *final = NULL;
  size_t flen = 0, fcap = 0;
  if (!append_str(&final, &flen, &fcap, "[start]\n") ||
      !append_str(&final, &flen, &fcap, rew)) {
    free(rew);
    free(final);
    return 0;
  }
  free(rew);
  *out = final;
  return 1;
}

int music3_build_prompt(const char *caption, const char *lyrics, char **out) {
  char *cap = NULL;
  char *ly = NULL;
  if (!music3_clean_caption(caption ? caption : "", &cap))
    return 0;
  if (!music3_normalize_lyrics(lyrics ? lyrics : "", &ly)) {
    free(cap);
    return 0;
  }
  char *p = NULL;
  size_t len = 0, c = 0;
  int ok = append_str(&p, &len, &c, "<|im_start|><|caption_start|>") &&
           append_str(&p, &len, &c, cap) &&
           append_str(&p, &len, &c, "<|caption_end|><|lyrics_start|>") &&
           append_str(&p, &len, &c, ly) &&
           append_str(&p, &len, &c, "<|lyrics_end|><|im_end|><|audio_start|>");
  free(cap);
  free(ly);
  if (!ok) {
    free(p);
    return 0;
  }
  *out = p;
  return 1;
}
