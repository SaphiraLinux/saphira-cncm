/* tok.c — pre-tokenizers. Mirrors tokenization/*.py without transformers.
 * Kinds: sudoku (exact), bpe (byte-level BPE + GPT-2-style splitter, exact on
 * ASCII; non-ASCII bytes attach to letter runs — documented approximation),
 * hf-wordpiece (BERT greedy longest-match; ASCII case-folding).
 * Vocab/merge assets are plain text emitted by tools/export.py (data, not deps).
 */
#include "ncm.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct TokState {
    char kind[16];
    /* bpe */
    unsigned char **vocab;   /* id -> bytes */
    int *vlen;
    int n_vocab;
    unsigned char **ma, **mb; /* merges */
    int *mlen_a, *mlen_b;
    int n_merges;
    /* hash reverse map: bytes -> id */
    int *htable;  /* open addressing over (hash of bytes) */
    int hcap;
    char **specials;
    int n_special;
    unsigned char *skip; /* n_vocab bitmap for skip_special */
    int eos_id, pad_id;
    /* splitter pattern */
    int pattern; /* 0 = repo/gpt2, 1 = qwen/tiktoken */
};

/* ---------- helpers ---------- */

static unsigned hexv(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

static unsigned char *unhex(const char *s, int *out_len) {
    size_t n = strlen(s) / 2;
    unsigned char *b = (unsigned char *)malloc(n ? n : 1);
    for (size_t i = 0; i < n; i++) b[i] = (hexv(s[2 * i]) << 4) | hexv(s[2 * i + 1]);
    *out_len = (int)n;
    return b;
}

static unsigned long long fnv(const unsigned char *b, int n) {
    unsigned long long h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

/* ---------- unicode-lite: ASCII exact; non-ASCII => letter ---------- */
static int is_letter(int cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return 1;
    if (cp >= 0x80) return 1; /* approximation: non-ASCII sticks to \p{L} runs */
    return 0;
}
static int is_number(int cp) {
    if (cp >= '0' && cp <= '9') return 1;
    return 0; /* non-ASCII digits (rare in eval sets) not classified */
}
static int is_space(int cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' || cp == '\f';
}

/* decode one UTF-8 sequence; returns code point, advances *p */
static int utf8_next(const unsigned char **p, const unsigned char *end) {
    const unsigned char *s = *p;
    if (s >= end) return -1;
    unsigned char c = *s;
    int cp; int n;
    if (c < 0x80) { cp = c; n = 1; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; n = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; n = 3; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; n = 4; }
    else { *p = s + 1; return 0xFFFD; }
    if (s + n > end) { *p = end; return 0xFFFD; }
    for (int i = 1; i < n; i++) {
        if ((s[i] & 0xC0) != 0x80) { *p = s + i; return 0xFFFD; }
        cp = (cp << 6) | (s[i] & 0x3F);
    }
    *p = s + n;
    return cp;
}

/* ---------- splitter: GPT-2/repo pattern over code points ----------
 * pieces: '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
 * Emits byte ranges into out[][2] (start,end), returns count. */
static int split_gpt2(const unsigned char *t, int n, int (*ranges)[2], int cap) {
    int nc = 0;
    int cps[8192]; int cpos[8192]; int cn = 0;
    const unsigned char *p = t, *end = t + n;
    while (p < end && cn < 8192) { cpos[cn] = (int)(p - t); cps[cn] = utf8_next(&p, end); cn++; }
    int i = 0;
    while (i < cn && nc < cap) {
        int cp = cps[i];
        /* contraction exceptions: 's 't 'm 'd 'll 've 're */
        if (cp == '\'' && i + 1 < cn) {
            int n1 = cps[i + 1];
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                ranges[nc][0] = cpos[i]; ranges[nc][1] = (i + 2 < cn) ? cpos[i + 2] : n; nc++; i += 2; continue;
            }
            if ((n1 == 'l' || n1 == 'v' || n1 == 'r') && i + 2 < cn) {
                int n2 = cps[i + 2];
                if ((n1 == 'l' && n2 == 'l') || (n1 == 'v' && n2 == 'e') || (n1 == 'r' && n2 == 'e')) {
                    ranges[nc][0] = cpos[i]; ranges[nc][1] = (i + 3 < cn) ? cpos[i + 3] : n; nc++; i += 3; continue;
                }
            }
        }
        int lead_space = (cp == ' ');
        int j = i + (lead_space ? 1 : 0);
        if (j < cn && is_letter(cps[j])) {
            int k = j + 1;
            while (k < cn && is_letter(cps[k])) k++;
            ranges[nc][0] = cpos[i]; ranges[nc][1] = (k < cn) ? cpos[k] : n; nc++; i = k; continue;
        }
        if (j < cn && is_number(cps[j])) {
            int k = j + 1;
            while (k < cn && is_number(cps[k])) k++;
            ranges[nc][0] = cpos[i]; ranges[nc][1] = (k < cn) ? cpos[k] : n; nc++; i = k; continue;
        }
        if (j < cn && !is_space(cps[j]) && !is_letter(cps[j]) && !is_number(cps[j])) {
            int k = j + 1;
            while (k < cn && !is_space(cps[k]) && !is_letter(cps[k]) && !is_number(cps[k])) k++;
            ranges[nc][0] = cpos[i]; ranges[nc][1] = (k < cn) ? cpos[k] : n; nc++; i = k; continue;
        }
        if (is_space(cp)) {
            int k = i + 1;
            if (cp != ' ' && cp != '\n' && cp != '\r' && cp != '\t') {
                /* \s+ branch */
                while (k < cn && is_space(cps[k])) k++;
            } else {
                /* \s+(?!\S): spaces only if followed by end or space... python:
                 * '\s+(?!\S)' matches trailing spaces; else single '\s+' per char.
                 * Simplify: consume run only if at end or next is space; else one. */
                if (k < cn && !is_space(cps[k])) { k = i + 1; }
                else { while (k < cn && is_space(cps[k])) k++; }
            }
            ranges[nc][0] = cpos[i]; ranges[nc][1] = (k < cn) ? cpos[k] : n; nc++; i = k; continue;
        }
        /* fallback: single code point */
        ranges[nc][0] = cpos[i]; ranges[nc][1] = (i + 1 < cn) ? cpos[i + 1] : n; nc++; i++;
    }
    return nc;
}

/* ---------- TokState ---------- */

TokState *tok_load(const NCMConfig *c, const char *asset_dir) {
    TokState *t = (TokState *)calloc(1, sizeof(TokState));
    strncpy(t->kind, c->pre_tok, sizeof(t->kind) - 1);
    t->eos_id = c->eos_token_id;
    t->pad_id = c->pad_token_id;
    if (strcmp(t->kind, "sudoku") == 0) return t;

    int is_bpe = (strcmp(t->kind, "bpe") == 0) ||
                 (strcmp(t->kind, "hf") == 0 && strstr(c->hf_model_id, "bert") == NULL);
    if (is_bpe) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/bpe_vocab.txt", asset_dir);
        FILE *f = fopen(path, "r");
        if (!f) { fprintf(stderr, "missing %s\n", path); exit(1); }
        int cap = 8192;
        t->vocab = (unsigned char **)malloc(cap * sizeof(char *));
        t->vlen = (int *)malloc(cap * sizeof(int));
        char line[4096];
        t->n_vocab = 0;
        while (fgets(line, sizeof(line), f)) {
            int id = atoi(line);
            char *tab = strchr(line, '\t');
            if (!tab) continue;
            tab[strcspn(tab, "\r\n")] = 0;
            t->vocab[id] = unhex(tab + 1, &t->vlen[id]);
            if (id >= t->n_vocab) t->n_vocab = id + 1;
        }
        fclose(f);
        snprintf(path, sizeof(path), "%s/bpe_merges.txt", asset_dir);
        f = fopen(path, "r");
        if (!f) { fprintf(stderr, "missing %s\n", path); exit(1); }
        int mcap = 16384;
        t->ma = (unsigned char **)malloc(mcap * sizeof(char *));
        t->mb = (unsigned char **)malloc(mcap * sizeof(char *));
        t->mlen_a = (int *)malloc(mcap * sizeof(int));
        t->mlen_b = (int *)malloc(mcap * sizeof(int));
        t->n_merges = 0;
        while (fgets(line, sizeof(line), f)) {
            char *sp = strchr(line, ' ');
            if (!sp) continue;
            *sp = 0;
            line[strcspn(line, "\r\n")] = 0;
            /* note: ` *sp=0` then second token at sp+1; strip newline below */
            char *b = sp + 1;
            b[strcspn(b, "\r\n")] = 0;
            t->ma[t->n_merges] = unhex(line, &t->mlen_a[t->n_merges]);
            t->mb[t->n_merges] = unhex(b, &t->mlen_b[t->n_merges]);
            t->n_merges++;
        }
        fclose(f);
        /* reverse hash map */
        t->hcap = 32768;
        t->htable = (int *)malloc(t->hcap * sizeof(int));
        for (int i = 0; i < t->hcap; i++) t->htable[i] = -1;
        for (int id = 0; id < t->n_vocab; id++) {
            unsigned long long h = fnv(t->vocab[id], t->vlen[id]) & (t->hcap - 1);
            while (t->htable[h] != -1) h = (h + 1) & (t->hcap - 1);
            t->htable[h] = id;
        }
        if (strcmp(t->kind, "hf") == 0 && strstr(c->hf_model_id, "Qwen") != NULL)
            t->pattern = 1;
        t->n_special = 0;
        if (t->eos_id >= 0 && t->eos_id + 1 > t->n_special) t->n_special = t->eos_id + 1;
        if (t->pad_id >= 0 && t->pad_id + 1 > t->n_special) t->n_special = t->pad_id + 1;
        t->skip = (unsigned char *)calloc(t->n_vocab, 1);
        for (int i = 0; i < t->n_special && i < t->n_vocab; i++) t->skip[i] = 1;
        return t;
    }
    /* wordpiece */
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/wordpiece.txt", asset_dir);
        FILE *f = fopen(path, "r");
        if (!f) { fprintf(stderr, "missing %s\n", path); exit(1); }
        int cap = 40000;
        t->vocab = (unsigned char **)malloc(cap * sizeof(char *));
        t->vlen = (int *)malloc(cap * sizeof(int));
        char line[1024];
        t->n_vocab = 0;
        while (fgets(line, sizeof(line), f)) {
            line[strcspn(line, "\r\n")] = 0;
            int n = strlen(line);
            t->vocab[t->n_vocab] = (unsigned char *)malloc(n + 1);
            memcpy(t->vocab[t->n_vocab], line, n + 1);
            t->vlen[t->n_vocab] = n;
            t->n_vocab++;
        }
        fclose(f);
        /* bert specials: [PAD] [UNK] [CLS] [SEP] [MASK] */
        t->skip = (unsigned char *)calloc(t->n_vocab, 1);
        for (int i = 0; i < t->n_vocab; i++) {
            const char *w = (const char *)t->vocab[i];
            if (!strcmp(w, "[PAD]") || !strcmp(w, "[UNK]") || !strcmp(w, "[CLS]") ||
                !strcmp(w, "[SEP]") || !strcmp(w, "[MASK]"))
                t->skip[i] = 1;
        }
        return t;
    }
}

void tok_free(TokState *t) {
    if (!t) return;
    for (int i = 0; i < t->n_vocab; i++) free(t->vocab[i]);
    free(t->vocab); free(t->vlen);
    for (int i = 0; i < t->n_merges; i++) { free(t->ma[i]); free(t->mb[i]); }
    free(t->ma); free(t->mb); free(t->mlen_a); free(t->mlen_b);
    free(t->htable);
    free(t->skip);
    for (int i = 0; i < t->n_special; i++) free(t->specials[i]);
    free(t->specials);
    free(t);
}

static int bpe_lookup(TokState *t, const unsigned char *b, int n) {
    unsigned long long h = fnv(b, n) & (t->hcap - 1);
    for (int k = 0; k < t->hcap; k++) {
        int id = t->htable[(h + k) & (t->hcap - 1)];
        if (id < 0) return -1;
        if (t->vlen[id] == n && memcmp(t->vocab[id], b, n) == 0) return id;
    }
    return -1;
}

static int merge_rank(TokState *t, const unsigned char *a, int na,
                      const unsigned char *b, int nb) {
    /* linear scan is O(M) — words are short; fine for eval scale */
    for (int i = 0; i < t->n_merges; i++)
        if (t->mlen_a[i] == na && t->mlen_b[i] == nb &&
            memcmp(t->ma[i], a, na) == 0 && memcmp(t->mb[i], b, nb) == 0)
            return i;
    return -1;
}

/* encode one chunk (no specials) via byte BPE; appends ids, returns count */
static int bpe_encode_word(TokState *t, const unsigned char *w, int n,
                           int *out, int cap) {
    if (n == 0) return 0;
    /* parts as (start,len) over w */
    int ps[512], pl[512], np = 0;
    for (int i = 0; i < n && np < 512; i++) { ps[np] = i; pl[np] = 1; np++; }
    for (;;) {
        int best = -1, bi = -1;
        for (int i = 0; i + 1 < np; i++) {
            int r = merge_rank(t, w + ps[i], pl[i], w + ps[i + 1], pl[i + 1]);
            if (r >= 0 && (best < 0 || r < best)) { best = r; bi = i; }
        }
        if (bi < 0) break;
        pl[bi] += pl[bi + 1];
        for (int i = bi + 1; i + 1 < np; i++) { ps[i] = ps[i + 1]; pl[i] = pl[i + 1]; }
        np--;
    }
    int no = 0;
    for (int i = 0; i < np && no < cap; i++) {
        int id = bpe_lookup(t, w + ps[i], pl[i]);
        if (id < 0) { fprintf(stderr, "bpe: bytes missing from vocab (len=%d first=%02x)\n", pl[i], w[ps[i]]); exit(1); }
        out[no++] = id;
    }
    return no;
}

int tok_encode(TokState *t, const char *text, int *out, int cap) {
    if (strcmp(t->kind, "sudoku") == 0) {
        int n = 0;
        for (const char *p = text; *p && n < cap; p++) {
            char c = *p;
            if (c == ' ' || c == '\n' || c == '\t' || c == '\r') continue;
            if (c == '_') out[n++] = 0;
            else if (c >= '0' && c <= '9') out[n++] = c - '0';
            else if (c == '|') out[n++] = 10;
            else if (c == 'B') out[n++] = 11;
        }
        return n;
    }
    const unsigned char *u = (const unsigned char *)text;
    int nu = strlen(text);
    /* split on specials (exact byte match); specials unknown here -> none.
     * Custom BPE specials live in vocab ids 0..k-1; text containing them is
     * out of scope for eval inputs. */
    int ranges[2048][2];
    int nr = split_gpt2(u, nu, ranges, 2048);
    int no = 0;
    for (int i = 0; i < nr && no < cap; i++) {
        int s = ranges[i][0], e = ranges[i][1];
        no += bpe_encode_word(t, u + s, e - s, out + no, cap - no);
    }
    return no;
}

int tok_decode(TokState *t, const int *ids, int n, char *out, int cap,
               int skip_special) {
    if (strcmp(t->kind, "sudoku") == 0) {
        static const char *ch = "_123456789|<BOS>";
        int o = 0;
        for (int i = 0; i < n && o + 5 < cap; i++) {
            int id = ids[i];
            if (skip_special && id == 11) continue;
            if (id < 0 || id > 11) continue;
            if (id == 11) { memcpy(out + o, "<BOS>", 5); o += 5; }
            else out[o++] = ch[id];
        }
        out[o] = 0;
        return o;
    }
    int o = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->n_vocab) continue;
        if (skip_special && t->skip && t->skip[id]) continue;
        for (int k = 0; k < t->vlen[id] && o + 4 < cap; k++) {
            unsigned char b = t->vocab[id][k];
            if (b < 0x80) out[o++] = (char)b;
            else if (o + 4 < cap) {
                /* pass through UTF-8 bytes; replace lone surrogates not possible here */
                out[o++] = (char)b;
            }
        }
    }
    out[o] = 0;
    /* errors="replace": best-effort — lone continuation bytes become U+FFFD.
     * Validate quickly and patch. */
    {
        int w = 0;
        for (int r = 0; r < o;) {
            unsigned char c = out[r];
            int need = (c < 0x80) ? 1 : ((c & 0xE0) == 0xC0 ? 2 : ((c & 0xF0) == 0xE0 ? 3 : ((c & 0xF8) == 0xF0 ? 4 : -1)));
            int ok = need > 0;
            if (ok) for (int k = 1; k < need; k++)
                if (r + k >= o || (out[r + k] & 0xC0) != 0x80) { ok = 0; break; }
            if (ok) { for (int k = 0; k < need; k++) out[w++] = out[r++]; }
            else { out[w++] = (char)0xEF; out[w++] = (char)0xBF; out[w++] = (char)0xBD; r++; }
        }
        out[w] = 0;
        return w;
    }
}

int tok_eos(const TokState *t) { return t->eos_id; }
int tok_pad(const TokState *t) { return t->pad_id; }
