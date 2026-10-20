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
    int mapped;  /* 1 = GPT-2 bytes_to_unicode layer (gpt2-style HF BPE) */
};

/* ---------- GPT-2 bytes_to_unicode (fixed algorithmic mapping) ---------- */
static unsigned b2u_cp[256];
static unsigned char b2u_utf8[256][4];
static int b2u_utf8_len[256];
static int b2u_rev[512]; /* mapped code point -> original byte, -1 if none */
static int b2u_ready = 0;

static void b2u_init(void) {
    if (b2u_ready) return;
    int bs[256], cs[256], nb = 0;
    for (int b = 0; b < 256; b++) b2u_rev[b] = -1;
    /* printable ASCII + Latin-1 supplement that map to themselves */
    for (int b = '!'; b <= '~'; b++) { bs[nb] = b; cs[nb] = b; nb++; }
    for (int b = 0xA1; b <= 0xAC; b++) { bs[nb] = b; cs[nb] = b; nb++; }
    for (int b = 0xAE; b <= 0xFF; b++) { bs[nb] = b; cs[nb] = b; nb++; }
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int found = 0;
        for (int i = 0; i < nb; i++) if (bs[i] == b) { found = 1; break; }
        if (!found) { bs[nb] = b; cs[nb] = 256 + n; n++; nb++; }
    }
    for (int i = 0; i < 256; i++) {
        int b = bs[i], c = cs[i];
        b2u_cp[b] = (unsigned)c;
        b2u_rev[c] = b;
        /* UTF-8 encode c */
        if (c < 0x80) { b2u_utf8[b][0] = (unsigned char)c; b2u_utf8_len[b] = 1; }
        else if (c < 0x800) {
            b2u_utf8[b][0] = (unsigned char)(0xC0 | (c >> 6));
            b2u_utf8[b][1] = (unsigned char)(0x80 | (c & 0x3F));
            b2u_utf8_len[b] = 2;
        } else {
            b2u_utf8[b][0] = (unsigned char)(0xE0 | (c >> 12));
            b2u_utf8[b][1] = (unsigned char)(0x80 | ((c >> 6) & 0x3F));
            b2u_utf8[b][2] = (unsigned char)(0x80 | (c & 0x3F));
            b2u_utf8_len[b] = 3;
        }
    }
    b2u_ready = 1;
}

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
        char line[8192];
        t->n_vocab = 0;
        while (fgets(line, sizeof(line), f)) {
            int id = atoi(line);
            char *tab = strchr(line, '\t');
            if (!tab) continue;
            tab[strcspn(tab, "\r\n")] = 0;
            while (id >= cap) {
                cap *= 2;
                t->vocab = (unsigned char **)realloc(t->vocab, cap * sizeof(char *));
                t->vlen = (int *)realloc(t->vlen, cap * sizeof(int));
            }
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
        /* grow on demand below */
        while (fgets(line, sizeof(line), f)) {
            char *sp = strchr(line, ' ');
            if (!sp) continue;
            *sp = 0;
            line[strcspn(line, "\r\n")] = 0;
            /* note: ` *sp=0` then second token at sp+1; strip newline below */
            char *b = sp + 1;
            b[strcspn(b, "\r\n")] = 0;
            if (t->n_merges >= mcap) {
                mcap *= 2;
                t->ma = (unsigned char **)realloc(t->ma, mcap * sizeof(char *));
                t->mb = (unsigned char **)realloc(t->mb, mcap * sizeof(char *));
                t->mlen_a = (int *)realloc(t->mlen_a, mcap * sizeof(int));
                t->mlen_b = (int *)realloc(t->mlen_b, mcap * sizeof(int));
            }
            t->ma[t->n_merges] = unhex(line, &t->mlen_a[t->n_merges]);
            t->mb[t->n_merges] = unhex(b, &t->mlen_b[t->n_merges]);
            t->n_merges++;
        }
        fclose(f);
        if (strcmp(t->kind, "hf") == 0 && strstr(c->hf_model_id, "Qwen") != NULL)
            t->pattern = 1;
        if (strcmp(t->kind, "hf") == 0 && strstr(c->hf_model_id, "Qwen") == NULL
            && strstr(c->hf_model_id, "bert") == NULL && strstr(c->hf_model_id, "Bert") == NULL) {
            b2u_init();
            t->mapped = 1;
        }
        if (t->mapped) {
            /* translate mapped-UTF8 vocab/merges back to original bytes so the
             * byte-level BPE engine applies directly (1:1 byte correspondence) */
            for (int id = 0; id < t->n_vocab; id++) {
                unsigned char *o = t->vocab[id];
                int n = t->vlen[id];
                unsigned char *nb = (unsigned char *)malloc(n + 1);
                int w = 0, r = 0;
                while (r < n) {
                    unsigned char cc = o[r];
                    int cp, need;
                    if (cc < 0x80) { cp = cc; need = 1; }
                    else if ((cc & 0xE0) == 0xC0 && r + 1 < n) { cp = ((cc & 0x1F) << 6) | (o[r+1] & 0x3F); need = 2; }
                    else if ((cc & 0xF0) == 0xE0 && r + 2 < n) { cp = ((cc & 0x0F) << 12) | ((o[r+1] & 0x3F) << 6) | (o[r+2] & 0x3F); need = 3; }
                    else { cp = -1; need = 1; }
                    int b = (cp >= 0 && cp < 512) ? b2u_rev[cp] : -1;
                    if (b >= 0) nb[w++] = (unsigned char)b;
                    else { for (int k = 0; k < need && r + k < n; k++) nb[w++] = o[r + k]; }
                    r += need;
                }
                free(t->vocab[id]);
                t->vocab[id] = nb;
                t->vlen[id] = w;
            }
            for (int i = 0; i < t->n_merges; i++) {
                for (int k = 0; k < 2; k++) {
                    unsigned char *o = k ? t->mb[i] : t->ma[i];
                    int n = k ? t->mlen_b[i] : t->mlen_a[i];
                    unsigned char *nb = (unsigned char *)malloc(n + 1);
                    int w = 0, r = 0;
                    while (r < n) {
                        unsigned char cc = o[r];
                        int cp, need;
                        if (cc < 0x80) { cp = cc; need = 1; }
                        else if ((cc & 0xE0) == 0xC0 && r + 1 < n) { cp = ((cc & 0x1F) << 6) | (o[r+1] & 0x3F); need = 2; }
                        else if ((cc & 0xF0) == 0xE0 && r + 2 < n) { cp = ((cc & 0x0F) << 12) | ((o[r+1] & 0x3F) << 6) | (o[r+2] & 0x3F); need = 3; }
                        else { cp = -1; need = 1; }
                        int b = (cp >= 0 && cp < 512) ? b2u_rev[cp] : -1;
                        if (b >= 0) nb[w++] = (unsigned char)b;
                        else { for (int q = 0; q < need && r + q < n; q++) nb[w++] = o[r + q]; }
                        r += need;
                    }
                    free(o);
                    if (k) { t->mb[i] = nb; t->mlen_b[i] = w; }
                    else { t->ma[i] = nb; t->mlen_a[i] = w; }
                }
            }
            t->mapped = 0; /* translated to raw bytes; engine + decode run unmapped */
        }
        /* reverse hash map (sized for vocab) */
        t->hcap = 131072;
        while (t->hcap < t->n_vocab * 2) t->hcap *= 2;
        t->htable = (int *)malloc(t->hcap * sizeof(int));
        for (int i = 0; i < t->hcap; i++) t->htable[i] = -1;
        for (int id = 0; id < t->n_vocab; id++) {
            unsigned long long h = fnv(t->vocab[id], t->vlen[id]) & (t->hcap - 1);
            while (t->htable[h] != -1) h = (h + 1) & (t->hcap - 1);
            t->htable[h] = id;
        }
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
    if (t->mapped) {
        /* unmap: mapped UTF-8 -> code points -> original bytes */
        unsigned char *tmp = (unsigned char *)malloc(o + 1);
        int w2 = 0;
        for (int r = 0; r < o;) {
            unsigned char c = out[r];
            int cp, need;
            if (c < 0x80) { cp = c; need = 1; }
            else if ((c & 0xE0) == 0xC0 && r + 1 < o) { cp = ((c & 0x1F) << 6) | (out[r+1] & 0x3F); need = 2; }
            else if ((c & 0xF0) == 0xE0 && r + 2 < o) { cp = ((c & 0x0F) << 12) | ((out[r+1] & 0x3F) << 6) | (out[r+2] & 0x3F); need = 3; }
            else if ((c & 0xF8) == 0xF0 && r + 3 < o) { cp = ((c & 0x07) << 18) | ((out[r+1] & 0x3F) << 12) | ((out[r+2] & 0x3F) << 6) | (out[r+3] & 0x3F); need = 4; }
            else { cp = 0xFFFD; need = 1; }
            int b = (cp >= 0 && cp < 512) ? b2u_rev[cp] : -1;
            if (b >= 0) tmp[w2++] = (unsigned char)b;
            else {
                /* not a mapped char: copy raw bytes through */
                for (int k = 0; k < need && r + k < o; k++) tmp[w2++] = out[r + k];
            }
            r += need;
        }
        memcpy(out, tmp, w2);
        o = w2;
        out[o] = 0;
        free(tmp);
    }
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
