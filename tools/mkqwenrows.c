/* tools/mkqwenrows.c — pack Qwen LM training rows (.nctrain).
 *
 * Two input modes (one document per line):
 *   text mode (default): raw text lines, tokenized with the Qwen tokenizer
 *     from --conf/--asset-dir. Documents are joined with the EOS id.
 *   --ids mode: each line is a CSV of token ids (e.g. another tokenizer's
 *     output reused as a token stream); ids must fit the Qwen vocab.
 *     Distribution learning is tokenizer-agnostic: the model learns
 *     P(next|context) over whatever stream it is fed, so recycled id
 *     streams are a legitimate functional corpus as long as every id is
 *     a valid Qwen vocab entry.
 *
 * Output rows: (--tokens) ints each = T+1 ids; input = first T, target =
 * last T. Layout matches train_data_load ("NCMT", u64 nr, u64 rl, int32).
 *
 * Usage: mkqwenrows --conf M.conf --asset-dir D --in docs.txt
 *       --out rows.nctrain --tokens T+1 [--ids] [--max-rows N]
 *       [--strip ID] [--min-unique N]
 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/types.h>

#include "ncm.h"

static const char *optval(int argc, char **argv, const char *flag, const char *dflt) {
    for (int i = 1; i < argc - 1; i++)
        if (!strcmp(argv[i], flag)) return argv[i + 1];
    return dflt;
}

static int hasflag(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], flag)) return 1;
    return 0;
}

/* parse a CSV id line into out (cap); returns count or -1 on garbage */
static int parse_ids_line(const char *s, int *out, int cap) {
    int n = 0;
    const char *p = s;
    while (*p && n < cap) {
        while (*p == ' ' || *p == '\t' || *p == ',') p++;
        if (!*p || *p == '\n' || *p == '\r') break;
        char *e = NULL;
        long v = strtol(p, &e, 10);
        if (e == p) return -1;
        out[n++] = (int)v;
        p = e;
    }
    return n;
}

int main(int argc, char **argv) {
    const char *conf = optval(argc, argv, "--conf", NULL);
    const char *adir = optval(argc, argv, "--asset-dir", NULL);
    const char *inp = optval(argc, argv, "--in", NULL);
    const char *outp = optval(argc, argv, "--out", NULL);
    int per = atoi(optval(argc, argv, "--tokens", "129"));
    long max_rows = atol(optval(argc, argv, "--max-rows", "0"));
    int idmode = hasflag(argc, argv, "--ids");
    const char *strip_s = optval(argc, argv, "--strip", NULL);
    int strip = strip_s ? atoi(strip_s) : -1;
    int minuniq = atoi(optval(argc, argv, "--min-unique", "0"));
    if (!conf || !adir || !inp || !outp || per < 2) {
        fprintf(stderr,
                "usage: mkqwenrows --conf M.conf --asset-dir D --in docs.txt\n"
                "       --out rows.nctrain --tokens T+1 [--ids] [--max-rows N]\n"
                "       [--strip ID] [--min-unique N]\n");
        return 2;
    }
    NCMConfig c;
    if (config_load(&c, conf)) return 1;
    TokState *tok = tok_load(&c, adir);
    if (!tok) { fprintf(stderr, "tok load failed\n"); return 1; }
    int eos = tok_eos(tok);

    FILE *f = fopen(inp, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", inp); return 1; }
    size_t cap = 1 << 20, n = 0;
    int *ids = (int *)malloc(cap * sizeof(int));
    char *line = NULL;
    size_t lcap = 0;
    ssize_t len;
    long ndoc = 0;
    int *tmp = (int *)malloc(65536 * sizeof(int));
    while ((len = getline(&line, &lcap, f)) >= 0) {
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
        if (!len) continue;
        int got = 0;
        if (idmode) {
            got = parse_ids_line(line, tmp, 65536);
            if (got < 0) { fprintf(stderr, "bad ids line %ld\n", ndoc + 1); return 1; }
        } else {
            got = tok_encode(tok, line, tmp, 65536);
            if (got < 0) { fprintf(stderr, "tokenize failed on doc %ld\n", ndoc + 1); return 1; }
        }
        /* strip pad ids everywhere (fixed-length pickles pad interior
         * blocks too, not just tails); drop docs left empty */
        if (strip >= 0) {
            int w = 0;
            for (int i = 0; i < got; i++)
                if (tmp[i] != strip) tmp[w++] = tmp[i];
            got = w;
        }
        if (got == 0) continue;
        if (minuniq > 0) {
            /* exact distinct-token count via insertion sort (docs are short) */
            int *s2 = (int *)malloc(got * sizeof(int));
            memcpy(s2, tmp, got * sizeof(int));
            for (int i = 1; i < got; i++) {
                int k = s2[i], j = i - 1;
                while (j >= 0 && s2[j] > k) { s2[j + 1] = s2[j]; j--; }
                s2[j + 1] = k;
            }
            int distinct = 1;
            for (int i = 1; i < got; i++) if (s2[i] != s2[i - 1]) distinct++;
            free(s2);
            if (distinct < minuniq) continue;
        }
        if (n + (size_t)got + 1 > cap) {
            while (n + (size_t)got + 1 > cap) cap *= 2;
            ids = (int *)realloc(ids, cap * sizeof(int));
        }
        memcpy(ids + n, tmp, (size_t)got * sizeof(int));
        n += got;
        ids[n++] = eos; /* document boundary */
        ndoc++;
    }
    free(tmp);
    free(line);
    fclose(f);
    /* Qwen vocab range check (tok ids are guaranteed; --ids mode needs it) */
    for (size_t i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= 151936) {
            fprintf(stderr, "id %d out of Qwen vocab range at position %zu\n", ids[i], i);
            return 1;
        }
    }
    long nrows = (long)(n / (size_t)per);
    if (max_rows > 0 && nrows > max_rows) nrows = max_rows;
    FILE *o = fopen(outp, "wb");
    if (!o) { fprintf(stderr, "cannot write %s\n", outp); return 1; }
    fwrite("NCMT", 1, 4, o);
    uint64_t nr = (uint64_t)nrows, rl = (uint64_t)per;
    fwrite(&nr, 8, 1, o);
    fwrite(&rl, 8, 1, o);
    fwrite(ids, sizeof(int), (size_t)nrows * (size_t)per, o);
    fclose(o);
    printf("mkqwenrows: %ld docs %zu ids -> %ld rows of %d to %s\n",
           ndoc, n, nrows, per, outp);
    free(ids);
    tok_free(tok);
    return 0;
}
