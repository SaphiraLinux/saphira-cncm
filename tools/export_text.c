/* tools/export_text.c — bible (or any ref-tagged) corpus -> token-id .nctrain.
 *
 * Input:  verse lines  TRANS|Book|Chapter|Verse<TAB>text
 *         (verse text may itself contain newlines: a new verse starts only
 *         at a line matching the ref pattern; other lines are continuations).
 * Output: .nctrain rows [P prefix ids | T content token ids], fixed layout:
 *           row[0] = translation index (corpus order of first appearance)
 *           row[1] = canonical book index (0..65, fixed table below)
 *           row[2] = chapter number of the row's first verse
 *           row[3..] = BPE token ids, packed verse after verse
 *         Short final rows are padded with the configured pad id
 *         (tok.pad_id, else tok.eos_id); padding is recorded, never hidden.
 * Sidecar: JSON manifest mapping every row to its verse spans, so the
 * original text and all verse boundaries reconstruct exactly (see
 * tests/roundtrip_text.py).
 *
 * Usage: export_text --corpus bible_corpus.txt --conf M.conf
 *          --asset-dir D --out bible.nctrain --manifest bible_manifest.json
 *          [--prefix P] [--tokens T] [--translations CSV] [--max-rows N]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "ncm.h"

static const char *kBooks[66] = {
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
    "Ezra", "Nehemiah", "Esther", "Job", "Psalm",
    "Proverbs", "Ecclesiastes", "Song Of Solomon", "Isaiah", "Jeremiah",
    "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel",
    "Amos", "Obadiah", "Jonah", "Micah", "Nahum",
    "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi",
    "Matthew", "Mark", "Luke", "John", "Acts",
    "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews",
    "James", "1 Peter", "2 Peter", "1 John", "2 John",
    "3 John", "Jude", "Revelation"
};

static int book_index(const char *name) {
    for (int i = 0; i < 66; i++)
        if (!strcmp(name, kBooks[i])) return i;
    return -1;
}

/* Parse "TRANS|Book|Chapter|Verse". Returns 0 on success. ch/v are ints. */
static int parse_ref(const char *s, size_t n, char *trans, size_t tcap,
                     char *book, size_t bcap, int *ch, int *v) {
    const char *p1 = memchr(s, '|', n);
    if (!p1) return -1;
    const char *p2 = memchr(p1 + 1, '|', n - (size_t)(p1 + 1 - s));
    if (!p2) return -1;
    const char *p3 = memchr(p2 + 1, '|', n - (size_t)(p2 + 1 - s));
    if (!p3) return -1;
    size_t tl = (size_t)(p1 - s), bl = (size_t)(p2 - (p1 + 1));
    if (tl == 0 || tl >= tcap || bl == 0 || bl >= bcap) return -1;
    memcpy(trans, s, tl);
    trans[tl] = 0;
    memcpy(book, p1 + 1, bl);
    book[bl] = 0;
    /* chapter and verse must be all digits */
    const char *cs = p2 + 1, *ce = p3, *vs = p3 + 1, *ve = s + n;
    if (ce <= cs || ve <= vs) return -1;
    int c = 0, vv = 0;
    for (const char *p = cs; p < ce; p++) {
        if (*p < '0' || *p > '9') return -1;
        c = c * 10 + (*p - '0');
    }
    for (const char *p = vs; p < ve; p++) {
        if (*p < '0' || *p > '9') return -1;
        vv = vv * 10 + (*p - '0');
    }
    *ch = c;
    *v = vv;
    return 0;
}

static const char *optval(int argc, char **argv, const char *flag, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++)
        if (!strcmp(argv[i], flag)) return argv[i + 1];
    return dflt;
}

static int optset(int argc, char **argv, const char *flag, char out[64][256]) {
    const char *v = optval(argc, argv, flag, NULL);
    if (!v) return 0;
    int n = 0;
    const char *p = v;
    while (*p && n < 64) {
        const char *c = strchr(p, ',');
        size_t l = c ? (size_t)(c - p) : strlen(p);
        if (l > 255) l = 255;
        memcpy(out[n], p, l);
        out[n][l] = 0;
        n++;
        if (!c) break;
        p = c + 1;
    }
    return n;
}

static int want_trans(char wants[64][256], int nwant, const char *tr) {
    if (!nwant) return 1;
    for (int i = 0; i < nwant; i++)
        if (!strcmp(wants[i], tr)) return 1;
    return 0;
}

int main(int argc, char **argv) {
    const char *corpus = optval(argc, argv, "--corpus", NULL);
    const char *conf = optval(argc, argv, "--conf", NULL);
    const char *adir = optval(argc, argv, "--asset-dir", NULL);
    const char *outp = optval(argc, argv, "--out", NULL);
    const char *manp = optval(argc, argv, "--manifest", NULL);
    int P = atoi(optval(argc, argv, "--prefix", "3"));
    int T = atoi(optval(argc, argv, "--tokens", "128"));
    long max_rows = atol(optval(argc, argv, "--max-rows", "0"));
    char wants[64][256];
    int nwant = optset(argc, argv, "--translations", wants);
    if (!corpus || !conf || !adir || !outp || !manp || P != 3 || T <= 0) {
        fprintf(stderr,
                "usage: export_text --corpus FILE --conf M.conf --asset-dir D\n"
                "       --out rows.nctrain --manifest rows.json [--prefix 3]\n"
                "       [--tokens 128] [--translations CSV] [--max-rows N]\n"
                "  prefix is fixed [trans_idx, book_idx, chapter]; short final\n"
                "  rows are padded with tok.pad_id (else tok.eos_id), recorded.\n");
        return 2;
    }
    NCMConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (config_load(&cfg, conf)) return 1;
    TokState *tok = tok_load(&cfg, adir);
    int pad = cfg.pad_token_id >= 0 ? cfg.pad_token_id : cfg.eos_token_id;
    if (pad < 0) { fprintf(stderr, "need tok.pad_id or tok.eos_id\n"); return 1; }

    FILE *f = fopen(corpus, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", corpus); return 1; }
    FILE *fo = fopen(outp, "wb");
    if (!fo) { fprintf(stderr, "cannot write %s\n", outp); return 1; }
    FILE *mo = fopen(manp, "w");
    if (!mo) { fprintf(stderr, "cannot write %s\n", manp); return 1; }
    fwrite("NCMT", 1, 4, fo);
    long cnt_pos = ftell(fo);
    uint64_t zero = 0, rowlen = (uint64_t)(P + T);
    fwrite(&zero, 8, 1, fo);
    fwrite(&rowlen, 8, 1, fo);

    /* translation order of first appearance */
    char trnames[64][256];
    int ntr = 0;
    if (P + T > 8192) { fprintf(stderr, "prefix+tokens too large\n"); return 1; }
    int row[8192];
    int span_b[8192], span_c[8192], span_v[8192], span_s[8192], span_e[8192];
    int nspan = 0;
    int cur_n = 0; /* content tokens buffered in row[P..] */
    int cur_tr = -1, cur_book = -1, cur_ch = 0; /* prefix of open row */
    char cur_bname[256] = {0};
    char cur_tname[256] = {0};
    long nrows = 0;
    int *ids = malloc(65536 * sizeof(int));

    /* manifest prologue (rows appended below; closed at end) */
    fprintf(mo, "{\n\"corpus\": \"%s\",\n", corpus);
    fprintf(mo, "\"tokenizer\": \"qwen\",\n");
    fprintf(mo, "\"pad_id\": %d, \"eos_id\": %d,\n", pad, cfg.eos_token_id);
    fprintf(mo, "\"prefix\": %d, \"tokens\": %d,\n", P, T);
    fprintf(mo, "\"rows\": [");

    int first_row = 1;

    /* verse assembly: ref line starts a verse, other lines continue it */
    char *line = NULL;
    size_t lcap = 0;
    ssize_t ll;
    char vtext[1 << 20];
    size_t vlen = 0;
    char vtr[256] = {0}, vbook[256] = {0};
    int vch = 0, vv = 0, have_verse = 0;

    long total_verses = 0;
#define SPAN_OPEN(_bi, _ch, _vv, _s) do { \
        if (nspan >= 8192) { fprintf(stderr, "too many spans\n"); return 1; } \
        span_b[nspan] = (_bi); span_c[nspan] = (_ch); span_v[nspan] = (_vv); span_s[nspan] = (_s); \
    } while (0)
#define SPAN_CLOSE(_e) do { span_e[nspan] = (_e); nspan++; } while (0)
#define SET_PREFIX(_ti, _bi, _ch, _bn) do { \
        cur_tr = (_ti); cur_book = (_bi); cur_ch = (_ch); \
        snprintf(cur_bname, sizeof(cur_bname), "%s", (_bn)); \
        snprintf(cur_tname, sizeof(cur_tname), "%s", trnames[_ti]); \
    } while (0)
#define EMIT_ROW(_npad) do { \
        row[0] = cur_tr; row[1] = cur_book; row[2] = cur_ch; \
        fwrite(row, sizeof(int), (size_t)(P + T), fo); \
        fprintf(mo, "%s{\"r\": %ld, \"trans\": \"%s\", \"book\": \"%s\", \"chapter\": %d, \"spans\": [", \
                first_row ? "" : ",", nrows, cur_tname, cur_bname, cur_ch); \
        first_row = 0; \
        for (int si = 0; si < nspan; si++) \
            fprintf(mo, "%s{\"b\": %d, \"c\": %d, \"v\": %d, \"s\": %d, \"e\": %d}", si ? "," : "", \
                    span_b[si], span_c[si], span_v[si], span_s[si], span_e[si]); \
        fprintf(mo, "], \"padded\": %d}", (_npad)); \
        nrows++; \
        cur_n = 0; nspan = 0; \
    } while (0)
#define FLUSH_VERSE() do { \
        if (!have_verse) break; \
        if (!want_trans(wants, nwant, vtr)) break; \
        int ti = -1; \
        for (int i = 0; i < ntr; i++) if (!strcmp(trnames[i], vtr)) { ti = i; break; } \
        if (ti < 0) { \
            if (ntr >= 64) { fprintf(stderr, "too many translations\n"); return 1; } \
            snprintf(trnames[ntr], sizeof(trnames[ntr]), "%s", vtr); \
            ti = ntr++; \
        } \
        int bi = book_index(vbook); \
        if (bi < 0) { fprintf(stderr, "unknown book: %s\n", vbook); return 1; } \
        total_verses++; \
        int n = tok_encode(tok, vtext, ids, 65536); \
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; } \
        /* one translation per row: a new translation always starts a fresh row */ \
        if (nspan > 0 && ti != cur_tr) { \
            int npad0 = T - cur_n; \
            for (int i = 0; i < npad0; i++) row[P + cur_n + i] = pad; \
            EMIT_ROW(npad0); \
            if (max_rows > 0 && nrows >= max_rows) goto done; \
        } \
        int off = 0, seg_open = 0; \
        if (n == 0) { \
            /* zero-length span pins even empty verses to the current row */ \
            if (nspan == 0 && cur_n == 0) { SET_PREFIX(ti, bi, vch, vbook); } \
            SPAN_OPEN(bi, vch, vv, cur_n); SPAN_CLOSE(cur_n); \
        } else while (off < n) { \
            if (!seg_open) { \
                if (nspan == 0 && cur_n == 0) { SET_PREFIX(ti, bi, vch, vbook); } \
                SPAN_OPEN(bi, vch, vv, cur_n); seg_open = 1; \
            } \
            int room = T - cur_n; \
            int take = n - off < room ? n - off : room; \
            memcpy(row + P + cur_n, ids + off, (size_t)take * sizeof(int)); \
            cur_n += take; off += take; \
            if (cur_n == T) { SPAN_CLOSE(cur_n); seg_open = 0; EMIT_ROW(0); \
                if (max_rows > 0 && nrows >= max_rows) goto done; } \
        } \
        if (seg_open) { SPAN_CLOSE(cur_n); } \
    } while (0)

    while ((ll = getline(&line, &lcap, f)) >= 0) {
        while (ll > 0 && (line[ll - 1] == '\n' || line[ll - 1] == '\r')) line[--ll] = 0;
        char *tab = memchr(line, '\t', (size_t)ll);
        int is_ref = 0;
        char rtr[256], rbook[256];
        int rch = 0, rv = 0;
        if (tab) {
            size_t rlen = (size_t)(tab - line);
            if (!parse_ref(line, rlen, rtr, sizeof(rtr), rbook, sizeof(rbook), &rch, &rv))
                is_ref = 1;
        }
        if (is_ref) {
            FLUSH_VERSE();
            snprintf(vtr, sizeof(vtr), "%s", rtr);
            snprintf(vbook, sizeof(vbook), "%s", rbook);
            vch = rch;
            vv = rv;
            size_t tl = strlen(tab + 1);
            if (tl >= sizeof(vtext)) { fprintf(stderr, "verse too long\n"); return 1; }
            memcpy(vtext, tab + 1, tl + 1);
            vlen = tl;
            have_verse = 1;
        } else {
            size_t al = strlen(line);
            if (vlen + 1 + al >= sizeof(vtext)) { fprintf(stderr, "verse too long\n"); return 1; }
            vtext[vlen++] = '\n';
            memcpy(vtext + vlen, line, al + 1);
            vlen += al;
        }
    }
    FLUSH_VERSE();
done:;
    /* pad + emit final partial row (if any content or spans) */
    if ((cur_n > 0 || nspan > 0) && !(max_rows > 0 && nrows >= max_rows)) {
        int npad = T - cur_n;
        for (int i = 0; i < npad; i++) row[P + cur_n + i] = pad;
        EMIT_ROW(npad);
    }
    free(ids);
    free(line);
    fclose(f);
    long endpos = ftell(fo);
    fseek(fo, cnt_pos, SEEK_SET);
    uint64_t nr = (uint64_t)nrows;
    fwrite(&nr, 8, 1, fo);
    fseek(fo, endpos, SEEK_SET);
    fclose(fo);
    fprintf(mo, "],\n\"translations\": [");
    for (int i = 0; i < ntr; i++) fprintf(mo, "%s\"%s\"", i ? ", " : "", trnames[i]);
    fprintf(mo, "],\n\"books\": [");
    for (int i = 0; i < 66; i++) fprintf(mo, "%s\"%s\"", i ? ", " : "", kBooks[i]);
    fprintf(mo, "],\n\"n_rows\": %ld, \"n_verses\": %ld}\n", nrows, total_verses);
    fclose(mo);
    printf("rows=%ld verses=%ld translations=%d -> %s + %s\n", nrows, total_verses, ntr, outp, manp);
    tok_free(tok);
#undef FLUSH_VERSE
#undef SPAN_OPEN
#undef SPAN_CLOSE
#undef SET_PREFIX
#undef EMIT_ROW
    return 0;
}
