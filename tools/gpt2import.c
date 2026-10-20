/* tools/gpt2import.c — offline GPT-2 checkpoint converter (pure C, no torch).
 *
 * Reads HF model.safetensors + vocab.json + merges.txt, writes:
 *   <out>/model.ncm   (NCM1: f32 tensors, 2-D [in,out] transposed to [out][in])
 *   <out>/bpe_vocab.txt, <out>/bpe_merges.txt  (cncm hex format, raw bytes)
 *
 * Skips h.*.attn.bias (causal mask buffer; rebuilt at runtime).
 * Usage: gpt2import <model.safetensors> <vocab.json> <merges.txt> <outdir>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- minimal JSON scanner (safetensors header + vocab.json) ---------- */

typedef struct { const unsigned char *p, *end; } JS;

static void js_ws(JS *j) {
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r')) j->p++;
}

/* Parse JSON string into malloc'd UTF-8 buffer (handles escapes incl \u). */
static int js_string(JS *j, unsigned char **out, size_t *n) {
    js_ws(j);
    if (j->p >= j->end || *j->p != '"') return -1;
    j->p++;
    size_t cap = 64, len = 0;
    unsigned char *b = malloc(cap);
    while (j->p < j->end && *j->p != '"') {
        unsigned c;
        if (*j->p == '\\') {
            j->p++;
            if (j->p >= j->end) { free(b); return -1; }
            char e = (char)*j->p++;
            if (e == '"' || e == '\\' || e == '/') c = (unsigned)e;
            else if (e == 'b') c = '\b';
            else if (e == 'f') c = '\f';
            else if (e == 'n') c = '\n';
            else if (e == 'r') c = '\r';
            else if (e == 't') c = '\t';
            else if (e == 'u') {
                if (j->end - j->p < 4) { free(b); return -1; }
                unsigned v = 0;
                for (int k = 0; k < 4; k++) {
                    char h = (char)j->p[k];
                    v <<= 4;
                    if (h >= '0' && h <= '9') v |= (unsigned)(h - '0');
                    else if (h >= 'a' && h <= 'f') v |= (unsigned)(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') v |= (unsigned)(h - 'A' + 10);
                    else { free(b); return -1; }
                }
                j->p += 4;
                /* surrogate pair */
                if (v >= 0xD800 && v <= 0xDBFF && (j->end - j->p) >= 6 &&
                    j->p[0] == '\\' && j->p[1] == 'u') {
                    unsigned w = 0;
                    int ok = 1;
                    for (int k = 0; k < 4; k++) {
                        char h = (char)j->p[2 + k];
                        w <<= 4;
                        if (h >= '0' && h <= '9') w |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') w |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') w |= (unsigned)(h - 'A' + 10);
                        else ok = 0;
                    }
                    if (ok && w >= 0xDC00 && w <= 0xDFFF) {
                        v = 0x10000 + ((v - 0xD800) << 10) + (w - 0xDC00);
                        j->p += 6;
                    }
                }
                c = v;
            } else { free(b); return -1; }
        } else {
            c = *j->p++;
            /* pass through UTF-8 continuation bytes of literal multibyte seq */
            if (c >= 0x80) {
                int need = (c >= 0xF0) ? 3 : (c >= 0xE0) ? 2 : (c >= 0xC0) ? 1 : 0;
                /* store lead now, continuations below */
                if (len + 1 >= cap) { cap *= 2; b = realloc(b, cap); }
                b[len++] = (unsigned char)c;
                for (int k = 0; k < need; k++) {
                    if (j->p >= j->end) { free(b); return -1; }
                    if (len + 1 >= cap) { cap *= 2; b = realloc(b, cap); }
                    b[len++] = *j->p++;
                }
                continue;
            }
        }
        /* encode c as UTF-8 */
        unsigned char tmp[4];
        int tn = 0;
        if (c < 0x80) tmp[tn++] = (unsigned char)c;
        else if (c < 0x800) { tmp[tn++] = (unsigned char)(0xC0 | (c >> 6)); tmp[tn++] = (unsigned char)(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { tmp[tn++] = (unsigned char)(0xE0 | (c >> 12)); tmp[tn++] = (unsigned char)(0x80 | ((c >> 6) & 0x3F)); tmp[tn++] = (unsigned char)(0x80 | (c & 0x3F)); }
        else { tmp[tn++] = (unsigned char)(0xF0 | (c >> 18)); tmp[tn++] = (unsigned char)(0x80 | ((c >> 12) & 0x3F)); tmp[tn++] = (unsigned char)(0x80 | ((c >> 6) & 0x3F)); tmp[tn++] = (unsigned char)(0x80 | (c & 0x3F)); }
        while (len + (size_t)tn + 1 >= cap) { cap *= 2; b = realloc(b, cap); }
        memcpy(b + len, tmp, (size_t)tn);
        len += (size_t)tn;
    }
    if (j->p >= j->end) { free(b); return -1; }
    j->p++; /* closing quote */
    b[len] = 0;
    *out = b;
    *n = len;
    return 0;
}

static int js_int(JS *j, long long *v) {
    js_ws(j);
    int neg = 0;
    if (j->p < j->end && *j->p == '-') { neg = 1; j->p++; }
    if (j->p >= j->end || *j->p < '0' || *j->p > '9') return -1;
    long long x = 0;
    while (j->p < j->end && *j->p >= '0' && *j->p <= '9') x = x * 10 + (*j->p++ - '0');
    *v = neg ? -x : x;
    return 0;
}

/* Skip any JSON value. */
static int js_skip(JS *j) {
    js_ws(j);
    if (j->p >= j->end) return -1;
    if (*j->p == '"') {
        unsigned char *s;
        size_t n;
        int r = js_string(j, &s, &n);
        free(s);
        return r;
    }
    if (*j->p == '{') {
        j->p++;
        js_ws(j);
        if (j->p < j->end && *j->p == '}') { j->p++; return 0; }
        for (;;) {
            unsigned char *k;
            size_t kn;
            if (js_string(j, &k, &kn)) return -1;
            free(k);
            js_ws(j);
            if (j->p >= j->end || *j->p != ':') return -1;
            j->p++;
            if (js_skip(j)) return -1;
            js_ws(j);
            if (j->p >= j->end) return -1;
            if (*j->p == ',') { j->p++; continue; }
            if (*j->p == '}') { j->p++; return 0; }
            return -1;
        }
    }
    if (*j->p == '[') {
        j->p++;
        js_ws(j);
        if (j->p < j->end && *j->p == ']') { j->p++; return 0; }
        for (;;) {
            if (js_skip(j)) return -1;
            js_ws(j);
            if (j->p >= j->end) return -1;
            if (*j->p == ',') { j->p++; continue; }
            if (*j->p == ']') { j->p++; return 0; }
            return -1;
        }
    }
    /* number / literal */
    while (j->p < j->end && *j->p != ',' && *j->p != '}' && *j->p != ']' &&
           *j->p != ' ' && *j->p != '\t' && *j->p != '\n' && *j->p != '\r')
        j->p++;
    return 0;
}

/* ---------- GPT-2 bytes<->unicode maps ---------- */

static void b2u_init_maps(char umap[256][8], int ulen[256], int inv_cp[0x500], int inv_b[0x500]) {
    int bs[512], n = 0;
    for (int b = '!'; b <= '~'; b++) bs[n++] = b;
    for (int b = 0xA1; b <= 0xAC; b++) bs[n++] = b;
    for (int b = 0xAE; b <= 0xFF; b++) bs[n++] = b;
    int is_in[256] = {0};
    for (int i = 0; i < n; i++) is_in[bs[i]] = 1;
    int cs[512];
    for (int i = 0; i < n; i++) cs[i] = bs[i];
    int m = 256;
    for (int b = 0; b < 256; b++)
        if (!is_in[b]) {
            bs[n] = b;
            cs[n] = m++;
            n++;
        }
    for (int i = 0; i < 256; i++) {
        int cp = cs[i];
        if (cp < 0x80) { umap[i][0] = (char)cp; ulen[i] = 1; }
        else if (cp < 0x800) {
            umap[i][0] = (char)(0xC0 | (cp >> 6));
            umap[i][1] = (char)(0x80 | (cp & 0x3F));
            ulen[i] = 2;
        } else {
            umap[i][0] = (char)(0xE0 | (cp >> 12));
            umap[i][1] = (char)(0x80 | ((cp >> 6) & 0x3F));
            umap[i][2] = (char)(0x80 | (cp & 0x3F));
            ulen[i] = 3;
        }
    }
    for (int i = 0; i < 0x500; i++) { inv_cp[i] = -1; inv_b[i] = -1; }
    for (int i = 0; i < 256; i++) {
        /* inverse: mapped codepoint cs[i] came from byte bs[i] */
        inv_cp[cs[i]] = bs[i];
    }
}

/* Decode UTF-8 mapped string to raw bytes via inverse map. Returns count or -1. */
static int unmap_str(const unsigned char *s, size_t n, char umap[256][8], int ulen[256],
                     int inv_cp[0x500], unsigned char *out, size_t cap) {
    (void)umap;
    (void)ulen;
    size_t o = 0;
    size_t i = 0;
    while (i < n) {
        unsigned cp;
        unsigned char c = s[i];
        if (c < 0x80) { cp = c; i++; }
        else if ((c & 0xE0) == 0xC0 && i + 1 < n) { cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F); i += 2; }
        else if ((c & 0xF0) == 0xE0 && i + 2 < n) { cp = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F); i += 3; }
        else return -1;
        if (cp >= 0x500 || inv_cp[cp] < 0) return -1;
        if (o >= cap) return -1;
        out[o++] = (unsigned char)inv_cp[cp];
    }
    return (int)o;
}

/* ---------- .ncm writer (matches tensor.c weights_load) ---------- */

static void w32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void w64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }

static int ncm_add(FILE *f, const char *name, int ndim, const int64_t *shape, const float *data) {
    size_t nb = strlen(name);
    if (nb > 255) return -1;
    unsigned char nl = (unsigned char)nb, nd = (unsigned char)ndim;
    fwrite(&nl, 1, 1, f);
    fwrite(name, 1, nb, f);
    fwrite(&nd, 1, 1, f);
    int64_t n = 1;
    for (int d = 0; d < ndim; d++) { w32(f, (uint32_t)shape[d]); n *= shape[d]; }
    w64(f, (uint64_t)n * 4);
    fwrite(data, 4, (size_t)n, f);
    return 0;
}

static float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    uint32_t f;
    if (e == 0) f = (s << 31) | (m ? 0x33800000u + (m << 13) : 0);
    else if (e == 31) f = (s << 31) | 0x7F800000u | (m << 13);
    else f = (s << 31) | ((e + 112) << 23) | (m << 13);
    float r;
    memcpy(&r, &f, 4);
    return r;
}

static float bf16_to_f32(uint16_t h) {
    uint32_t f = (uint32_t)h << 16;
    float r;
    memcpy(&r, &f, 4);
    return r;
}

/* Transpose 2-D row-major [R][C] -> [C][R]. */
static void transpose2(const float *in, float *out, int64_t R, int64_t C) {
    for (int64_t r = 0; r < R; r++)
        for (int64_t c = 0; c < C; c++) out[c * R + r] = in[r * C + c];
}

static int endswith(const char *s, const char *suf) {
    size_t a = strlen(s), b = strlen(suf);
    return a >= b && !strcmp(s + a - b, suf);
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: gpt2import <model.safetensors> <vocab.json> <merges.txt> <outdir>\n");
        return 2;
    }
    const char *sf_path = argv[1], *vj_path = argv[2], *mg_path = argv[3], *outdir = argv[4];

    /* ---- safetensors header ---- */
    FILE *f = fopen(sf_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", sf_path); return 1; }
    uint64_t hlen = 0;
    if (fread(&hlen, 8, 1, f) != 1) { fprintf(stderr, "bad header\n"); return 1; }
    unsigned char *hbuf = malloc((size_t)hlen);
    if (fread(hbuf, 1, (size_t)hlen, f) != (size_t)hlen) { fprintf(stderr, "short header\n"); return 1; }
    long data_start = 8 + (long)hlen;

    char ncm_path[1024], vb_path[1024], mg_out[1024];
    snprintf(ncm_path, sizeof(ncm_path), "%s/model.ncm", outdir);
    snprintf(vb_path, sizeof(vb_path), "%s/bpe_vocab.txt", outdir);
    snprintf(mg_out, sizeof(mg_out), "%s/bpe_merges.txt", outdir);
    FILE *fo = fopen(ncm_path, "wb");
    if (!fo) { fprintf(stderr, "cannot write %s\n", ncm_path); return 1; }
    fwrite("NCM1", 1, 4, fo);
    long cnt_pos = ftell(fo);
    w32(fo, 0);
    uint32_t count = 0;

    JS j = {hbuf, hbuf + hlen};
    js_ws(&j);
    if (j.p >= j.end || *j.p != '{') { fprintf(stderr, "bad json\n"); return 1; }
    j.p++;
    for (;;) {
        js_ws(&j);
        if (j.p < j.end && *j.p == '}') { j.p++; break; }
        unsigned char *key = NULL;
        size_t klen = 0;
        if (js_string(&j, &key, &klen)) { fprintf(stderr, "bad key\n"); return 1; }
        js_ws(&j);
        if (j.p >= j.end || *j.p != ':') { fprintf(stderr, "bad colon\n"); return 1; }
        j.p++;
        if (!strcmp((char *)key, "__metadata__")) {
            if (js_skip(&j)) { fprintf(stderr, "bad metadata\n"); return 1; }
        } else {
            /* value: {"dtype":..,"shape":[..],"data_offsets":[a,b]} (order free) */
            char dtype[16] = {0};
            int64_t shape[8] = {0};
            int ndim = 0;
            long long off0 = -1, off1 = -1;
            js_ws(&j);
            if (j.p >= j.end || *j.p != '{') { fprintf(stderr, "bad tensor obj\n"); return 1; }
            j.p++;
            for (;;) {
                unsigned char *fk = NULL;
                size_t fklen = 0;
                js_ws(&j);
                if (j.p < j.end && *j.p == '}') { j.p++; break; }
                if (js_string(&j, &fk, &fklen)) { fprintf(stderr, "bad field\n"); return 1; }
                js_ws(&j);
                if (j.p >= j.end || *j.p != ':') { fprintf(stderr, "bad colon2\n"); return 1; }
                j.p++;
                if (!strcmp((char *)fk, "dtype")) {
                    unsigned char *dv = NULL;
                    size_t dvn = 0;
                    if (js_string(&j, &dv, &dvn)) { fprintf(stderr, "bad dtype\n"); return 1; }
                    snprintf(dtype, sizeof(dtype), "%s", dv);
                    free(dv);
                } else if (!strcmp((char *)fk, "shape")) {
                    js_ws(&j);
                    if (j.p >= j.end || *j.p != '[') { fprintf(stderr, "bad shape\n"); return 1; }
                    j.p++;
                    ndim = 0;
                    for (;;) {
                        js_ws(&j);
                        if (j.p < j.end && *j.p == ']') { j.p++; break; }
                        long long v = 0;
                        if (js_int(&j, &v)) { fprintf(stderr, "bad dim\n"); return 1; }
                        if (ndim < 8) shape[ndim++] = v;
                        js_ws(&j);
                        if (j.p < j.end && *j.p == ',') { j.p++; continue; }
                        if (j.p < j.end && *j.p == ']') { j.p++; break; }
                        fprintf(stderr, "bad shape sep\n");
                        return 1;
                    }
                } else if (!strcmp((char *)fk, "data_offsets")) {
                    js_ws(&j);
                    if (j.p >= j.end || *j.p != '[') { fprintf(stderr, "bad offsets\n"); return 1; }
                    j.p++;
                    if (js_int(&j, &off0)) return 1;
                    js_ws(&j);
                    if (j.p >= j.end || *j.p != ',') return 1;
                    j.p++;
                    if (js_int(&j, &off1)) return 1;
                    js_ws(&j);
                    if (j.p >= j.end || *j.p != ']') return 1;
                    j.p++;
                } else {
                    if (js_skip(&j)) return 1;
                }
                free(fk);
                js_ws(&j);
                if (j.p < j.end && *j.p == ',') { j.p++; continue; }
                if (j.p < j.end && *j.p == '}') { j.p++; break; }
                fprintf(stderr, "bad tensor sep\n");
                return 1;
            }
            /* decide keep/transpose */
            int skip = endswith((char *)key, ".attn.bias");
            if (!skip) {
                int is2d = (ndim == 2);
                int doT = is2d && (endswith((char *)key, ".weight") &&
                                   (strstr((char *)key, "c_attn") || strstr((char *)key, "c_proj") ||
                                    strstr((char *)key, "c_fc") || strstr((char *)key, "proj")));
                /* lm_head absent in GPT-2 (tied); wte/wpe/ln/bias kept as-is */
                int64_t n = 1;
                for (int d = 0; d < ndim; d++) n *= shape[d];
                int el = 0;
                if (!strcmp(dtype, "F32")) el = 4;
                else if (!strcmp(dtype, "F16")) el = 2;
                else if (!strcmp(dtype, "BF16")) el = 2;
                else { fprintf(stderr, "unsupported dtype %s for %s\n", dtype, key); return 1; }
                size_t nbytes = (size_t)n * (size_t)el;
                unsigned char *raw = malloc(nbytes);
                if (fseek(f, data_start + off0, SEEK_SET) != 0) { fprintf(stderr, "seek fail\n"); return 1; }
                if (fread(raw, 1, nbytes, f) != nbytes) { fprintf(stderr, "short tensor %s\n", key); return 1; }
                float *fd = malloc((size_t)n * 4);
                if (el == 4) memcpy(fd, raw, nbytes);
                else if (!strcmp(dtype, "F16")) {
                    uint16_t *u = (uint16_t *)raw;
                    for (int64_t i = 0; i < n; i++) fd[i] = f16_to_f32(u[i]);
                } else {
                    uint16_t *u = (uint16_t *)raw;
                    for (int64_t i = 0; i < n; i++) fd[i] = bf16_to_f32(u[i]);
                }
                free(raw);
                float *out = fd;
                int64_t osh[8];
                for (int d = 0; d < ndim; d++) osh[d] = shape[d];
                if (doT) {
                    float *t = malloc((size_t)n * 4);
                    transpose2(fd, t, shape[0], shape[1]);
                    free(fd);
                    out = t;
                    osh[0] = shape[1];
                    osh[1] = shape[0];
                }
                if (ncm_add(fo, (char *)key, ndim, osh, out)) { fprintf(stderr, "ncm write fail\n"); return 1; }
                free(out);
                count++;
            }
        }
        free(key);
        js_ws(&j);
        if (j.p < j.end && *j.p == ',') { j.p++; continue; }
        if (j.p < j.end && *j.p == '}') { j.p++; break; }
        fprintf(stderr, "bad top sep\n");
        return 1;
    }
    free(hbuf);
    fclose(f);
    long endpos = ftell(fo);
    fseek(fo, cnt_pos, SEEK_SET);
    w32(fo, count);
    fseek(fo, endpos, SEEK_SET);
    fclose(fo);
    printf("tensors: %u -> %s\n", count, ncm_path);

    /* ---- vocab.json + merges.txt -> cncm hex format ---- */
    char umap[256][8];
    int ulen[256], inv_cp[0x500], inv_b[0x500];
    b2u_init_maps(umap, ulen, inv_cp, inv_b);
    (void)inv_b;

    FILE *vf = fopen(vj_path, "rb");
    if (!vf) { fprintf(stderr, "cannot open %s\n", vj_path); return 1; }
    fseek(vf, 0, SEEK_END);
    long vlen = ftell(vf);
    fseek(vf, 0, SEEK_SET);
    unsigned char *vbuf = malloc((size_t)vlen);
    if (fread(vbuf, 1, (size_t)vlen, vf) != (size_t)vlen) { fprintf(stderr, "short vocab\n"); return 1; }
    fclose(vf);
    FILE *vo = fopen(vb_path, "w");
    if (!vo) { fprintf(stderr, "cannot write %s\n", vb_path); return 1; }
    JS v = {vbuf, vbuf + vlen};
    js_ws(&v);
    if (v.p >= v.end || *v.p != '{') { fprintf(stderr, "bad vocab json\n"); return 1; }
    v.p++;
    uint32_t nvoc = 0;
    unsigned char rbuf[4096];
    for (;;) {
        unsigned char *tok = NULL;
        size_t toklen = 0;
        js_ws(&v);
        if (v.p < v.end && *v.p == '}') { v.p++; break; }
        if (js_string(&v, &tok, &toklen)) { fprintf(stderr, "bad vocab key\n"); return 1; }
        js_ws(&v);
        if (v.p >= v.end || *v.p != ':') { fprintf(stderr, "bad vocab colon\n"); return 1; }
        v.p++;
        long long id = 0;
        if (js_int(&v, &id)) { fprintf(stderr, "bad vocab id\n"); return 1; }
        int rn = unmap_str(tok, toklen, umap, ulen, inv_cp, rbuf, sizeof(rbuf));
        if (rn < 0) { fprintf(stderr, "unmappable vocab token id %lld\n", id); return 1; }
        fprintf(vo, "%lld\t", id);
        for (int i = 0; i < rn; i++) fprintf(vo, "%02x", rbuf[i]);
        fprintf(vo, "\n");
        nvoc++;
        free(tok);
        js_ws(&v);
        if (v.p < v.end && *v.p == ',') { v.p++; continue; }
        if (v.p < v.end && *v.p == '}') { v.p++; break; }
        fprintf(stderr, "bad vocab sep\n");
        return 1;
    }
    free(vbuf);
    fclose(vo);
    printf("vocab entries: %u -> %s\n", nvoc, vb_path);

    FILE *mf = fopen(mg_path, "r");
    if (!mf) { fprintf(stderr, "cannot open %s\n", mg_path); return 1; }
    FILE *mo = fopen(mg_out, "w");
    if (!mo) { fprintf(stderr, "cannot write %s\n", mg_out); return 1; }
    char line[8192];
    uint32_t nmg = 0;
    while (fgets(line, sizeof(line), mf)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0] || line[0] == '#') continue;
        char *sp = strchr(line, ' ');
        if (!sp) { fprintf(stderr, "bad merge line\n"); return 1; }
        *sp = 0;
        /* A/B are UTF-8 mapped strings; decode+unmap each */
        const unsigned char *parts[2] = {(unsigned char *)line, (unsigned char *)(sp + 1)};
        int first = 1;
        for (int s = 0; s < 2; s++) {
            const unsigned char *u = parts[s];
            size_t un = strlen((const char *)u);
            /* decode UTF-8 to codepoints then inverse-map */
            size_t i = 0;
            unsigned char rb[4096];
            int rn = 0;
            while (i < un) {
                unsigned cp;
                unsigned char c = u[i];
                if (c < 0x80) { cp = c; i++; }
                else if ((c & 0xE0) == 0xC0 && i + 1 < un) { cp = ((c & 0x1F) << 6) | (u[i + 1] & 0x3F); i += 2; }
                else if ((c & 0xF0) == 0xE0 && i + 2 < un) { cp = ((c & 0x0F) << 12) | ((u[i + 1] & 0x3F) << 6) | (u[i + 2] & 0x3F); i += 3; }
                else { fprintf(stderr, "bad utf8 in merges\n"); return 1; }
                if (cp >= 0x500 || inv_cp[cp] < 0) { fprintf(stderr, "unmappable merge cp U+%04X\n", cp); return 1; }
                rb[rn++] = (unsigned char)inv_cp[cp];
            }
            if (!first) fputc(' ', mo);
            first = 0;
            for (int i = 0; i < rn; i++) fprintf(mo, "%02x", rb[i]);
        }
        fputc('\n', mo);
        nmg++;
    }
    fclose(mf);
    fclose(mo);
    printf("merges: %u -> %s\n", nmg, mg_out);
    return 0;
}
