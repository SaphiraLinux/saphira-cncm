/* ckpt.c — versioned training checkpoints, format "NCK1" (v1).
 *
 * Contents: model weights + architecture (name/shape per tensor), AdamW
 * moments + step counter + hyperparameters, global step, batch-RNG state,
 * scheduler + training config, dataset fingerprint, tokenizer identity.
 *
 * Deliberately ABSENT: wall-clock, pid, hostname, argv. Two runs with an
 * identical trajectory produce byte-identical files, which is what makes
 * the exact-resume proof a plain file comparison.
 *
 * All integers little-endian (x86-64/ARM-LE hosts; format v1 is LE-only
 * and says so). Floats stored as IEEE-754 binary32/binary64 LE.
 *
 * Save path: write tmp.<pid> -> validate by re-read (magic+checksum) ->
 * fsync -> rename(2) over target -> fsync directory. The previous file is
 * never modified until the replacement has validated.
 *
 * Load path: read whole file -> verify checksum FIRST -> parse with
 * strict bounds checks into an image -> run every compatibility check ->
 * only then apply to the live WeightStore/AdamW. Any failure leaves the
 * caller's state untouched and reports a reason in err.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* O_DIRECTORY, fsync dir durability */
#endif
#include "ncm.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define NCK1_MAGIC "NCK1"
#define NCK1_VERSION 1u
#define NCK1_MAX_FILE (1ULL << 40) /* 1 TiB sanity cap */
#define NCK1_MAX_STR (1u << 20)    /* 1 MiB cap per embedded string */

/* ---------------- FNV-1a 64 ---------------- */

static uint64_t fnv1a(const void *p, size_t n, uint64_t h) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

uint64_t ckpt_file_hash(const char *path, uint64_t *nbytes_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { if (nbytes_out) *nbytes_out = 0; return 0; }
    uint64_t h = 14695981039346656037ULL, n = 0;
    unsigned char buf[1 << 20];
    size_t r;
    while ((r = fread(buf, 1, sizeof(buf), f)) > 0) {
        h = fnv1a(buf, r, h);
        n += r;
    }
    fclose(f);
    if (nbytes_out) *nbytes_out = n;
    return h;
}

/* ---------------- probe ---------------- */

int ckpt_probe(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4];
    size_t r = fread(magic, 1, 4, f);
    fclose(f);
    if (r != 4) return -1;
    if (memcmp(magic, NCK1_MAGIC, 4) == 0) return 1;
    if (memcmp(magic, "NCM1", 4) == 0) return 0;
    return -1;
}

/* ---------------- writer ---------------- */

typedef struct { FILE *f; uint64_t h; int err; } CW;

static void wraw(CW *w, const void *p, size_t n) {
    if (w->err || n == 0) return;
    if (fwrite(p, 1, n, w->f) != n) { w->err = 1; return; }
    w->h = fnv1a(p, n, w->h);
}
static void wu8(CW *w, unsigned v) { unsigned char b = (unsigned char)v; wraw(w, &b, 1); }
static void wu16(CW *w, unsigned v) {
    unsigned char b[2] = {(unsigned char)v, (unsigned char)(v >> 8)};
    wraw(w, b, 2);
}
static void wu32(CW *w, uint32_t v) {
    unsigned char b[4];
    for (int i = 0; i < 4; i++) b[i] = (unsigned char)(v >> (8 * i));
    wraw(w, b, 4);
}
static void wu64(CW *w, uint64_t v) {
    unsigned char b[8];
    for (int i = 0; i < 8; i++) b[i] = (unsigned char)(v >> (8 * i));
    wraw(w, b, 8);
}
static void wi64(CW *w, int64_t v) { wu64(w, (uint64_t)v); }
static void wf64(CW *w, double v) { uint64_t u; memcpy(&u, &v, 8); wu64(w, u); }
static void wi32(CW *w, int32_t v) { wu32(w, (uint32_t)v); }
static void wstr(CW *w, const char *s) {
    size_t n = s ? strlen(s) : 0;
    if (n > NCK1_MAX_STR) { w->err = 1; return; }
    wu16(w, (unsigned)n);
    wraw(w, s, n);
}

/* ---------------- reader (cursor over validated buffer) ---------------- */

typedef struct { const unsigned char *p, *end; int err; } CR;

static void rraw(CR *r, void *out, size_t n) {
    if (r->err) return;
    if ((size_t)(r->end - r->p) < n) { r->err = 1; return; }
    memcpy(out, r->p, n);
    r->p += n;
}
static uint8_t ru8(CR *r) { uint8_t v = 0; rraw(r, &v, 1); return v; }
static uint16_t ru16(CR *r) {
    unsigned char b[2] = {0, 0}; rraw(r, b, 2);
    return (uint16_t)(b[0] | ((uint16_t)b[1] << 8));
}
static uint32_t ru32(CR *r) {
    unsigned char b[4] = {0, 0, 0, 0}; rraw(r, b, 4);
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}
static uint64_t ru64(CR *r) {
    unsigned char b[8] = {0, 0, 0, 0, 0, 0, 0, 0}; rraw(r, b, 8);
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | b[i];
    return v;
}
static int64_t ri64(CR *r) { return (int64_t)ru64(r); }
static int32_t ri32(CR *r) { return (int32_t)ru32(r); }
static double rf64(CR *r) { uint64_t u = ru64(r); double v; memcpy(&v, &u, 8); return v; }
/* returned pointer aliases the buffer; sets *len_out */
static const char *rstr(CR *r, size_t *len_out) {
    unsigned n = ru16(r);
    if (r->err) return NULL;
    if (n > NCK1_MAX_STR || (size_t)(r->end - r->p) < n) { r->err = 1; return NULL; }
    const char *s = (const char *)r->p;
    r->p += n;
    if (len_out) *len_out = n;
    return s;
}

/* read whole file into *buf_out (caller frees). 0 ok. */
static int read_all(const char *path, unsigned char **buf_out, size_t *len_out,
                    char *err, size_t errcap) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(err, errcap, "cannot open %s: %s", path, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(fileno(f), &st)) {
        snprintf(err, errcap, "cannot stat %s", path);
        fclose(f);
        return -1;
    }
    if (st.st_size < 8 || (uint64_t)st.st_size > NCK1_MAX_FILE) {
        snprintf(err, errcap, "%s: implausible size %lld", path, (long long)st.st_size);
        fclose(f);
        return -1;
    }
    size_t len = (size_t)st.st_size;
    unsigned char *buf = (unsigned char *)malloc(len ? len : 1);
    if (!buf) { snprintf(err, errcap, "out of memory"); fclose(f); return -1; }
    if (fread(buf, 1, len, f) != len) {
        snprintf(err, errcap, "%s: short read", path);
        free(buf); fclose(f);
        return -1;
    }
    fclose(f);
    *buf_out = buf;
    *len_out = len;
    return 0;
}

/* ---------------- save ---------------- */

int ckpt_save_atomic(const WeightStore *ws, AdamW *opt, const CkptRunState *rs,
                     const char *path, char *err, size_t errcap) {
    if (err && errcap) err[0] = 0;
    if (rs->accum_pending != 0) {
        snprintf(err, errcap, "refusing save with %d pending accumulation steps",
                 rs->accum_pending);
        return -1;
    }
    /* Device moments must be current on host before writing. */
    adamw_sync_to_host(opt);

    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, (int)getpid());
    CW w;
    w.f = fopen(tmp, "wb");
    w.h = 14695981039346656037ULL;
    w.err = 0;
    if (!w.f) {
        snprintf(err, errcap, "cannot write %s: %s", tmp, strerror(errno));
        return -1;
    }
    wraw(&w, NCK1_MAGIC, 4);
    wu32(&w, NCK1_VERSION);
    wi64(&w, (int64_t)rs->global_step);
    wu64(&w, rs->batch_rng);
    wf64(&w, rs->base_lr);
    wf64(&w, rs->warmup);
    wf64(&w, rs->total);
    wi32(&w, (int32_t)rs->batch_B);
    wi32(&w, (int32_t)rs->accum_pending);
    wi64(&w, (int64_t)rs->n_rows);
    wi32(&w, (int32_t)rs->row_len);
    wu64(&w, rs->data_hash);
    wstr(&w, rs->data_path ? rs->data_path : "");
    wu64(&w, rs->conf_hash);
    wstr(&w, rs->tokenizer_id ? rs->tokenizer_id : "");
    wstr(&w, rs->train_cfg_json ? rs->train_cfg_json : "");
    wu32(&w, (uint32_t)ws->count);
    for (int i = 0; i < ws->count && !w.err; i++) {
        Tensor *t = ws->items[i].t;
        size_t nl = strlen(ws->items[i].name);
        if (nl > 255 || t->ndim > 8) { w.err = 1; break; }
        wu8(&w, (unsigned)nl);
        wraw(&w, ws->items[i].name, nl);
        wu8(&w, (unsigned)t->ndim);
        for (int d = 0; d < t->ndim; d++) wu32(&w, (uint32_t)t->shape[d]);
        uint64_t nbytes = (uint64_t)t->n * 4;
        wu64(&w, nbytes);
        float *h = (float *)malloc(nbytes ? nbytes : 1);
        if (!h) { w.err = 1; break; }
        mem_d2h(h, t->data, (size_t)nbytes);
        wraw(&w, h, (size_t)nbytes);
        /* wipe staging buffer: weights are sensitive key material */
        memset(h, 0, (size_t)nbytes);
        free(h);
    }
    wraw(&w, "ADAM", 4);
    wf64(&w, opt->b1); wf64(&w, opt->b2);
    wf64(&w, opt->eps); wf64(&w, opt->wd);
    wu32(&w, (uint32_t)opt->n);
    wi64(&w, (int64_t)opt->step);
    for (int i = 0; i < opt->n && !w.err; i++) {
        /* ws_index: resolve via pointer match against the store */
        int idx = -1;
        for (int j = 0; j < ws->count; j++) {
            if (ws->items[j].t->data == opt->params[i]) { idx = j; break; }
        }
        if (idx < 0) { w.err = 1; break; }
        uint64_t nbytes = (uint64_t)opt->sizes[i] * 4;
        wu32(&w, (uint32_t)idx);
        wf64(&w, opt->lr[i]);
        wu64(&w, nbytes);
        wraw(&w, opt->m[i], (size_t)nbytes);
        wu64(&w, nbytes);
        wraw(&w, opt->v[i], (size_t)nbytes);
    }
    /* trailing checksum over all preceding bytes */
    uint64_t sum = w.h;
    int werr = w.err;
    if (!werr) {
        unsigned char b[8];
        for (int i = 0; i < 8; i++) b[i] = (unsigned char)(sum >> (8 * i));
        if (fwrite(b, 1, 8, w.f) != 8) werr = 1;
    }
    if (fclose(w.f)) werr = 1;
    if (werr) {
        unlink(tmp);
        snprintf(err, errcap, "failed writing %s; target %s untouched", tmp, path);
        return -1;
    }
    /* validate the fresh file by re-reading before it replaces anything */
    {
        unsigned char *buf = NULL;
        size_t len = 0;
        char verr[256] = {0};
        if (read_all(tmp, &buf, &len, verr, sizeof(verr))) {
            unlink(tmp);
            snprintf(err, errcap, "fresh file failed re-read: %s", verr);
            return -1;
        }
        uint64_t file_sum = 0;
        memcpy(&file_sum, buf + len - 8, 8); /* LE host */
        uint64_t calc = fnv1a(buf, len - 8, 14695981039346656037ULL);
        int ok = (memcmp(buf, NCK1_MAGIC, 4) == 0 && file_sum == calc);
        /* wipe + free: buffer held full weights */
        memset(buf, 0, len);
        free(buf);
        if (!ok) {
            unlink(tmp);
            snprintf(err, errcap, "fresh file failed validation; target %s untouched", path);
            return -1;
        }
    }
    /* durable publish: fsync file, rename, fsync directory */
    {
        int fd = open(tmp, O_RDONLY);
        if (fd < 0 || fsync(fd)) {
            if (fd >= 0) close(fd);
            unlink(tmp);
            snprintf(err, errcap, "fsync failed on %s", tmp);
            return -1;
        }
        close(fd);
        if (rename(tmp, path)) {
            unlink(tmp);
            snprintf(err, errcap, "rename %s -> %s failed: %s", tmp, path, strerror(errno));
            return -1;
        }
        /* directory durability: the rename itself must survive a crash */
        char dir[4096];
        snprintf(dir, sizeof(dir), "%s", path);
        char *slash = strrchr(dir, '/');
        int dfd;
        if (slash) { *slash = 0; dfd = open(dir[0] ? dir : ".", O_RDONLY | O_DIRECTORY); }
        else dfd = open(".", O_RDONLY | O_DIRECTORY);
        if (dfd >= 0) { fsync(dfd); close(dfd); }
    }
    return 0;
}

/* ---------------- load ---------------- */

int ckpt_load_training(const char *path, WeightStore *ws, AdamW *opt,
                       const CkptExpect *ex, CkptRunState *out_rs,
                       char *err, size_t errcap) {
    if (err && errcap) err[0] = 0;
    unsigned char *buf = NULL;
    size_t len = 0;
    if (read_all(path, &buf, &len, err, errcap)) return -1;

    /* checksum first, before any parsing */
    uint64_t file_sum;
    memcpy(&file_sum, buf + len - 8, 8);
    if (fnv1a(buf, len - 8, 14695981039346656037ULL) != file_sum) {
        snprintf(err, errcap, "%s: checksum mismatch (corrupt or truncated)", path);
        memset(buf, 0, len); free(buf);
        return -1;
    }
    CR r = {buf, buf + len - 8, 0};
    char magic[4];
    rraw(&r, magic, 4);
    if (r.err || memcmp(magic, NCK1_MAGIC, 4) != 0) {
        snprintf(err, errcap, "%s: not an NCK1 checkpoint", path);
        memset(buf, 0, len); free(buf);
        return -1;
    }
    if (ru32(&r) != NCK1_VERSION) {
        snprintf(err, errcap, "%s: unsupported version (want %u)", path, NCK1_VERSION);
        memset(buf, 0, len); free(buf);
        return -1;
    }
    CkptRunState rs;
    memset(&rs, 0, sizeof(rs));
    rs.global_step = (long)ri64(&r);
    rs.batch_rng = ru64(&r);
    rs.base_lr = rf64(&r); rs.warmup = rf64(&r); rs.total = rf64(&r);
    rs.batch_B = (int)ri32(&r);
    rs.accum_pending = (int)ri32(&r);
    rs.n_rows = (long)ri64(&r);
    rs.row_len = (int)ri32(&r);
    rs.data_hash = ru64(&r);
    size_t dl = 0, tl = 0, jl = 0;
    const char *dpath = rstr(&r, &dl);
    rs.conf_hash = ru64(&r);
    const char *tokid = rstr(&r, &tl);
    const char *jblob = rstr(&r, &jl);
    (void)jblob;
    if (r.err || !dpath || !tokid) {
        snprintf(err, errcap, "%s: truncated header", path);
        memset(buf, 0, len); free(buf);
        return -1;
    }
    /* NUL-terminate copies for comparison (bounded, validated above) */
    char *dpathz = (char *)malloc(dl + 1), *tokidz = (char *)malloc(tl + 1);
    if (!dpathz || !tokidz) {
        snprintf(err, errcap, "out of memory");
        free(dpathz); free(tokidz);
        memset(buf, 0, len); free(buf);
        return -1;
    }
    memcpy(dpathz, dpath, dl); dpathz[dl] = 0;
    memcpy(tokidz, tokid, tl); tokidz[tl] = 0;

    if (rs.accum_pending != 0) {
        snprintf(err, errcap, "%s: mid-accumulation checkpoint (pending=%d); not supported",
                 path, rs.accum_pending);
        goto fail;
    }
    /* ---- identity checks: dataset / config / schedule ---- */
    if (rs.batch_B != ex->batch_B || rs.n_rows != ex->n_rows || rs.row_len != ex->row_len ||
        rs.data_hash != ex->data_hash) {
        snprintf(err, errcap, "%s: dataset mismatch (B/n_rows/row_len/hash)", path);
        goto fail;
    }
    if (rs.conf_hash != ex->conf_hash) {
        snprintf(err, errcap, "%s: model config mismatch (conf hash)", path);
        goto fail;
    }
    {
        const char *want_tok = ex->tokenizer_id ? ex->tokenizer_id : "";
        if (strcmp(tokidz, want_tok) != 0) {
            snprintf(err, errcap, "%s: tokenizer identity mismatch ('%s' vs '%s')",
                     path, tokidz, want_tok);
            goto fail;
        }
    }
    if (rs.base_lr != ex->base_lr || rs.warmup != ex->warmup || rs.total != ex->total) {
        snprintf(err, errcap, "%s: schedule mismatch (lr/warmup/total)", path);
        goto fail;
    }
    /* ---- tensors ---- */
    {
        uint32_t n = ru32(&r);
        if (r.err || (int)n != ws->count) {
            snprintf(err, errcap, "%s: tensor count mismatch (%u vs %d)",
                     path, r.err ? 0 : n, ws->count);
            goto fail;
        }
        /* validate all metadata before touching live weights */
        typedef struct { int idx; uint64_t nbytes; const unsigned char *data; } TImg;
        TImg *imgs = (TImg *)calloc(n ? n : 1, sizeof(TImg));
        if (!imgs) { snprintf(err, errcap, "out of memory"); goto fail; }
        int ok = 1;
        for (uint32_t i = 0; i < n && ok; i++) {
            unsigned nl = ru8(&r);
            const unsigned char *nm = r.p;
            if (r.err || nl == 0 || nl >= sizeof(ws->items[0].name) ||
                (size_t)(r.end - r.p) < nl) { ok = 0; break; }
            r.p += nl;
            char name[256];
            memcpy(name, nm, nl); name[nl] = 0;
            if (i >= (uint32_t)ws->count || strcmp(name, ws->items[i].name) != 0) {
                snprintf(err, errcap, "%s: tensor[%u] name/order mismatch ('%s')", path, i, name);
                ok = 0; break;
            }
            unsigned nd = ru8(&r);
            Tensor *t = ws->items[i].t;
            if ((int)nd != t->ndim) {
                snprintf(err, errcap, "%s: tensor '%s' ndim mismatch", path, name);
                ok = 0; break;
            }
            for (int d = 0; d < t->ndim; d++) {
                if ((int64_t)ru32(&r) != t->shape[d]) {
                    snprintf(err, errcap, "%s: tensor '%s' shape mismatch", path, name);
                    ok = 0; break;
                }
            }
            uint64_t nb = ru64(&r);
            if (!ok || nb != (uint64_t)t->n * 4 || (size_t)(r.end - r.p) < nb) {
                snprintf(err, errcap, "%s: tensor '%s' size mismatch", path, name);
                ok = 0; break;
            }
            imgs[i].idx = (int)i; imgs[i].nbytes = nb; imgs[i].data = r.p;
            r.p += nb;
        }
        if (!ok || r.err) { free(imgs); goto fail; }
        /* metadata fully validated: apply */
        for (uint32_t i = 0; i < n; i++)
            mem_h2d(ws->items[imgs[i].idx].t->data, imgs[i].data, (size_t)imgs[i].nbytes);
        free(imgs);
    }
    /* ---- optimizer ---- */
    {
        char om[4];
        rraw(&r, om, 4);
        double b1 = rf64(&r), b2 = rf64(&r), ep = rf64(&r), wd = rf64(&r);
        uint32_t on = ru32(&r);
        long ostep = (long)ri64(&r);
        if (r.err || memcmp(om, "ADAM", 4) != 0 || (int)on != opt->n ||
            b1 != ex->b1 || b2 != ex->b2 || ep != ex->eps || wd != ex->wd) {
            snprintf(err, errcap, "%s: optimizer mismatch (section/n/hyperparams)", path);
            goto fail;
        }
        if (ostep < 0 || ostep > (1L << 40)) {
            snprintf(err, errcap, "%s: implausible optimizer step %ld", path, ostep);
            goto fail;
        }
        /* Record moment locations during validation; apply only after the
         * whole section (including trailing-garbage check) passes. */
        typedef struct { const unsigned char *mp, *vp; uint64_t n; } MImg;
        MImg *mimgs = (MImg *)calloc(opt->n ? (size_t)opt->n : 1, sizeof(MImg));
        if (!mimgs) { snprintf(err, errcap, "out of memory"); goto fail; }
        int mok = 1;
        for (int i = 0; i < opt->n && mok; i++) {
            uint32_t idx = ru32(&r);
            double lrm = rf64(&r);
            uint64_t mnb = ru64(&r);
            const unsigned char *mp = r.p;
            if (r.err || (size_t)(r.end - r.p) < mnb) {
                snprintf(err, errcap, "%s: truncated moments (param %d)", path, i);
                mok = 0; break;
            }
            r.p += mnb;
            uint64_t vnb = ru64(&r);
            const unsigned char *vp = r.p;
            if (r.err || (size_t)(r.end - r.p) < vnb) {
                snprintf(err, errcap, "%s: truncated moments (param %d)", path, i);
                mok = 0; break;
            }
            r.p += vnb;
            /* per-param checks before any apply */
            if (idx >= (uint32_t)ws->count ||
                ws->items[idx].t->data != opt->params[i] ||
                mnb != (uint64_t)opt->sizes[i] * 4 || vnb != mnb || lrm != opt->lr[i]) {
                snprintf(err, errcap, "%s: optimizer param %d mismatch (map/size/lr)", path, i);
                mok = 0; break;
            }
            mimgs[i].mp = mp; mimgs[i].vp = vp; mimgs[i].n = mnb;
        }
        if (!mok) { free(mimgs); goto fail; }
        if (r.p != r.end) {
            snprintf(err, errcap, "%s: trailing garbage (%ld bytes)", path,
                     (long)(r.end - r.p));
            free(mimgs);
            goto fail;
        }
        for (int i = 0; i < opt->n; i++) {
            memcpy(opt->m[i], mimgs[i].mp, (size_t)mimgs[i].n);
            memcpy(opt->v[i], mimgs[i].vp, (size_t)mimgs[i].n);
        }
        free(mimgs);
        opt->step = ostep;
    }
    rs.data_path = NULL; /* caller owns path strings; scalars only */
    rs.tokenizer_id = NULL;
    rs.train_cfg_json = NULL;
    if (out_rs) *out_rs = rs;
    free(dpathz); free(tokidz);
    memset(buf, 0, len); free(buf);
    return 0;

fail:
    free(dpathz); free(tokidz);
    memset(buf, 0, len); free(buf);
    return -1;
}

/* ---------------- weights-only load (inference path) ---------------- */

/* Load ONLY the tensor section of an NCK1 file (weights + shapes). Lets
 * every inference/verify command accept a training checkpoint directly.
 * Checksum is verified first; shapes feed tensor_alloc normally. */
int ckpt_load_weights_only(const char *path, WeightStore *ws, char *err, size_t errcap) {
    if (err && errcap) err[0] = 0;
    unsigned char *buf = NULL;
    size_t len = 0;
    if (read_all(path, &buf, &len, err, errcap)) return -1;
    uint64_t file_sum;
    memcpy(&file_sum, buf + len - 8, 8);
    if (fnv1a(buf, len - 8, 14695981039346656037ULL) != file_sum) {
        snprintf(err, errcap, "%s: checksum mismatch", path);
        memset(buf, 0, len); free(buf);
        return -1;
    }
    CR r = {buf, buf + len - 8, 0};
    char magic[4];
    rraw(&r, magic, 4);
    if (r.err || memcmp(magic, NCK1_MAGIC, 4) != 0 || ru32(&r) != NCK1_VERSION) {
        snprintf(err, errcap, "%s: not an NCK1 checkpoint", path);
        memset(buf, 0, len); free(buf);
        return -1;
    }
    /* skip run-state header (variable-length strings parsed, not trusted) */
    (void)ri64(&r); (void)ru64(&r);
    (void)rf64(&r); (void)rf64(&r); (void)rf64(&r);
    (void)ri32(&r); (void)ri32(&r); (void)ri64(&r); (void)ri32(&r);
    (void)ru64(&r);
    { size_t l = 0; (void)rstr(&r, &l); }
    (void)ru64(&r);
    { size_t l = 0; (void)rstr(&r, &l); }
    { size_t l = 0; (void)rstr(&r, &l); }
    uint32_t n = ru32(&r);
    if (r.err || n > (1u << 24)) {
        snprintf(err, errcap, "%s: bad tensor count", path);
        memset(buf, 0, len); free(buf);
        return -1;
    }
    ws->items = (NamedTensor *)calloc(n ? n : 1, sizeof(NamedTensor));
    if (!ws->items) {
        snprintf(err, errcap, "out of memory");
        memset(buf, 0, len); free(buf);
        return -1;
    }
    ws->count = 0;
    for (uint32_t i = 0; i < n; i++) {
        unsigned nl = ru8(&r);
        if (r.err || nl == 0 || nl >= sizeof(ws->items[0].name)) goto wfail;
        if ((size_t)(r.end - r.p) < nl) goto wfail;
        memcpy(ws->items[i].name, r.p, nl);
        ws->items[i].name[nl] = 0;
        r.p += nl;
        unsigned nd = ru8(&r);
        if (r.err || nd > NCM_MAX_DIMS) goto wfail;
        int64_t shape[NCM_MAX_DIMS] = {0};
        for (unsigned d = 0; d < nd; d++) shape[d] = (int64_t)ru32(&r);
        uint64_t nb = ru64(&r);
        if (r.err || (size_t)(r.end - r.p) < nb) goto wfail;
        Tensor *t = tensor_alloc((int)nd, shape);
        if ((uint64_t)t->n * 4 != nb) {
            tensor_free(t);
            snprintf(err, errcap, "%s: tensor '%s' size mismatch", path, ws->items[i].name);
            goto wfail;
        }
        mem_h2d(t->data, r.p, (size_t)nb);
        r.p += nb;
        ws->items[i].t = t;
        ws->count++;
    }
    memset(buf, 0, len); free(buf);
    return 0;
wfail:
    if (!err[0]) snprintf(err, errcap, "%s: truncated tensor section", path);
    for (int i = 0; i < ws->count; i++) tensor_free(ws->items[i].t);
    free(ws->items);
    ws->items = NULL; ws->count = 0;
    memset(buf, 0, len); free(buf);
    return -1;
}

/* ---------------- info ---------------- */

int ckpt_info(const char *path) {
    char err[256] = {0};
    unsigned char *buf = NULL;
    size_t len = 0;
    if (read_all(path, &buf, &len, err, sizeof(err))) {
        fprintf(stderr, "ckpt-info: %s\n", err);
        return 1;
    }
    uint64_t file_sum;
    memcpy(&file_sum, buf + len - 8, 8);
    int cksum_ok = (fnv1a(buf, len - 8, 14695981039346656037ULL) == file_sum);
    CR r = {buf, buf + len - 8, 0};
    char magic[4];
    rraw(&r, magic, 4);
    if (r.err || memcmp(magic, NCK1_MAGIC, 4) != 0) {
        printf("file: %s\nformat: NOT NCK1 (probe=%d)\nchecksum: %s\n", path,
               ckpt_probe(path), cksum_ok ? "n/a" : "n/a");
        memset(buf, 0, len); free(buf);
        return 1;
    }
    uint32_t ver = ru32(&r);
    long gstep = (long)ri64(&r);
    unsigned long long brng = ru64(&r);
    double lr = rf64(&r), wu = rf64(&r), tot = rf64(&r);
    int B = (int)ri32(&r), acc = (int)ri32(&r);
    long nrows = (long)ri64(&r);
    int rlen = (int)ri32(&r);
    uint64_t dh = ru64(&r);
    size_t dl = 0, tl = 0, jl = 0;
    const char *dp = rstr(&r, &dl);
    uint64_t ch = ru64(&r);
    const char *tk = rstr(&r, &tl);
    const char *js = rstr(&r, &jl);
    uint32_t n = ru32(&r);
    if (r.err) {
        printf("file: %s\nformat: NCK1 (truncated header)\nchecksum: %s\n", path,
               cksum_ok ? "OK" : "MISMATCH");
        memset(buf, 0, len); free(buf);
        return 1;
    }
    uint64_t wbytes = 0;
    for (uint32_t i = 0; i < n && !r.err; i++) {
        unsigned nl = ru8(&r); r.p += nl;
        unsigned nd = ru8(&r);
        for (unsigned d = 0; d < nd; d++) (void)ru32(&r);
        uint64_t nb = ru64(&r); r.p += nb;
        wbytes += nb;
    }
    char om[4] = {0, 0, 0, 0};
    rraw(&r, om, 4);
    double b1 = rf64(&r), b2 = rf64(&r), ep = rf64(&r), wd = rf64(&r);
    uint32_t on = ru32(&r);
    long ostep = (long)ri64(&r);
    uint64_t mbytes = 0;
    for (uint32_t i = 0; i < on && !r.err; i++) {
        (void)ru32(&r); (void)rf64(&r);
        uint64_t mnb = ru64(&r); r.p += mnb;
        uint64_t vnb = ru64(&r); r.p += vnb;
        mbytes += mnb + vnb;
    }
    printf("file: %s (%llu bytes)\n", path, (unsigned long long)len);
    printf("format: NCK1 v%u  checksum: %s\n", ver, cksum_ok ? "OK" : "MISMATCH");
    printf("global_step: %ld  optimizer_step: %ld  accum_pending: %d\n", gstep, ostep, acc);
    printf("schedule: lr=%.6g warmup=%.6g total=%.6g\n", lr, wu, tot);
    printf("optimizer: ADAM(%.4f,%.4f,eps=%.1g,wd=%.4g) params=%u moments=%.1f MiB\n",
           b1, b2, ep, wd, on, mbytes / 1048576.0);
    printf("batch: B=%d rng=0x%016llx  data: rows=%ld row_len=%d hash=%016llx\n",
           B, brng, nrows, rlen, (unsigned long long)dh);
    printf("data_path: %.*s\n", (int)dl, dp ? dp : "");
    printf("conf_hash: %016llx  tokenizer: %.*s\n", (unsigned long long)ch,
           (int)tl, tk ? tk : "");
    printf("tensors: %u weights=%.1f MiB\n", n, wbytes / 1048576.0);
    printf("cfg: %.*s\n", (int)jl, js ? js : "");
    if (r.err || r.p != r.end || memcmp(om, "ADAM", 4) != 0) {
        printf("status: STRUCTURALLY INVALID\n");
        memset(buf, 0, len); free(buf);
        return 1;
    }
    printf("status: valid, exact-resume ready\n");
    memset(buf, 0, len); free(buf);
    return cksum_ok ? 0 : 1;
}
