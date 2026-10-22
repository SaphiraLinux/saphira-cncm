/* tests/ckpt_unit.c — synthetic NCK1 round-trip + refusal tests.
 * No weights, no data files: builds a tiny WeightStore + AdamW in memory.
 * Usage: ./ckpt_unit   (exits nonzero on first failure) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include "ncm.h"

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL %s\n", msg); fails++; } \
    else { printf("ok   %s\n", msg); } } while (0)

/* build a 2-tensor store with known bytes; caller frees via weights_free */
static void make_store(WeightStore *ws, float fill) {
    ws->items = (NamedTensor *)calloc(2, sizeof(NamedTensor));
    ws->count = 2;
    strcpy(ws->items[0].name, "blk.weight");
    strcpy(ws->items[1].name, "blk.bias");
    int64_t s0[2] = {4, 8}, s1[1] = {8};
    ws->items[0].t = tensor_alloc(2, s0);
    ws->items[1].t = tensor_alloc(1, s1);
    float *d0 = (float *)malloc((size_t)ws->items[0].t->n * 4);
    float *d1 = (float *)malloc((size_t)ws->items[1].t->n * 4);
    for (int64_t k = 0; k < ws->items[0].t->n; k++) d0[k] = fill + (float)k * 0.25f;
    for (int64_t k = 0; k < ws->items[1].t->n; k++) d1[k] = fill - (float)k * 0.5f;
    mem_h2d(ws->items[0].t->data, d0, (size_t)ws->items[0].t->n * 4);
    mem_h2d(ws->items[1].t->data, d1, (size_t)ws->items[1].t->n * 4);
    free(d0); free(d1);
}

static AdamW *make_opt(WeightStore *ws) {
    AdamW *o = (AdamW *)calloc(1, sizeof(AdamW));
    o->b1 = 0.9; o->b2 = 0.95; o->eps = 1e-8; o->wd = 0.01;
    o->n = ws->count;
    o->params = malloc(2 * sizeof(float *));
    o->m = malloc(2 * sizeof(float *));
    o->v = malloc(2 * sizeof(float *));
    o->g = malloc(2 * sizeof(float *));
    o->sizes = malloc(2 * sizeof(int64_t));
    o->lr = malloc(2 * sizeof(double));
    for (int i = 0; i < 2; i++) {
        o->params[i] = ws->items[i].t->data;
        o->sizes[i] = ws->items[i].t->n;
        o->m[i] = calloc((size_t)o->sizes[i], 4);
        o->v[i] = calloc((size_t)o->sizes[i], 4);
        o->g[i] = calloc((size_t)o->sizes[i], 4);
        o->lr[i] = 1.0;
        for (int64_t k = 0; k < o->sizes[i]; k++) {
            o->m[i][k] = 0.01f * (float)(i + 1) + 0.001f * (float)k;
            o->v[i][k] = 0.02f * (float)(i + 1) + 0.002f * (float)k;
        }
    }
    o->step = 7;
    return o;
}

static CkptRunState mkrs(void) {
    CkptRunState rs;
    memset(&rs, 0, sizeof(rs));
    rs.global_step = 7; rs.batch_rng = 0x123456789ABCDEFULL;
    rs.base_lr = 3e-4; rs.warmup = 100; rs.total = 1000;
    rs.batch_B = 4; rs.accum_pending = 0;
    rs.n_rows = 64; rs.row_len = 413;
    rs.data_hash = 0xDEADBEEFCAFEBABEULL;
    rs.data_path = "test-data.nctrain";
    rs.conf_hash = 0x0123456789ABCDEFULL;
    rs.tokenizer_id = "unit-test";
    rs.noise_sd = 0.0;
    rs.train_cfg_json = "{\"test\":true}";
    return rs;
}

static CkptExpect mkex(void) {
    CkptExpect ex;
    memset(&ex, 0, sizeof(ex));
    ex.batch_B = 4; ex.n_rows = 64; ex.row_len = 413;
    ex.data_hash = 0xDEADBEEFCAFEBABEULL;
    ex.conf_hash = 0x0123456789ABCDEFULL;
    ex.tokenizer_id = "unit-test";
    ex.base_lr = 3e-4; ex.warmup = 100; ex.total = 1000;
    ex.b1 = 0.9; ex.b2 = 0.95; ex.eps = 1e-8; ex.wd = 0.01;
    return ex;
}

static int file_size(const char *p) {
    FILE *f = fopen(p, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    return (int)n;
}

int main(void) {
    arena_init(1ULL << 20);
    char err[512];
    /* scratch honors $TMPDIR (persistent) with /tmp fallback */
    static char F[1024];
    const char *td = getenv("TMPDIR");
    snprintf(F, sizeof(F), "%s/opencode/ckpt_unit.nckpt", td ? td : "/tmp");

    /* ---- round trip ---- */
    WeightStore ws = {0};
    make_store(&ws, 1.0f);
    AdamW *opt = make_opt(&ws);
    CkptRunState rs = mkrs();
    CHECK(ckpt_probe("/nonexistent-xyz") < 0, "probe missing file");
    CHECK(ckpt_save_atomic(&ws, opt, &rs, F, err, sizeof(err)) == 0, "save ok");
    CHECK(ckpt_probe(F) == 1, "probe NCK1");
    CHECK(file_size(F) > 0, "file nonempty");

    WeightStore ws2 = {0};
    make_store(&ws2, -99.0f); /* different bytes, same shapes */
    AdamW *opt2 = make_opt(&ws2);
    opt2->step = 0;
    CkptExpect ex = mkex();
    CkptRunState out;
    CHECK(ckpt_load_training(F, &ws2, opt2, &ex, &out, err, sizeof(err)) == 0,
          "exact load ok");
    CHECK(out.global_step == 7 && out.batch_rng == 0x123456789ABCDEFULL,
          "run scalars restored");
    CHECK(opt2->step == 7, "optimizer step restored");
    /* weights + moments byte-compared via host staging */
    {
        int same = 1;
        for (int i = 0; i < 2 && same; i++) {
            float *a = malloc((size_t)ws.items[i].t->n * 4);
            float *b = malloc((size_t)ws.items[i].t->n * 4);
            mem_d2h(a, ws.items[i].t->data, (size_t)ws.items[i].t->n * 4);
            mem_d2h(b, ws2.items[i].t->data, (size_t)ws2.items[i].t->n * 4);
            if (memcmp(a, b, (size_t)ws.items[i].t->n * 4)) same = 0;
            if (memcmp(opt->m[i], opt2->m[i], (size_t)opt->sizes[i] * 4)) same = 0;
            if (memcmp(opt->v[i], opt2->v[i], (size_t)opt->sizes[i] * 4)) same = 0;
            free(a); free(b);
        }
        CHECK(same, "weights+moments bit-identical after load");
    }
    /* weights-only shim path */
    {
        WeightStore ws3 = {0};
        CHECK(ckpt_load_weights_only(F, &ws3, err, sizeof(err)) == 0 &&
              ws3.count == 2, "weights-only load ok");
        weights_free(&ws3);
    }
    CHECK(ckpt_info(F) == 0, "ckpt-info valid");

    /* ---- refusals: corruptions must not touch live state ---- */
    /* snapshot live bytes to prove no modification on refuse */
    float *live0 = malloc((size_t)ws2.items[0].t->n * 4);
    mem_d2h(live0, ws2.items[0].t->data, (size_t)ws2.items[0].t->n * 4);
    /* flip a byte in the middle (payload) */
    {
        FILE *f = fopen(F, "r+b");
        fseek(f, 200, SEEK_SET);
        int c = fgetc(f);
        fseek(f, 200, SEEK_SET);
        fputc(c ^ 0xFF, f);
        fclose(f);
        WeightStore wx = {0};
        make_store(&wx, -99.0f);
        AdamW *ox = make_opt(&wx);
        CkptExpect exx = mkex();
        int rc = ckpt_load_training(F, &wx, ox, &exx, NULL, err, sizeof(err));
        CHECK(rc != 0, "bit-flip refused");
        float *after = malloc((size_t)wx.items[0].t->n * 4);
        mem_d2h(after, wx.items[0].t->data, (size_t)wx.items[0].t->n * 4);
        float *exp = malloc((size_t)wx.items[0].t->n * 4);
        for (int64_t k = 0; k < wx.items[0].t->n; k++)
            exp[k] = -99.0f + (float)k * 0.25f;
        CHECK(memcmp(after, exp, (size_t)wx.items[0].t->n * 4) == 0,
              "refused load left live weights untouched");
        free(after); free(exp);
        weights_free(&wx);
        adamw_free(ox);
        CHECK(ckpt_info(F) != 0, "ckpt-info flags corruption");
    }
    /* restore good file, then truncate */
    CHECK(ckpt_save_atomic(&ws, opt, &rs, F, err, sizeof(err)) == 0, "re-save ok");
    {
        int n = file_size(F);
        FILE *f = fopen(F, "r+b");
        ftruncate(fileno(f), n > 100 ? n - 100 : 0);
        fclose(f);
        WeightStore wx = {0};
        make_store(&wx, -99.0f);
        AdamW *ox = make_opt(&wx);
        CkptExpect exx = mkex();
        CHECK(ckpt_load_training(F, &wx, ox, &exx, NULL, err, sizeof(err)) != 0,
              "truncation refused");
        weights_free(&wx);
        adamw_free(ox);
    }
    /* restore, then identity mismatches */
    CHECK(ckpt_save_atomic(&ws, opt, &rs, F, err, sizeof(err)) == 0, "re-save ok 2");
    {
        WeightStore wx = {0};
        make_store(&wx, -99.0f);
        AdamW *ox = make_opt(&wx);
        CkptExpect exx = mkex();
        exx.base_lr = 0.001;
        CHECK(ckpt_load_training(F, &wx, ox, &exx, NULL, err, sizeof(err)) != 0,
              "schedule mismatch refused");
        exx = mkex();
        exx.tokenizer_id = "other";
        CHECK(ckpt_load_training(F, &wx, ox, &exx, NULL, err, sizeof(err)) != 0,
              "tokenizer mismatch refused");
        exx = mkex();
        exx.data_hash = 0x1111;
        CHECK(ckpt_load_training(F, &wx, ox, &exx, NULL, err, sizeof(err)) != 0,
              "dataset mismatch refused");
        weights_free(&wx);
        adamw_free(ox);
    }
    /* live store from the good load still intact */
    {
        float *after = malloc((size_t)ws2.items[0].t->n * 4);
        mem_d2h(after, ws2.items[0].t->data, (size_t)ws2.items[0].t->n * 4);
        CHECK(memcmp(after, live0, (size_t)ws2.items[0].t->n * 4) == 0,
              "good-loaded state survived later refusals");
        free(after);
    }
    free(live0);
    weights_free(&ws); weights_free(&ws2);
    adamw_free(opt); adamw_free(opt2);
    remove(F);
    if (fails) { printf("%d FAILURES\n", fails); return 1; }
    printf("ckpt_unit: all passed\n");
    return 0;
}
