/* fwdcheck: train_ncp_forward vs ncp_forward on identical x. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ncm.h"

extern double train_ncp_loss_fwd(Tape *tp, void *m, void *ws, void *c,
                                 const float *x, const int *prefix_ids,
                                 const int *codes, void *l0q, int B, double *ppl);

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.conf model.ncm data.nctrain\n", argv[0]); return 2; }
    NCMConfig cfg;
    if (config_load(&cfg, argv[1])) return 1;
    WeightStore ws = {0};
    if (weights_load(&ws, argv[2])) return 1;
    arena_init(1ULL << 30);
    parena_init(1ULL << 28);
    VQVAEState *vqvae = vqvae_load(&cfg, &ws);
    NCPState *m = ncp_load(&cfg, &ws, vqvae);
    /* build dummy x: random prefix + random codes path skipped; use real data */
    FILE *f = fopen(argv[3], "rb");
    char mg[4]; fread(mg, 1, 4, f);
    uint64_t nr, rl;
    fread(&nr, 8, 1, f); fread(&rl, 8, 1, f);
    int *row = malloc(rl * 4);
    fread(row, 4, rl, f);
    fclose(f);
    int P = ncp_P(m);
    int total = 0;
    for (int i = 0; i < cfg.n_qlevels; i++) total += cfg.levels[i];
    int B = 1;
    /* x via frozen path (same as train step): prefix wte + vq ncm input */
    float *xfull = arena_alloc((size_t)B * (P + total - 0) * 256);
    (void)xfull;
    printf("fwdcheck placeholder\n");
    return 0;
}
