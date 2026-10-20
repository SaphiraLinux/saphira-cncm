/* metrics.c — sacrebleu/rouge-compatible scores (full port lands with eval). */
#include "ncm.h"
#include <stdio.h>
#include <stdlib.h>
double metric_bleu_corpus(char **hyps, char **refs, int n) { (void)hyps; (void)refs; (void)n; return -1; }
double metric_chrf(char **hyps, char **refs, int n) { (void)hyps; (void)refs; (void)n; return -1; }
void metric_rouge(char **hyps, char **refs, int n, double *r1, double *r2, double *rl) {
    (void)hyps; (void)refs; (void)n; *r1 = *r2 = *rl = -1;
}
double gpt2_genppl(const char *a, const char *b, char **hyps, int n, int max_len) {
    (void)a; (void)b; (void)hyps; (void)n; (void)max_len; return -1;
}
