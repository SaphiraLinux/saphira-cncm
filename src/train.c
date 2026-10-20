/* train.c — training path (lands after inference parity; see TRAINING.md). */
#include "ncm.h"
#include <stdio.h>
int train_run(const NCMConfig *c, const char *weights_path, const char *data_path, int max_steps) {
    (void)c; (void)weights_path; (void)data_path; (void)max_steps;
    fprintf(stderr, "train: not yet implemented in this snapshot\n");
    return 1;
}
