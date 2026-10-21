/* tests/fontcheck.c — render the viewer bitmap font strip to a BMP file.
 * No video init needed (software surface only); headless-safe. */
#include <stdio.h>
#include "ncm.h"

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: fontcheck out.bmp\n"); return 2; }
    return viewer_font_test_bmp(argv[1]);
}
