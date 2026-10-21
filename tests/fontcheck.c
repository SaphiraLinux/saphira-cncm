/* tests/fontcheck.c — render viewer bitmaps to BMP files.
 * fontcheck out.bmp            — full charset strip (software surface only)
 * fontcheck frame PATH MODE    — full viewer frame via the live render path
 *                                (own hidden window + software renderer;
 *                                works headless under the dummy driver).
 * Headless-safe in both modes. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ncm.h"

int main(int argc, char **argv) {
    if (argc == 2) return viewer_font_test_bmp(argv[1]);
    if (argc == 4 && !strcmp(argv[1], "frame"))
        return viewer_test_frame_bmp(argv[2], atoi(argv[3]));
    fprintf(stderr, "usage: fontcheck out.bmp | fontcheck frame out.bmp 0|1\n");
    return 2;
}
