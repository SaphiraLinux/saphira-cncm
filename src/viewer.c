/* viewer.c — native SDL2 live viewer for CNCM training and solving.
 *
 * A render thread owns the window; computation threads push small state
 * snapshots (ViewSnap) via trylock — rendering NEVER blocks training or
 * solving, and snapshots carry plain data only (no arena, no RNG, no
 * model math), so the viewer cannot change numerics or determinism.
 * Without a display SDL_Init fails and every entry point below becomes
 * a no-op (SDL_VIDEODRIVER=dummy exercises the real path headless).
 */
#include "ncm.h"

#ifdef CNCM_HAVE_SDL2
#include <SDL.h>
#include <stdio.h>
#include <string.h>

/* ---------------- 5x7 bitmap font (uppercase + digits + punctuation) ----
 * One byte per row, bits [4..0] left to right. Verified by the md5 test
 * (tests/test_viewer.sh renders the full strip to BMP). */
static const char VFONT_CHARS[] =
    " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:%/-()+=" "<>!";
static const unsigned char VFONT[][7] = {
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00}, /* space */
    {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}, /* A */
    {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}, /* B */
    {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}, /* C */
    {0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}, /* D */
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}, /* E */
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}, /* F */
    {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}, /* G */
    {0x11,0x11,0x11,0x1F,0x11,0x11,0x11}, /* H */
    {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}, /* I */
    {0x07,0x02,0x02,0x02,0x02,0x12,0x0C}, /* J */
    {0x11,0x12,0x14,0x18,0x14,0x12,0x11}, /* K */
    {0x10,0x10,0x10,0x10,0x10,0x10,0x1F}, /* L */
    {0x11,0x1B,0x15,0x15,0x11,0x11,0x11}, /* M */
    {0x11,0x19,0x15,0x13,0x11,0x11,0x11}, /* N */
    {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}, /* O */
    {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}, /* P */
    {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}, /* Q */
    {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}, /* R */
    {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}, /* S */
    {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}, /* T */
    {0x11,0x11,0x11,0x11,0x11,0x11,0x0E}, /* U */
    {0x11,0x11,0x11,0x11,0x11,0x0A,0x04}, /* V */
    {0x11,0x11,0x11,0x15,0x15,0x15,0x0A}, /* W */
    {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}, /* X */
    {0x11,0x11,0x0A,0x04,0x04,0x04,0x04}, /* Y */
    {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}, /* Z */
    {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, /* 0 */
    {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, /* 1 */
    {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}, /* 2 */
    {0x1F,0x02,0x04,0x02,0x01,0x11,0x0E}, /* 3 */
    {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, /* 4 */
    {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}, /* 5 */
    {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}, /* 6 */
    {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, /* 7 */
    {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, /* 8 */
    {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}, /* 9 */
    {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}, /* . */
    {0x00,0x00,0x00,0x00,0x0C,0x0C,0x08}, /* , */
    {0x00,0x0C,0x0C,0x00,0x0C,0x0C,0x00}, /* : */
    {0x18,0x19,0x02,0x04,0x08,0x13,0x03}, /* % */
    {0x01,0x01,0x02,0x04,0x08,0x10,0x10}, /* / */
    {0x00,0x00,0x00,0x1F,0x00,0x00,0x00}, /* - */
    {0x02,0x04,0x08,0x08,0x08,0x04,0x02}, /* ( */
    {0x08,0x04,0x02,0x02,0x02,0x04,0x08}, /* ) */
    {0x00,0x04,0x04,0x1F,0x04,0x04,0x00}, /* + */
    {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00}, /* = */
    {0x02,0x04,0x08,0x10,0x08,0x04,0x02}, /* < */
    {0x08,0x04,0x02,0x01,0x02,0x04,0x08}, /* > */
    {0x04,0x04,0x04,0x04,0x04,0x00,0x04}, /* ! */
};

static const unsigned char *vfont_glyph(char c) {
    const char *p = strchr(VFONT_CHARS, c);
    if (!p) return VFONT[0];
    return VFONT[p - VFONT_CHARS];
}

/* draw one glyph at (x,y), scale s, onto the current render target */
static void vfont_draw(SDL_Renderer *rn, int x, int y, char c, int s,
                       Uint8 r, Uint8 gg, Uint8 b) {
    const unsigned char *gl = vfont_glyph(c);
    SDL_SetRenderDrawColor(rn, r, gg, b, 255);
    for (int row = 0; row < 7; row++)
        for (int col = 0; col < 5; col++)
            if (gl[row] & (1 << (4 - col))) {
                SDL_Rect rc = {x + col * s, y + row * s, s, s};
                SDL_RenderFillRect(rn, &rc);
            }
}

static void vtext(SDL_Renderer *rn, int x, int y, const char *s, int sc,
                  Uint8 r, Uint8 gg, Uint8 b) {
    for (; *s; s++) { vfont_draw(rn, x, y, *s, sc, r, gg, b); x += 6 * sc; }
}

/* ---------------- viewer state ---------------- */
static SDL_Window *Vwin;
static SDL_Renderer *Vren;
static SDL_Thread *Vthr;
static SDL_mutex *Vmtx;
static SDL_atomic_t Vstop;
static int Vactive;
static ViewSnap Vsnap;
static int Vdirty;
/* loss history (render thread only) */
#define VHIST 512
static double Vhist[VHIST];
static int Vhn;
static int Vlast_step = -1;

static void vdraw_board(int ox, int oy, int cell) {
    /* cells */
    for (int r = 0; r < 9; r++) for (int c = 0; c < 9; c++) {
        int i = r * 9 + c;
        int x = ox + c * cell, y = oy + r * cell;
        Uint8 br = 255, bg = 255, bb2 = 255;
        if (Vsnap.clues[0] && Vsnap.clues[i] != '.' && Vsnap.clues[i] != 0) {
            br = 242; bg = 242; bb2 = 242;
        }
        int bad = 0;
        for (int k = 0; k < Vsnap.nr; k++) if (Vsnap.bad_r[k] == r) bad = 1;
        for (int k = 0; k < Vsnap.nc; k++) if (Vsnap.bad_c[k] == c) bad = 1;
        for (int k = 0; k < Vsnap.nb; k++)
            if (Vsnap.bad_b[k] == (r / 3) * 3 + c / 3) bad = 1;
        if (bad) { br = 255; bg = 215; bb2 = 215; }
        SDL_SetRenderDrawColor(Vren, br, bg, bb2, 255);
        SDL_Rect rc = {x, y, cell, cell};
        SDL_RenderFillRect(Vren, &rc);
        char ch = 0;
        Uint8 tr = 11, tg = 92, tb = 173;
        if (Vsnap.clues[0] && Vsnap.clues[i] != '.' && Vsnap.clues[i] != 0) {
            ch = Vsnap.clues[i]; tr = 17; tg = 17; tb = 17;
        } else if (Vsnap.board[0] && Vsnap.board[i] >= '1' && Vsnap.board[i] <= '9') {
            ch = Vsnap.board[i];
        }
        if (ch) {
            int sc = cell / 12;
            if (sc < 1) sc = 1;
            int tw = 5 * sc, th = 7 * sc;
            vfont_draw(Vren, x + (cell - tw) / 2, y + (cell - th) / 2,
                       ch, sc, tr, tg, tb);
        }
    }
    /* grid lines: thin inside, thick on box borders */
    for (int i = 0; i <= 9; i++) {
        int thick = (i % 3 == 0);
        SDL_SetRenderDrawColor(Vren, 34, 34, 34, 255);
        int p = ox + i * cell - (thick ? 1 : 0);
        SDL_Rect v = {p, oy - (thick ? 1 : 0), thick ? 3 : 1, 9 * cell + (thick ? 2 : 0)};
        SDL_RenderFillRect(Vren, &v);
        int q = oy + i * cell - (thick ? 1 : 0);
        SDL_Rect h = {ox - (thick ? 1 : 0), q, 9 * cell + (thick ? 2 : 0), thick ? 3 : 1};
        SDL_RenderFillRect(Vren, &h);
    }
}

static void vdraw_panel(int x0, int y0, int sc) {
    char buf[192];
    int y = y0, lh = 9 * sc;
    vtext(Vren, x0, y, Vsnap.mode == 0 ? "MODE: TRAINING" : "MODE: SOLVING", sc, 17, 17, 17); y += lh;
    snprintf(buf, sizeof(buf), "MODEL: %s", Vsnap.model);
    vtext(Vren, x0, y, buf, sc, 17, 17, 17); y += lh;
    snprintf(buf, sizeof(buf), "BACKEND: %s", Vsnap.backend);
    vtext(Vren, x0, y, buf, sc, 17, 17, 17); y += lh;
    if (Vsnap.mode == 0) {
        snprintf(buf, sizeof(buf), "STEP: %d/%d", Vsnap.step, Vsnap.total);
        vtext(Vren, x0, y, buf, sc, 17, 17, 17); y += lh;
        snprintf(buf, sizeof(buf), "LOSS: %.4F", Vsnap.loss);
        vtext(Vren, x0, y, buf, sc, 17, 17, 17); y += lh;
        snprintf(buf, sizeof(buf), "ACC: %.1F%%", Vsnap.acc * 100.0);
        vtext(Vren, x0, y, buf, sc, 17, 17, 17); y += lh;
        snprintf(buf, sizeof(buf), "LR: %.2E", Vsnap.lr);
        vtext(Vren, x0, y, buf, sc, 17, 17, 17); y += lh;
    } else {
        snprintf(buf, sizeof(buf), "LEVEL: %d/%d", Vsnap.step, Vsnap.total);
        vtext(Vren, x0, y, buf, sc, 17, 17, 17); y += lh;
        snprintf(buf, sizeof(buf), "VALID: %s", Vsnap.board[0] ? (Vsnap.valid ? "YES" : "NO") : "-");
        vtext(Vren, x0, y, buf, sc, Vsnap.valid ? 26 : 180, Vsnap.valid ? 127 : 30, Vsnap.valid ? 55 : 30); y += lh;
        y += lh; /* keep rows aligned with training panel */
        y += lh;
    }
    snprintf(buf, sizeof(buf), "ELAPSED: %.1FS", Vsnap.elapsed);
    vtext(Vren, x0, y, buf, sc, 17, 17, 17); y += lh;
    if (Vsnap.ckpt) { vtext(Vren, x0, y, "CKPT: SAVED", sc, 26, 127, 55); }
    else { vtext(Vren, x0, y, "CKPT: -", sc, 17, 17, 17); }
    y += lh;
    /* wrap status over up to 2 lines */
    char st[160];
    snprintf(st, sizeof(st), "%s", Vsnap.status);
    st[38] = 0;
    vtext(Vren, x0, y, st, sc, 80, 80, 80);
}

static void vdraw_curve(int x, int y, int w, int h) {
    SDL_SetRenderDrawColor(Vren, 240, 240, 240, 255);
    SDL_Rect bg = {x, y, w, h};
    SDL_RenderFillRect(Vren, &bg);
    SDL_SetRenderDrawColor(Vren, 200, 200, 200, 255);
    SDL_RenderDrawRect(Vren, &bg);
    if (Vhn < 2) return;
    double mn = Vhist[0], mx = Vhist[0];
    for (int i = 1; i < Vhn; i++) {
        if (Vhist[i] < mn) mn = Vhist[i];
        if (Vhist[i] > mx) mx = Vhist[i];
    }
    if (mx - mn < 1e-9) { mn -= 0.5; mx += 0.5; }
    SDL_SetRenderDrawColor(Vren, 11, 92, 173, 255);
    int prevx = 0, prevy = 0;
    for (int i = 0; i < Vhn; i++) {
        int px = x + (int)((double)i / (VHIST - 1) * (w - 1));
        int py = y + h - 1 - (int)((Vhist[i] - mn) / (mx - mn) * (h - 1));
        if (i) SDL_RenderDrawLine(Vren, prevx, prevy, px, py);
        prevx = px; prevy = py;
    }
}

static void vdraw_levels(int x, int y, int w, int h) {
    int n = Vsnap.nlv > 0 ? Vsnap.nlv : 12;
    int bw = (w - (n - 1) * 4) / n;
    for (int i = 0; i < n && i < VIEW_MAXLV; i++) {
        int done = i < Vsnap.lvl_done_count;
        SDL_SetRenderDrawColor(Vren, done ? 26 : 230, done ? 127 : 230, done ? 55 : 230, 255);
        SDL_Rect rc = {x + i * (bw + 4), y, bw, h};
        SDL_RenderFillRect(Vren, &rc);
    }
}

static void vrender(void) {
    int W, H;
    SDL_GetWindowSize(Vwin, &W, &H);
    SDL_SetRenderDrawColor(Vren, 255, 255, 255, 255);
    SDL_RenderClear(Vren);
    int pad = 16;
    int panel_w = 300;
    int board_px = W - panel_w - pad * 3;
    int avail_h = H - pad * 3 - 64;
    if (board_px > avail_h) board_px = avail_h;
    if (board_px < 180) board_px = 180;
    int cell = board_px / 9;
    board_px = cell * 9;
    int ox = pad, oy = pad;
    vdraw_board(ox, oy, cell);
    int x0 = ox + board_px + pad * 2;
    vdraw_panel(x0, oy, 2);
    int by = oy + board_px + pad;
    int bh = H - by - pad;
    if (bh < 40) bh = 40;
    if (Vsnap.mode == 0) vdraw_curve(ox, by, board_px, bh);
    else vdraw_levels(ox, by, board_px, bh < 60 ? bh : 60);
    SDL_RenderPresent(Vren);
}

static int vthread(void *arg) {
    (void)arg;
    for (;;) {
        if (SDL_AtomicGet(&Vstop)) break;
        SDL_Event ev;
        while (SDL_PollEvent(&ev))
            if (ev.type == SDL_QUIT) SDL_AtomicSet(&Vstop, 1);
        if (SDL_LockMutex(Vmtx) == 0) {
            if (Vdirty) {
                Vdirty = 0;
                /* feed loss history on new training steps */
                if (Vsnap.mode == 0 && Vsnap.step != Vlast_step) {
                    Vlast_step = Vsnap.step;
                    if (Vhn < VHIST) Vhist[Vhn++] = Vsnap.loss;
                    else { memmove(Vhist, Vhist + 1, (VHIST - 1) * sizeof(double)); Vhist[VHIST - 1] = Vsnap.loss; }
                }
            }
            SDL_UnlockMutex(Vmtx);
        }
        vrender();
        SDL_Delay(66);
    }
    return 0;
}

int viewer_open(const char *title) {
    if (Vactive) return 0;
    if (SDL_Init(SDL_INIT_VIDEO) != 0) return 1;
    memset(&Vsnap, 0, sizeof(Vsnap));
    Vwin = SDL_CreateWindow(title ? title : "CNCM",
                            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                            820, 620, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    if (!Vwin) { SDL_Quit(); return 1; }
    Vren = SDL_CreateRenderer(Vwin, -1, SDL_RENDERER_SOFTWARE);
    if (!Vren) { SDL_DestroyWindow(Vwin); Vwin = NULL; SDL_Quit(); return 1; }
    Vmtx = SDL_CreateMutex();
    if (!Vmtx) {
        SDL_DestroyRenderer(Vren); Vren = NULL;
        SDL_DestroyWindow(Vwin); Vwin = NULL;
        SDL_Quit();
        return 1;
    }
    SDL_AtomicSet(&Vstop, 0);
    Vdirty = 0; Vhn = 0; Vlast_step = -1;
    Vthr = SDL_CreateThread(vthread, "cncm-viewer", NULL);
    if (!Vthr) {
        SDL_DestroyMutex(Vmtx); Vmtx = NULL;
        SDL_DestroyRenderer(Vren); Vren = NULL;
        SDL_DestroyWindow(Vwin); Vwin = NULL;
        SDL_Quit();
        return 1;
    }
    Vactive = 1;
    return 0;
}

void viewer_push(const ViewSnap *s) {
    if (!Vactive || !s) return;
    if (SDL_TryLockMutex(Vmtx) == 0) {
        Vsnap = *s;
        Vdirty = 1;
        SDL_UnlockMutex(Vmtx);
    }
    /* trylock: a busy renderer simply skips this frame — training never waits */
}

int viewer_stop_requested(void) {
    if (!Vactive) return 0;
    return SDL_AtomicGet(&Vstop);
}

/* After the final state is pushed: keep the window up until the user
 * closes it (lets the result be seen). No-op headless or on dummy video
 * (tests must not hang), and returns at once if stop was requested. */
void viewer_hold(void) {
    if (!Vactive) return;
    const char *drv = SDL_GetCurrentVideoDriver();
    if (!drv || !strcmp(drv, "dummy")) return;
    printf("viewer: close the window to finish\n");
    fflush(stdout);
    while (!SDL_AtomicGet(&Vstop)) SDL_Delay(100);
}

void viewer_close(void) {
    if (!Vactive) return;
    Vactive = 0;
    SDL_AtomicSet(&Vstop, 1);
    SDL_WaitThread(Vthr, NULL);
    Vthr = NULL;
    SDL_DestroyMutex(Vmtx); Vmtx = NULL;
    SDL_DestroyRenderer(Vren); Vren = NULL;
    SDL_DestroyWindow(Vwin); Vwin = NULL;
    SDL_Quit();
}

int viewer_active(void) { return Vactive; }

/* test/diagnostic helper: render the full charset strip to a BMP file.
 * Works without video init (software surface only). */
int viewer_font_test_bmp(const char *path) {
    if (SDL_Init(0) != 0) return 1;
    int n = (int)strlen(VFONT_CHARS);
    int sc = 3, cw = 6 * sc, chh = 7 * sc + 4;
    SDL_Surface *sf = SDL_CreateRGBSurface(0, n * cw + 8, chh + 8, 32,
                                           0x00FF0000, 0x0000FF00,
                                           0x000000FF, 0xFF000000);
    if (!sf) { SDL_Quit(); return 1; }
    SDL_FillRect(sf, NULL, SDL_MapRGB(sf->format, 255, 255, 255));
    /* blit glyphs directly onto the surface */
    for (int i = 0; i < n; i++) {
        const unsigned char *gl = VFONT[i];
        int x0 = 4 + i * cw, y0 = 4;
        for (int row = 0; row < 7; row++)
            for (int col = 0; col < 5; col++)
                if (gl[row] & (1 << (4 - col))) {
                    SDL_Rect rc = {x0 + col * sc, y0 + row * sc, sc, sc};
                    SDL_FillRect(sf, &rc, SDL_MapRGB(sf->format, 17, 17, 17));
                }
    }
    int rc = SDL_SaveBMP(sf, path);
    SDL_FreeSurface(sf);
    SDL_Quit();
    return rc;
}

#else
/* No SDL2 at build time: every entry point is a silent no-op. */
int viewer_open(const char *title) { (void)title; return 1; }
void viewer_push(const ViewSnap *s) { (void)s; }
int viewer_stop_requested(void) { return 0; }
void viewer_hold(void) { }
void viewer_close(void) { }
int viewer_active(void) { return 0; }
int viewer_font_test_bmp(const char *path) { (void)path; return 2; }
#endif /* CNCM_HAVE_SDL2 */
