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
#include <stdlib.h>
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

/* One frame's worth of render inputs, copied under lock (thread path) or
 * built synthetically (frame test path). Lets the draw code run without
 * touching globals, so tests exercise the exact live pixels. */
typedef struct {
    SDL_Renderer *rn;
    ViewSnap snap;
    double hist[VHIST];
    int hn;
} VFrame;

static void vdraw_board(const VFrame *f, int ox, int oy, int cell) {
    /* cells */
    for (int r = 0; r < 9; r++) for (int c = 0; c < 9; c++) {
        int i = r * 9 + c;
        int x = ox + c * cell, y = oy + r * cell;
        Uint8 br = 255, bg = 255, bb2 = 255;
        if (f->snap.clues[0] && f->snap.clues[i] != '.' && f->snap.clues[i] != 0) {
            br = 242; bg = 242; bb2 = 242;
        }
        int bad = 0;
        for (int k = 0; k < f->snap.nr; k++) if (f->snap.bad_r[k] == r) bad = 1;
        for (int k = 0; k < f->snap.nc; k++) if (f->snap.bad_c[k] == c) bad = 1;
        for (int k = 0; k < f->snap.nb; k++)
            if (f->snap.bad_b[k] == (r / 3) * 3 + c / 3) bad = 1;
        if (bad) { br = 255; bg = 215; bb2 = 215; }
        SDL_SetRenderDrawColor(f->rn, br, bg, bb2, 255);
        SDL_Rect rc = {x, y, cell, cell};
        SDL_RenderFillRect(f->rn, &rc);
        char ch = 0;
        Uint8 tr = 11, tg = 92, tb = 173;
        if (f->snap.clues[0] && f->snap.clues[i] != '.' && f->snap.clues[i] != 0) {
            ch = f->snap.clues[i]; tr = 17; tg = 17; tb = 17;
        } else if (f->snap.board[0] && f->snap.board[i] >= '1' && f->snap.board[i] <= '9') {
            ch = f->snap.board[i];
        }
        if (ch) {
            int sc = cell / 12;
            if (sc < 1) sc = 1;
            int tw = 5 * sc, th = 7 * sc;
            vfont_draw(f->rn, x + (cell - tw) / 2, y + (cell - th) / 2,
                       ch, sc, tr, tg, tb);
        }
    }
    /* grid lines: thin inside, thick on box borders */
    for (int i = 0; i <= 9; i++) {
        int thick = (i % 3 == 0);
        SDL_SetRenderDrawColor(f->rn, 34, 34, 34, 255);
        int p = ox + i * cell - (thick ? 1 : 0);
        SDL_Rect v = {p, oy - (thick ? 1 : 0), thick ? 3 : 1, 9 * cell + (thick ? 2 : 0)};
        SDL_RenderFillRect(f->rn, &v);
        int q = oy + i * cell - (thick ? 1 : 0);
        SDL_Rect h = {ox - (thick ? 1 : 0), q, 9 * cell + (thick ? 2 : 0), thick ? 3 : 1};
        SDL_RenderFillRect(f->rn, &h);
    }
}

static void vdraw_panel(const VFrame *f, int x0, int y0, int sc) {
    char buf[192];
    int y = y0, lh = 9 * sc;
    vtext(f->rn, x0, y, f->snap.mode == 0 ? "MODE: TRAINING" : "MODE: SOLVING", sc, 17, 17, 17); y += lh;
    snprintf(buf, sizeof(buf), "MODEL: %s", f->snap.model);
    vtext(f->rn, x0, y, buf, sc, 17, 17, 17); y += lh;
    snprintf(buf, sizeof(buf), "BACKEND: %s", f->snap.backend);
    vtext(f->rn, x0, y, buf, sc, 17, 17, 17); y += lh;
    if (f->snap.mode == 0) {
        snprintf(buf, sizeof(buf), "STEP: %d/%d", f->snap.step, f->snap.total);
        vtext(f->rn, x0, y, buf, sc, 17, 17, 17); y += lh;
        snprintf(buf, sizeof(buf), "LOSS: %.4F", f->snap.loss);
        vtext(f->rn, x0, y, buf, sc, 17, 17, 17); y += lh;
        snprintf(buf, sizeof(buf), "ACC: %.1F%%", f->snap.acc * 100.0);
        vtext(f->rn, x0, y, buf, sc, 17, 17, 17); y += lh;
        snprintf(buf, sizeof(buf), "LR: %.2E", f->snap.lr);
        vtext(f->rn, x0, y, buf, sc, 17, 17, 17); y += lh;
    } else {
        snprintf(buf, sizeof(buf), "LEVEL: %d/%d", f->snap.step, f->snap.total);
        vtext(f->rn, x0, y, buf, sc, 17, 17, 17); y += lh;
        snprintf(buf, sizeof(buf), "VALID: %s", f->snap.board[0] ? (f->snap.valid ? "YES" : "NO") : "-");
        vtext(f->rn, x0, y, buf, sc, f->snap.valid ? 26 : 180, f->snap.valid ? 127 : 30, f->snap.valid ? 55 : 30); y += lh;
        y += lh; /* keep rows aligned with training panel */
        y += lh;
    }
    snprintf(buf, sizeof(buf), "ELAPSED: %.1FS", f->snap.elapsed);
    vtext(f->rn, x0, y, buf, sc, 17, 17, 17); y += lh;
    if (f->snap.ckpt) { vtext(f->rn, x0, y, "CKPT: SAVED", sc, 26, 127, 55); }
    else { vtext(f->rn, x0, y, "CKPT: -", sc, 17, 17, 17); }
    y += lh;
    /* wrap status over up to 2 lines */
    char st[160];
    snprintf(st, sizeof(st), "%s", f->snap.status);
    st[38] = 0;
    vtext(f->rn, x0, y, st, sc, 80, 80, 80);
}

static void vdraw_curve(const VFrame *f, int x, int y, int w, int h) {
    SDL_SetRenderDrawColor(f->rn, 240, 240, 240, 255);
    SDL_Rect bg = {x, y, w, h};
    SDL_RenderFillRect(f->rn, &bg);
    SDL_SetRenderDrawColor(f->rn, 200, 200, 200, 255);
    SDL_RenderDrawRect(f->rn, &bg);
    if (f->hn < 2) return;
    double mn = f->hist[0], mx = f->hist[0];
    for (int i = 1; i < f->hn; i++) {
        if (f->hist[i] < mn) mn = f->hist[i];
        if (f->hist[i] > mx) mx = f->hist[i];
    }
    if (mx - mn < 1e-9) { mn -= 0.5; mx += 0.5; }
    SDL_SetRenderDrawColor(f->rn, 11, 92, 173, 255);
    int prevx = 0, prevy = 0;
    for (int i = 0; i < f->hn; i++) {
        int px = x + (int)((double)i / (VHIST - 1) * (w - 1));
        int py = y + h - 1 - (int)((f->hist[i] - mn) / (mx - mn) * (h - 1));
        if (i) SDL_RenderDrawLine(f->rn, prevx, prevy, px, py);
        prevx = px; prevy = py;
    }
}

static void vdraw_levels(const VFrame *f, int x, int y, int w, int h) {
    int n = f->snap.nlv > 0 ? f->snap.nlv : 12;
    int bw = (w - (n - 1) * 4) / n;
    for (int i = 0; i < n && i < VIEW_MAXLV; i++) {
        int done = i < f->snap.lvl_done_count;
        SDL_SetRenderDrawColor(f->rn, done ? 26 : 230, done ? 127 : 230, done ? 55 : 230, 255);
        SDL_Rect rc = {x + i * (bw + 4), y, bw, h};
        SDL_RenderFillRect(f->rn, &rc);
    }
}

static void vrender_frame(VFrame *f, int W, int H) {
    SDL_SetRenderDrawColor(f->rn, 255, 255, 255, 255);
    SDL_RenderClear(f->rn);
    int pad = 16;
    int panel_w = 300;
    int board_px = W - panel_w - pad * 3;
    int avail_h = H - pad * 3 - 64;
    if (board_px > avail_h) board_px = avail_h;
    if (board_px < 180) board_px = 180;
    int cell = board_px / 9;
    board_px = cell * 9;
    int ox = pad, oy = pad;
    vdraw_board(f, ox, oy, cell);
    int x0 = ox + board_px + pad * 2;
    vdraw_panel(f, x0, oy, 2);
    int by = oy + board_px + pad;
    int bh = H - by - pad;
    if (bh < 40) bh = 40;
    if (f->snap.mode == 0) vdraw_curve(f, ox, by, board_px, bh);
    else vdraw_levels(f, ox, by, board_px, bh < 60 ? bh : 60);
}

static int vthread(void *arg) {
    (void)arg;
    const char *shotpath = getenv("CNCM_VIEW_SHOT");
    unsigned long frames = 0;
    for (;;) {
        if (SDL_AtomicGet(&Vstop)) break;
        SDL_Event ev;
        while (SDL_PollEvent(&ev))
            if (ev.type == SDL_QUIT) SDL_AtomicSet(&Vstop, 1);
        VFrame f;
        memset(&f, 0, sizeof(f));
        f.rn = Vren;
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
            f.snap = Vsnap;
            f.hn = Vhn < VHIST ? Vhn : VHIST;
            if (f.hn > 0) memcpy(f.hist, Vhist, (size_t)f.hn * sizeof(double));
            SDL_UnlockMutex(Vmtx);
        }
        int W, H;
        SDL_GetWindowSize(Vwin, &W, &H);
        vrender_frame(&f, W, H);
        SDL_RenderPresent(Vren);
        /* diagnostic snapshot of the live backbuffer (real presented pixels):
         * CNCM_VIEW_SHOT=/tmp/shot.bmp dumps every ~20s. Proves what the
         * renderer produces on any backend, no external tools needed. */
        if (shotpath && (++frames % 300 == 0)) {
            SDL_Surface *shot = SDL_CreateRGBSurface(0, W, H, 32,
                0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000);
            if (shot) {
                if (SDL_RenderReadPixels(Vren, NULL, shot->format->format,
                                         shot->pixels, shot->pitch) == 0)
                    SDL_SaveBMP(shot, shotpath);
                SDL_FreeSurface(shot);
            }
        }
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
    {
        /* observability: which backend actually drives the window */
        SDL_RendererInfo ri;
        const char *drv = SDL_GetCurrentVideoDriver();
        const char *rnm = (SDL_GetRendererInfo(Vren, &ri) == 0) ? ri.name : "?";
        fprintf(stderr, "[viewer] video=%s renderer=%s\n", drv ? drv : "?", rnm);
        /* synchronous first frame on the calling thread: proves the
         * renderer presents before the render thread takes over */
        VFrame f0;
        memset(&f0, 0, sizeof(f0));
        f0.rn = Vren;
        f0.snap = Vsnap;
        int W0, H0;
        SDL_GetWindowSize(Vwin, &W0, &H0);
        vrender_frame(&f0, W0, H0);
        SDL_RenderPresent(Vren);
    }
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

/* test helper: render one full viewer frame (mode 0=train, 1=solve) to a
 * BMP using the same vrender_frame the live thread uses, on a private
 * window+software renderer (works headless under the dummy driver).
 * Returns 0 on success. */
static void vframe_demo_snap(ViewSnap *s, int mode) {
    memset(s, 0, sizeof(*s));
    s->mode = mode;
    snprintf(s->model, sizeof(s->model), "sudoku-easy");
    snprintf(s->backend, sizeof(s->backend), "cpu");
    snprintf(s->status, sizeof(s->status), "frame test");
    const char *cl = "..3......"
                     "8......21"
                     ".17......"
                     ".....7..."
                     "....18..."
                     "....7..9."
                     ".5..6...."
                     "..4..2..1"
                     "..69..5..";
    memcpy(s->clues, cl, 81);
    s->clues[81] = 0;
    if (mode == 0) {
        s->step = 12; s->total = 100;
        s->loss = 9.8765; s->acc = 0.873; s->lr = 3e-5; s->elapsed = 123.4;
        s->nlv = 12;
        for (int i = 0; i < 12; i++) {
            s->lvl_loss[i] = 10.0 - 0.2 * i;
            s->lvl_acc[i] = 0.5 + 0.03 * i;
            s->lvl_n[i] = 8;
            s->lvl_ms[i] = 200.0 + 5 * i;
        }
        s->lvl_done_count = 0;
        s->ckpt = 1;
    } else {
        s->step = 12; s->total = 12;
        s->nlv = 12;
        for (int i = 0; i < 12; i++) {
            s->lvl_n[i] = 8;
            s->lvl_ms[i] = 200.0 + 5 * i;
        }
        s->lvl_done_count = 12;
        const char *bd = "921647385764835291583912764452369178179458623836721549217584936698173452345296817";
        memcpy(s->board, bd, 81);
        s->board[81] = 0;
        s->valid = 1;
        s->elapsed = 3.3;
    }
}

int viewer_test_frame_bmp(const char *path, int mode) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) return 1;
    SDL_Window *wn = SDL_CreateWindow("cncm-frame-test", 0, 0, 820, 620, SDL_WINDOW_HIDDEN);
    if (!wn) { SDL_Quit(); return 1; }
    SDL_Renderer *rn = SDL_CreateRenderer(wn, -1, SDL_RENDERER_SOFTWARE);
    if (!rn) { SDL_DestroyWindow(wn); SDL_Quit(); return 1; }
    VFrame f;
    memset(&f, 0, sizeof(f));
    f.rn = rn;
    vframe_demo_snap(&f.snap, mode);
    if (mode == 0) {
        f.hn = 200;
        for (int i = 0; i < f.hn; i++) f.hist[i] = 14.0 - 0.03 * i + 0.2 * ((i % 7) / 7.0);
    }
    vrender_frame(&f, 820, 620);
    SDL_RenderPresent(rn);
    /* read back the presented pixels */
    SDL_Surface *shot = SDL_CreateRGBSurface(0, 820, 620, 32,
                                             0x00FF0000, 0x0000FF00,
                                             0x000000FF, 0xFF000000);
    int rc = 1;
    if (shot) {
        if (SDL_RenderReadPixels(rn, NULL, shot->format->format,
                                 shot->pixels, shot->pitch) == 0)
            rc = SDL_SaveBMP(shot, path);
        SDL_FreeSurface(shot);
    }
    SDL_DestroyRenderer(rn);
    SDL_DestroyWindow(wn);
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
int viewer_test_frame_bmp(const char *path, int mode) { (void)path; (void)mode; return 2; }
#endif /* CNCM_HAVE_SDL2 */

/* ---------------- console viewer (ANSI terminal, no threads) ----------------
 * Synchronous redraw on every push: no window system, no compositor, no
 * background thread, nothing to stall. Safe over SSH, tmux, pipes, CI.
 * On a tty the frame redraws in place; piped output is one compact line
 * per snapshot (still complete: board shown on solve completion). */
#if defined(__unix__) || defined(__APPLE__) || defined(__linux__)
#define CNCM_HAVE_UNISTD 1
#endif
#ifdef CNCM_HAVE_UNISTD
#include <unistd.h>
#include <sys/ioctl.h>
#endif

static int con_active;
static int con_tty;
static int con_lines;
static int con_width = 80;
static double con_hist[VHIST];
static int con_hn;
static int con_last_step = -1;

/* NOTE: console output is plain text by design (no ANSI colors), so it is
 * greppable, pipe-safe, and identical under NO_COLOR. Structure (grid
 * lines, !bad! markers, [tui]/VALID keywords) carries the meaning. */

static int con_term_width(void) {
#ifdef CNCM_HAVE_UNISTD
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 20)
        return w.ws_col;
#endif
    return 80;
}

int con_viewer_open(const char *title) {
    if (con_active) return 0;
    con_tty = 0;
#ifdef CNCM_HAVE_UNISTD
    con_tty = isatty(STDOUT_FILENO);
#endif
    if (con_tty) con_width = con_term_width();
    con_active = 1;
    con_lines = 0;
    con_hn = 0;
    con_last_step = -1;
    printf("CNCM console view: %s\n", title ? title : "cncm");
    fflush(stdout);
    con_lines = 1;
    return 0;
}

/* one board line into buf (<= 64 chars): cells with | separators */
static void con_board_row(const ViewSnap *s, int r, char *out) {
    int p = 0;
    for (int c = 0; c < 9; c++) {
        char ch = '.';
        int is_clue = 0;
        if (s->clues[0] && s->clues[r * 9 + c] != '.' && s->clues[r * 9 + c] != 0) {
            ch = s->clues[r * 9 + c];
            is_clue = 1;
        } else if (s->board[0] && s->board[r * 9 + c] >= '1' && s->board[r * 9 + c] <= '9') {
            ch = s->board[r * 9 + c];
        }
        int bad = 0;
        for (int k = 0; k < s->nr; k++) if (s->bad_r[k] == r) bad = 1;
        for (int k = 0; k < s->nc; k++) if (s->bad_c[k] == c) bad = 1;
        for (int k = 0; k < s->nb; k++)
            if (s->bad_b[k] == (r / 3) * 3 + c / 3) bad = 1;
        if (bad) out[p++] = '!';
        out[p++] = is_clue ? ch : (ch == '.' ? '.' : ch);
        if (bad) out[p++] = '!';
        if (c == 2 || c == 5) { out[p++] = ' '; out[p++] = '|'; out[p++] = ' '; }
        else if (c < 8) out[p++] = ' ';
    }
    out[p] = 0;
}

/* ascii sparkline of recent values into buf (width w) */
static void con_spark(const double *v, int n, int w, char *out) {
    static const char *levels = " .:-=+*#%@";
    if (n <= 0) { out[0] = 0; return; }
    double mn = v[0], mx = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] < mn) mn = v[i];
        if (v[i] > mx) mx = v[i];
    }
    if (mx - mn < 1e-9) { mn -= 0.5; mx += 0.5; }
    int m = n < w ? n : w;
    int start = n - m;
    for (int i = 0; i < m; i++) {
        double f = (v[start + i] - mn) / (mx - mn);
        int li = (int)(f * 9.0 + 0.5);
        if (li < 0) li = 0;
        if (li > 9) li = 9;
        out[i] = levels[li];
    }
    out[m] = 0;
}

static void con_emit_line(const char *s) {
    int n = (int)strlen(s);
    if (n > con_width) n = con_width;
    fwrite(s, 1, (size_t)n, stdout);
    fputc('\n', stdout);
}

void con_viewer_push(const ViewSnap *s) {
    char line[256], spark[128], brow[80];
    if (!con_active || !s) return;
    if (s->mode == 0 && s->step != con_last_step) {
        con_last_step = s->step;
        if (con_hn < VHIST) con_hist[con_hn++] = s->loss;
        else {
            memmove(con_hist, con_hist + 1, (VHIST - 1) * sizeof(double));
            con_hist[VHIST - 1] = s->loss;
        }
    }
    if (!con_tty) {
        /* piped: one compact, greppable line per snapshot */
        if (s->mode == 0)
            printf("[tui] train step %d/%d loss=%.4f acc=%.3f lr=%.2g t=%.1fs%s\n",
                   s->step, s->total, s->loss, s->acc, s->lr, s->elapsed,
                   s->ckpt ? " ckpt" : "");
        else if (s->board[0])
            printf("[tui] solve done valid=%d elapsed=%.1fs board=%s\n",
                   s->valid, s->elapsed, s->board);
        else
            printf("[tui] solve level %d/%d\n", s->step, s->total);
        fflush(stdout);
        return;
    }
    /* tty: move up over the last frame, redraw in place */
    if (con_lines > 0) printf("\x1b[%dA", con_lines);
    int n = 0;
    snprintf(line, sizeof(line), "CNCM %s [%s]  step %d/%d  t=%.1fs",
             s->mode == 0 ? "train" : "solve", s->backend[0] ? s->backend : "?",
             s->step, s->total, s->elapsed);
    con_emit_line(line);
    n = 1;
    for (int r = 0; r < 9; r++) {
        con_board_row(s, r, brow);
        if (r == 0) {
            char tmp[128];
            if (s->mode == 0)
                snprintf(tmp, sizeof(tmp), "loss %.4f  acc %.3f", s->loss, s->acc);
            else
                snprintf(tmp, sizeof(tmp), "level %d/%d  valid %s", s->step, s->total,
                         s->board[0] ? (s->valid ? "YES" : "NO") : "-");
            snprintf(line, sizeof(line), "%-21s %s", brow, tmp);
        } else if (r == 3) {
            con_spark(con_hist, con_hn, 28, spark);
            snprintf(line, sizeof(line), "%-21s loss %s", brow, con_hn > 1 ? spark : "-");
        } else if (r == 6) {
            char bars[64];
            int bi = 0;
            if (s->mode == 0) {
                for (int k = 0; k < s->nlv && k < 12 && bi < (int)sizeof(bars) - 1; k++) {
                    int fl = (int)(s->lvl_acc[k] * 9.0 + 0.5);
                    if (fl < 0) fl = 0;
                    if (fl > 9) fl = 9;
                    bars[bi++] = " .:-=+*#%@"[fl];
                }
            } else {
                for (int k = 0; k < s->nlv && bi < (int)sizeof(bars) - 1; k++)
                    bars[bi++] = (k < s->lvl_done_count) ? '#' : '.';
            }
            bars[bi] = 0;
            if (s->mode == 0)
                snprintf(line, sizeof(line), "%-21s acc [%s]%s", brow, bars,
                         s->ckpt ? " CKPT" : "");
            else
                snprintf(line, sizeof(line), "%-21s lv [%s]", brow, bars);
        } else {
            snprintf(line, sizeof(line), "%s", brow);
        }
        con_emit_line(line);
        n++;
        if (r == 2 || r == 5) {
            con_emit_line("------+-------+-------");
            n++;
        }
    }
    snprintf(line, sizeof(line), "%s", s->status);
    con_emit_line(line);
    n++;
    printf("\x1b[J");
    fflush(stdout);
    con_lines = n;
}

void con_viewer_close(void) {
    if (!con_active) return;
    con_active = 0;
    if (con_tty) printf("\n[viewer done]\n");
    fflush(stdout);
    con_lines = 0;
}
