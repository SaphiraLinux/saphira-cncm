/* sudoku_size.h — grid-size parameters for Sudoku support.
 *
 * 9x9 behaviour is the frozen reference: N=9, box=3, P=91, ids
 * 0=empty, 1-9=digits, 10='|', 11=wrap. The formulas below reproduce
 * those exact numbers for N=9 and extend to 16 (box 4, P=273,
 * symbols 1-9,A-G) and 25 (box 5, P=651, symbols 1-9,A-P).
 *
 * Header-only, dependency-free; safe to include from C files.
 */
#ifndef SUDOKU_SIZE_H
#define SUDOKU_SIZE_H

#define SUDOKU_MAX_N 25
#define SUDOKU_MAX_CELLS (SUDOKU_MAX_N * SUDOKU_MAX_N)       /* 625 */
#define SUDOKU_MAX_TEXT (SUDOKU_MAX_CELLS + SUDOKU_MAX_N)    /* cells + seps */
#define SUDOKU_MAX_PREFIX (SUDOKU_MAX_TEXT + 2)              /* + wrap pair */

typedef struct {
    int n;          /* grid width (9, 16, 25) */
    int box;        /* subgrid width (3, 4, 5) */
    int ncells;     /* n*n */
    int nsep;       /* row separators in text form (n-1) */
    int text_len;   /* ncells + nsep (board text without wrap) */
    int prefix_len; /* text_len + 2 (wrap pair) */
    int sep_id;     /* row-separator token id (n+1; 10 for 9x9) */
    int wrap_id;    /* frame token id (n+2; 11 for 9x9) */
} SudokuSize;

/* Fill s for grid width n. Returns 0 ok, nonzero on unsupported n. */
static inline int sudoku_size_for(int n, SudokuSize *s) {
    int box = 0;
    if (n == 9) box = 3;
    else if (n == 16) box = 4;
    else if (n == 25) box = 5;
    else return -1;
    s->n = n;
    s->box = box;
    s->ncells = n * n;
    s->nsep = n - 1;
    s->text_len = n * n + (n - 1);
    s->prefix_len = n * n + (n - 1) + 2;
    s->sep_id = n + 1;
    s->wrap_id = n + 2;
    return 0;
}

/* 9x9 reference parameters. */
static inline SudokuSize sudoku_size_default(void) {
    SudokuSize s;
    sudoku_size_for(9, &s);
    return s;
}

/* Guess grid size from a prefix length: P = n^2 + n + 1.
 * Returns 0 with *out filled, nonzero when P matches no known size. */
static inline int sudoku_size_from_prefix(int P, SudokuSize *out) {
    int sizes[3] = {9, 16, 25};
    for (int i = 0; i < 3; i++) {
        SudokuSize s = sudoku_size_default();
        sudoku_size_for(sizes[i], &s);
        if (s.prefix_len == P) {
            if (out) *out = s;
            return 0;
        }
    }
    return -1;
}

/* Cell value (1..n) to display symbol: 1-9 -> '1'-'9', 10.. -> 'A'... */
static inline char sudoku_val_char(int v) {
    if (v >= 1 && v <= 9) return (char)('0' + v);
    if (v >= 10 && v <= 35) return (char)('A' + (v - 10));
    return 0;
}

/* Display symbol back to value. Returns -1 when invalid or out of range
 * for grid width n ('.' and 0 are handled by callers, not here). */
static inline int sudoku_char_val(char c, int n) {
    int v = -1;
    if (c >= '1' && c <= '9') v = c - '0';
    else if (c >= 'A' && c <= 'Z') v = 10 + (c - 'A');
    if (v < 1 || v > n) return -1;
    return v;
}

/* Token id for a cell value v (0=empty, 1..n) under grid width n.
 * For n=9 this yields the historical mapping (0..9, sep 10, wrap 11). */
static inline int sudoku_val_id(int v, int n) {
    if (v < 0 || v > n) return -1;
    return v;
}

#endif /* SUDOKU_SIZE_H */
