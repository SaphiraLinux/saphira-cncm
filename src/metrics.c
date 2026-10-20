/* metrics.c — sacrebleu/rouge_score-compatible text metrics (host-side).
 * Validated against sacrebleu 2.x and rouge_score on NCM outputs.
 */
#include "ncm.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned long long key; int hc, rc; char used; } NgramTab;
typedef struct { unsigned long long key; int c; char used; } NgramTabC;
typedef struct { unsigned long long k; int hc, rc; char u; } NgramTabU;
typedef struct { unsigned long long k; int id; char u; } IdTab;

/* ---------------- 13a tokenization (sacrebleu) ---------------- */
/* Pad ASCII punctuation; pad ./, unless adjacent to a digit; pad - after digit. */

static int is_13a_punct(unsigned char c) {
    if (c >= '{' && c <= '~') return 1;   /* { | } ~ */
    if (c >= '[' && c <= '`') return 1;   /* [ \ ] ^ _ ` */
    if (c == ' ') return 1;
    if (c >= '!' && c <= '&') return 1;   /* ! " # $ % & */
    if (c >= '(' && c <= '+') return 1;   /* ( ) * + */
    if (c >= ':' && c <= '@') return 1;   /* : ; < = > ? @ */
    if (c == '/') return 1;
    return 0;
}

static char *tok_13a(const char *s) {
    size_t n = strlen(s);
    /* worst case: every byte becomes " X " (3x) */
    char *o = (char *)malloc(3 * n + 8);
    size_t w = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = s[i];
        unsigned char prev = i > 0 ? (unsigned char)s[i - 1] : 0;
        unsigned char next = i + 1 < n ? (unsigned char)s[i + 1] : 0;
        int is_pd = (c == '.' || c == ',');
        int is_dash = (c == '-');
        int do_pad = 0;
        if (is_13a_punct(c)) do_pad = 1;
        if (is_pd) {
            int pre_digit = prev >= '0' && prev <= '9';
            int fol_digit = next >= '0' && next <= '9';
            do_pad = !(pre_digit && !0) && !(fol_digit && !0) ? 0 : 0;
            /* sacrebleu: pad unless (preceded by digit [for the (pre) rule])
             * rule A: ([^0-9])([.,]) -> pad; rule B: ([.,])([^0-9]) -> pad */
            int ruleA = !pre_digit; /* (possibly absent→pad) */
            int ruleB = !fol_digit;
            do_pad = (ruleA || ruleB) ? 1 : 0;
            if (is_13a_punct(c)) do_pad = 1; /* . , already covered, keep */
        }
        if (is_dash) {
            int pre_digit = prev >= '0' && prev <= '9';
            do_pad = pre_digit ? 1 : (is_13a_punct(c) ? 1 : 0);
        }
        if (do_pad) {
            o[w++] = ' '; o[w++] = (char)c; o[w++] = ' ';
        } else {
            o[w++] = (char)c;
        }
    }
    o[w] = 0;
    return o;
}

/* split on whitespace (in place); returns token count, toks[] into buf */
static int split_ws(char *s, char **toks, int cap) {
    int n = 0;
    int in = 0;
    for (; *s && n < cap; s++) {
        if (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') {
            *s = 0; in = 0;
        } else {
            if (!in) { toks[n++] = s; in = 1; }
        }
    }
    return n;
}

/* ---------------- corpus BLEU (sacrebleu defaults) ---------------- */

typedef struct { char **toks; int n; int cap; char *buf; } TokLine;

static TokLine tok_line_13a(const char *s) {
    TokLine t = {0, 0, 0, NULL};
    t.buf = tok_13a(s);
    t.cap = 4096;
    t.toks = (char **)malloc(t.cap * sizeof(char *));
    t.n = split_ws(t.buf, t.toks, t.cap);
    return t;
}

static void tokline_free(TokLine *t) { free(t->buf); free(t->toks); }

/* n-gram precision with exponential smoothing (sacrebleu default) */
double metric_bleu_corpus(char **hyps, char **refs, int n) {
    /* count matches/clipped over corpus */
    long match[4] = {0, 0, 0, 0};
    long total[4] = {0, 0, 0, 0};
    long hyp_len = 0, ref_len = 0;
    for (int i = 0; i < n; i++) {
        TokLine h = tok_line_13a(hyps[i]);
        TokLine r = tok_line_13a(refs[i]);
        hyp_len += h.n;
        /* closest reference length (single ref): ref len */
        ref_len += r.n;
        for (int o = 0; o < 4; o++) {
            /* candidate n-grams */
            /* simple hash map via open addressing on hashed n-grams */
            int H = 8192;
            NgramTab *tab = (NgramTab *)calloc(H, sizeof(*tab));
            for (int s = 0; s + o < h.n; s++) {
                unsigned long long hh = 1469598103934665603ULL;
                for (int k = 0; k <= o; k++) {
                    const char *w = h.toks[s + k];
                    for (const char *p = w; *p; p++) { hh ^= (unsigned char)*p; hh *= 1099511628211ULL; }
                    hh ^= 0xFF; hh *= 1099511628211ULL;
                }
                unsigned long long at = hh & (H - 1);
                while (tab[at].used && tab[at].key != hh) at = (at + 1) & (H - 1);
                tab[at].key = hh; tab[at].used = 1; tab[at].hc++;
            }
            for (int s = 0; s + o < r.n; s++) {
                unsigned long long hh = 1469598103934665603ULL;
                for (int k = 0; k <= o; k++) {
                    const char *w = r.toks[s + k];
                    for (const char *p = w; *p; p++) { hh ^= (unsigned char)*p; hh *= 1099511628211ULL; }
                    hh ^= 0xFF; hh *= 1099511628211ULL;
                }
                unsigned long long at = hh & (H - 1);
                while (tab[at].used && tab[at].key != hh) at = (at + 1) & (H - 1);
                if (!tab[at].used) { tab[at].key = hh; tab[at].used = 1; tab[at].rc = 0; }
                tab[at].rc++;
            }
            for (int k = 0; k < H; k++)
                if (tab[k].used) {
                    if (tab[k].hc > tab[k].rc) match[o] += tab[k].rc;
                    else match[o] += tab[k].hc;
                    total[o] += tab[k].hc;
                }
            free(tab);
        }
        tokline_free(&h);
        tokline_free(&r);
    }
    /* exp smoothing: smooth_method='exp': for n>0 with match[n]==0 use
     * decaying average; sacrebleu: correct[n] = match[n] or previous avg. */
    double logsum = 0;
    double smooth = 1.0;
    for (int o = 0; o < 4; o++) {
        double m = (double)match[o], t = (double)total[o];
        double p;
        if (m == 0) {
            smooth *= 2.0;
            p = 1.0 / (smooth * (t > 0 ? t : 1));
        } else {
            p = m / t;
        }
        if (p <= 0) return 0.0;
        logsum += log(p);
    }
    logsum /= 4.0;
    double bp = hyp_len >= ref_len ? 1.0 : exp(1.0 - (double)ref_len / (hyp_len > 0 ? hyp_len : 1));
    return 100.0 * bp * exp(logsum);
}

/* ---------------- chrF (sacrebleu defaults: order 6, beta 2) ---------------- */



double metric_chrf(char **hyps, char **refs, int n) {
    /* corpus-level micro-average (sacrebleu): accumulate n-gram counts. */
    long tp = 0, htot = 0, rtot = 0;
    int H = 65536;
    NgramTabU *tab = (NgramTabU *)calloc(H, sizeof(*tab));
    for (int i = 0; i < n; i++) {
        size_t hl = strlen(hyps[i]), rl = strlen(refs[i]);
        char *hs = (char *)malloc(hl + 1), *rs = (char *)malloc(rl + 1);
        size_t hn = 0, rn = 0;
        for (size_t k = 0; k < hl; k++)
            if (hyps[i][k] != ' ' && hyps[i][k] != '\t' && hyps[i][k] != '\n' && hyps[i][k] != '\r') hs[hn++] = hyps[i][k];
        for (size_t k = 0; k < rl; k++)
            if (refs[i][k] != ' ' && refs[i][k] != '\t' && refs[i][k] != '\n' && refs[i][k] != '\r') rs[rn++] = refs[i][k];
        for (int o = 1; o <= 6; o++) {
            memset(tab, 0, H * sizeof(*tab));
            for (size_t s = 0; s + o <= hn; s++) {
                unsigned long long hh = 1469598103934665603ULL;
                for (int k = 0; k < o; k++) { hh ^= (unsigned char)hs[s + k]; hh *= 1099511628211ULL; }
                unsigned long long at = hh & (H - 1);
                while (tab[at].u && tab[at].k != hh) at = (at + 1) & (H - 1);
                tab[at].k = hh; tab[at].u = 1; tab[at].hc++;
                htot++;
            }
            for (size_t s = 0; s + o <= rn; s++) {
                unsigned long long hh = 1469598103934665603ULL;
                for (int k = 0; k < o; k++) { hh ^= (unsigned char)rs[s + k]; hh *= 1099511628211ULL; }
                unsigned long long at = hh & (H - 1);
                int guard = 0;
                while (guard++ < H) {
                    unsigned long long q = (at + guard - 1) & (H - 1);
                    if (!tab[q].u) break;
                    if (tab[q].k == hh && tab[q].hc > 0) { tab[q].hc--; tp++; break; }
                }
                rtot++;
            }
        }
        free(hs); free(rs);
    }
    free(tab);
    double chrP = htot ? (double)tp / htot : 0;
    double chrR = rtot ? (double)tp / rtot : 0;
    double beta = 2.0;
    if (chrP + chrR == 0) return 0;
    return 100.0 * (1 + beta * beta) * chrP * chrR / (beta * beta * chrP + chrR);
}

/* ---------------- Porter stemmer (for ROUGE) ---------------- */

static int p_is_vowel(const char *w, int i) {
    char c = w[i];
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') return 1;
    if (c == 'y' && i > 0) return !p_is_vowel(w, i - 1);
    return 0;
}

/* measure m = VC count */
static int p_measure(const char *w, int n) {
    int m = 0, i = 0;
    while (i < n && !p_is_vowel(w, i)) i++;
    while (i < n) {
        while (i < n && p_is_vowel(w, i)) i++;
        if (i >= n) break;
        m++;
        while (i < n && !p_is_vowel(w, i)) i++;
    }
    return m;
}

static int p_has_vowel(const char *w, int n) {
    for (int i = 0; i < n; i++) if (p_is_vowel(w, i)) return 1;
    return 0;
}

static int p_double_cons(const char *w, int n) {
    if (n < 2 || w[n - 1] != w[n - 2]) return 0;
    return !p_is_vowel(w, n - 1);
}

static int p_cvc(const char *w, int n) {
    if (n < 3) return 0;
    if (p_is_vowel(w, n - 1) || !p_is_vowel(w, n - 2) || p_is_vowel(w, n - 3)) return 0;
    char c = w[n - 1];
    return c != 'w' && c != 'x' && c != 'y';
}

/* stem in place; returns new length. Standard Porter (1980). */
static int porter_stem(char *w) {
    int n = strlen(w);
    if (n <= 2) return n;
    /* step 1a */
    if (n >= 4 && !strcmp(w + n - 4, "sses")) n -= 2;
    else if (n >= 3 && !strcmp(w + n - 3, "ies")) { strcpy(w + n - 3, "i"); n -= 2; }
    else if (n >= 2 && w[n - 1] == 's' && w[n - 2] != 's') n--;
    /* step 1b */
    int b = 0;
    if (n >= 3 && !strcmp(w + n - 3, "eed")) { if (p_measure(w, n - 3) > 0) n--; }
    else if (n >= 2 && !strcmp(w + n - 2, "ed")) { if (p_has_vowel(w, n - 2)) { n -= 2; b = 1; } }
    else if (n >= 3 && !strcmp(w + n - 3, "ing")) { if (p_has_vowel(w, n - 3)) { n -= 3; b = 1; } }
    if (b) {
        if (n >= 2 && (!strcmp(w + n - 2, "at") || !strcmp(w + n - 2, "bl") || !strcmp(w + n - 2, "iz"))) { w[n++] = 'e'; }
        else if (p_double_cons(w, n) && w[n - 1] != 'l' && w[n - 1] != 's' && w[n - 1] != 'z') n--;
        else if (p_measure(w, n) == 1 && p_cvc(w, n)) { w[n++] = 'e'; }
    }
    /* step 1c */
    if (n > 1 && w[n - 1] == 'y' && !p_is_vowel(w, n - 2)) w[n - 1] = 'i';
    /* step 2 */
    struct { const char *a, *re; } s2[] = {
        {"ational", "ate"}, {"tional", "tion"}, {"enci", "ence"}, {"anci", "ance"},
        {"izer", "ize"}, {"abli", "able"}, {"alli", "al"}, {"entli", "ent"},
        {"eli", "e"}, {"ousli", "ous"}, {"ization", "ize"}, {"ation", "ate"},
        {"ator", "ate"}, {"alism", "al"}, {"iveness", "ive"}, {"fulness", "ful"},
        {"ousness", "ous"}, {"aliti", "al"}, {"iviti", "ive"}, {"biliti", "ble"}, {NULL, NULL}};
    for (int i = 0; s2[i].a; i++) {
        int la = strlen(s2[i].a);
        if (n > la && !strcmp(w + n - la, s2[i].a) && p_measure(w, n - la) > 0) {
            strcpy(w + n - la, s2[i].re);
            n = strlen(w);
            break;
        }
    }
    /* step 3 */
    struct { const char *a, *re; } s3[] = {
        {"icate", "ic"}, {"ative", ""}, {"alize", "al"}, {"iciti", "ic"},
        {"ical", "ic"}, {"ful", ""}, {"ness", ""}, {NULL, NULL}};
    for (int i = 0; s3[i].a; i++) {
        int la = strlen(s3[i].a);
        if (n > la && !strcmp(w + n - la, s3[i].a) && p_measure(w, n - la) > 0) {
            strcpy(w + n - la, s3[i].re);
            n = strlen(w);
            break;
        }
    }
    /* step 4 */
    const char *s4[] = {"al", "ance", "ence", "er", "ic", "able", "ible", "ant",
                        "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
                        "ous", "ive", "ize", NULL};
    for (int i = 0; s4[i]; i++) {
        int la = strlen(s4[i]);
        if (n > la + 1 && !strcmp(w + n - la, s4[i]) && p_measure(w, n - la) > 1) {
            if (!strcmp(s4[i], "ion")) {
                if (w[n - la - 1] != 's' && w[n - la - 1] != 't') continue;
            }
            n -= la;
            w[n] = 0;
            break;
        }
    }
    /* step 5a */
    if (w[n - 1] == 'e') {
        int m = p_measure(w, n - 1);
        if (m > 1 || (m == 1 && !p_cvc(w, n - 1))) n--;
        w[n] = 0;
    }
    /* step 5b */
    if (n > 1 && w[n - 1] == 'l' && p_double_cons(w, n) && p_measure(w, n) > 1) n--;
    w[n] = 0;
    return n;
}

/* rouge tokenizer: lowercase alphanumeric runs */
static int rouge_tok(const char *s, char *buf, char **toks, int cap) {
    int n = 0, bi = 0, in = 0;
    char cur[256];
    int ci = 0;
    for (const char *p = s; ; p++) {
        char c = *p ? tolower((unsigned char)*p) : 0;
        int al = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
        if (al && ci < 250) { cur[ci++] = c; in = 1; }
        else {
            if (in && ci > 0) {
                cur[ci] = 0;
                ci = porter_stem(cur);
                if (n < cap && bi + ci + 1 < 65536) {
                    memcpy(buf + bi, cur, ci + 1);
                    toks[n++] = buf + bi;
                    bi += ci + 1;
                }
                ci = 0;
            }
            in = 0;
            if (!c) break;
        }
    }
    return n;
}

void metric_rouge(char **hyps, char **refs, int n, double *r1, double *r2, double *rl) {
    static char buf[2][65536];
    static char *tt[2][4096];
    double s1 = 0, s2 = 0, sl = 0;
    for (int i = 0; i < n; i++) {
        int nh = rouge_tok(hyps[i], buf[0], tt[0], 4096);
        int nr = rouge_tok(refs[i], buf[1], tt[1], 4096);
        /* unigram F */
        {
            /* multiset overlap via hash */
            int H = 4096;
            NgramTabU *tab = (NgramTabU *)calloc(H, sizeof(*tab));
            for (int k = 0; k < nh; k++) {
                unsigned long long hh = 1469598103934665603ULL;
                for (char *p = tt[0][k]; *p; p++) { hh ^= *p; hh *= 1099511628211ULL; }
                unsigned long long at = hh & (H - 1);
                while (tab[at].u && tab[at].k != hh) at = (at + 1) & (H - 1);
                tab[at].k = hh; tab[at].u = 1; tab[at].hc++;
            }
            for (int k = 0; k < nr; k++) {
                unsigned long long hh = 1469598103934665603ULL;
                for (char *p = tt[1][k]; *p; p++) { hh ^= *p; hh *= 1099511628211ULL; }
                unsigned long long at = hh & (H - 1);
                while (tab[at].u && tab[at].k != hh) at = (at + 1) & (H - 1);
                if (!tab[at].u) { tab[at].k = hh; tab[at].u = 1; }
                tab[at].rc++;
            }
            long ov = 0;
            for (int k = 0; k < H; k++)
                if (tab[k].u) ov += tab[k].hc < tab[k].rc ? tab[k].hc : tab[k].rc;
            free(tab);
            double p = nh ? (double)ov / nh : 0, rr = nr ? (double)ov / nr : 0;
            s1 += (p + rr) > 0 ? 2 * p * rr / (p + rr) : 0;
        }
        /* bigram F */
        {
            int H = 4096;
            NgramTabU *tab = (NgramTabU *)calloc(H, sizeof(*tab));
            for (int k = 0; k + 1 < nh; k++) {
                unsigned long long hh = 1469598103934665603ULL;
                for (int j = 0; j < 2; j++) {
                    for (char *p = tt[0][k + j]; *p; p++) { hh ^= *p; hh *= 1099511628211ULL; }
                    hh ^= 0xFF; hh *= 1099511628211ULL;
                }
                unsigned long long at = hh & (H - 1);
                while (tab[at].u && tab[at].k != hh) at = (at + 1) & (H - 1);
                tab[at].k = hh; tab[at].u = 1; tab[at].hc++;
            }
            for (int k = 0; k + 1 < nr; k++) {
                unsigned long long hh = 1469598103934665603ULL;
                for (int j = 0; j < 2; j++) {
                    for (char *p = tt[1][k + j]; *p; p++) { hh ^= *p; hh *= 1099511628211ULL; }
                    hh ^= 0xFF; hh *= 1099511628211ULL;
                }
                unsigned long long at = hh & (H - 1);
                while (tab[at].u && tab[at].k != hh) at = (at + 1) & (H - 1);
                if (!tab[at].u) { tab[at].k = hh; tab[at].u = 1; }
                tab[at].rc++;
            }
            long ov = 0, th = nh > 0 ? nh - 1 : 0, tr = nr > 0 ? nr - 1 : 0;
            for (int k = 0; k < H; k++)
                if (tab[k].u) ov += tab[k].hc < tab[k].rc ? tab[k].hc : tab[k].rc;
            free(tab);
            double p = th ? (double)ov / th : 0, rr = tr ? (double)ov / tr : 0;
            s2 += (p + rr) > 0 ? 2 * p * rr / (p + rr) : 0;
        }
        /* LCS F (Rouge-L): DP over token ids */
        {
            /* map tokens to small ints for DP */
            int *a = (int *)malloc(nh * sizeof(int)), *b = (int *)malloc(nr * sizeof(int));
            /* reuse hash for identity */
            int H = 4096;
            IdTab *tab = (IdTab *)calloc(H, sizeof(*tab));
            int next = 1;
            for (int k = 0; k < nh; k++) {
                unsigned long long hh = 1469598103934665603ULL;
                for (char *p = tt[0][k]; *p; p++) { hh ^= *p; hh *= 1099511628211ULL; }
                unsigned long long at = hh & (H - 1);
                while (tab[at].u && tab[at].k != hh) at = (at + 1) & (H - 1);
                if (!tab[at].u) { tab[at].k = hh; tab[at].u = 1; tab[at].id = next++; }
                a[k] = tab[at].id;
            }
            for (int k = 0; k < nr; k++) {
                unsigned long long hh = 1469598103934665603ULL;
                for (char *p = tt[1][k]; *p; p++) { hh ^= *p; hh *= 1099511628211ULL; }
                unsigned long long at = hh & (H - 1);
                while (tab[at].u && tab[at].k != hh) at = (at + 1) & (H - 1);
                if (!tab[at].u) { tab[at].k = hh; tab[at].u = 1; tab[at].id = next++; }
                b[k] = tab[at].id;
            }
            free(tab);
            /* full DP LCS (skipped for pathological sizes) */
            if ((int64_t)(nh + 1) * (nr + 1) > (1 << 24)) {
                sl += 0; /* degenerate fallback */
            } else {
                unsigned short *dp = (unsigned short *)calloc((nh + 1) * (nr + 1), 2);
                for (int x = 1; x <= nh; x++)
                    for (int y = 1; y <= nr; y++)
                        dp[x * (nr + 1) + y] = a[x - 1] == b[y - 1]
                            ? dp[(x - 1) * (nr + 1) + y - 1] + 1
                            : (dp[(x - 1) * (nr + 1) + y] > dp[x * (nr + 1) + y - 1]
                               ? dp[(x - 1) * (nr + 1) + y] : dp[x * (nr + 1) + y - 1]);
                int lcs = dp[nh * (nr + 1) + nr];
                free(dp);
                double p = nh ? (double)lcs / nh : 0, rr = nr ? (double)lcs / nr : 0;
                sl += (p + rr) > 0 ? 2 * p * rr / (p + rr) : 0;
            }
            free(a); free(b);
            (void)N;
        }
    }
    *r1 = 100 * s1 / (n ? n : 1);
    *r2 = 100 * s2 / (n ? n : 1);
    *rl = 100 * sl / (n ? n : 1);
}

double gpt2_genppl(const char *a, const char *b, char **h, int n, int max_len) {
    (void)a; (void)b; (void)h; (void)n; (void)max_len;
    return -1;
}
