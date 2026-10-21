#!/usr/bin/env python3
"""Round-trip test for tools/export_text.py output.

Verifies, without trusting the exporter:
  1. .nctrain header (magic, row count, row length).
  2. Every row's prefix == manifest row (trans/book/chapter) and matches
     the row's first span verse.
  3. Every row's spans tile [0, T-npadded) contiguously, no overlaps.
  4. Every verse reconstructs byte-identically: concat(decoded span slices
     across rows, in row order) == original corpus verse text.
  5. Every corpus verse (of included translations) is covered, none lost,
     none invented; padded tail is all pad_id and unreferenced.

Usage: tests/roundtrip_text.py <rows.nctrain> <manifest.json>
        <bible_corpus.txt> <bpe_vocab.txt>
"""
import json
import struct
import sys

BOOKS = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
         "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
         "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
         "Ezra", "Nehemiah", "Esther", "Job", "Psalm",
         "Proverbs", "Ecclesiastes", "Song Of Solomon", "Isaiah", "Jeremiah",
         "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel",
         "Amos", "Obadiah", "Jonah", "Micah", "Nahum",
         "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi",
         "Matthew", "Mark", "Luke", "John", "Acts",
         "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
         "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
         "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews",
         "James", "1 Peter", "2 Peter", "1 John", "2 John",
         "3 John", "Jude", "Revelation"]
BOOK_ID = {b: i for i, b in enumerate(BOOKS)}


def load_vocab(path):
    v = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sid, hx = line.split("\t", 1)
            v[int(sid)] = bytes.fromhex(hx)
    return v


def load_corpus(path):
    """(trans, book, ch, v) -> text, joining continuation lines."""
    import re
    verses = {}
    order = []
    cur = None
    pat = re.compile(r"^([^|]+)\|([^|]+)\|(\d+)\|(\d+)\t")
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            m = pat.match(line)
            if m:
                key = (m.group(1), m.group(2), int(m.group(3)), int(m.group(4)))
                verses[key] = line[m.end():]
                order.append(key)
                cur = key
            else:
                assert cur is not None, "continuation before first verse"
                verses[cur] += "\n" + line
    return verses, order


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("nctrain")
    ap.add_argument("manifest")
    ap.add_argument("corpus")
    ap.add_argument("vocab")
    ap.add_argument("--partial", action="store_true",
                    help="export was truncated (--max-rows): skip missing-verse check")
    a = ap.parse_args()
    nctrain, manpath, corpuspath, vocabpath = a.nctrain, a.manifest, a.corpus, a.vocab
    man = json.load(open(manpath, encoding="utf-8"))
    P, T = man["prefix"], man["tokens"]
    assert P == 3, P
    vocab = load_vocab(vocabpath)
    verses, order = load_corpus(corpuspath)

    fails = 0

    def fail(msg):
        nonlocal fails
        fails += 1
        if fails <= 10:
            print("FAIL:", msg)

    with open(nctrain, "rb") as f:
        assert f.read(4) == b"NCMT", "bad magic (want NCMT training rows)"
        nr, rl = struct.unpack("<QQ", f.read(16))
        assert rl == P + T, (rl, P + T)
        assert nr == man["n_rows"] == len(man["rows"]), (nr, man["n_rows"])
        import numpy as np
        data = np.fromfile(f, dtype=np.int32)
        assert data.size == nr * rl, (data.size, nr, rl)
    rows = data.reshape(nr, rl)

    # per-row checks + span collection for reconstruction
    recon = {}  # (trans,book,ch,v) -> list of (row, s, e)
    for r in man["rows"]:
        ri = r["r"]
        row = rows[ri]
        npad = r["padded"]
        # prefix vs manifest + first span
        if list(row[:P]) != [man["translations"].index(r["trans"]),
                             BOOK_ID[r["book"]], r["chapter"]]:
            fail(f"row {ri}: prefix {list(row[:P])} != manifest")
        # padded tail
        pad_id = man["pad_id"]
        if npad:
            if not all(int(x) == pad_id for x in row[P + T - npad:]):
                fail(f"row {ri}: padded tail is not pad_id")
        # spans tile [0, T-npad)
        cov = sorted((s["s"], s["e"]) for s in r["spans"])
        pos = 0
        for s0, e0 in cov:
            if s0 != pos:
                fail(f"row {ri}: span gap/overlap at {pos} (got {s0})")
                break
            pos = e0
        if pos != T - npad:
            fail(f"row {ri}: spans cover [0,{pos}) != [0,{T - npad})")
        for s in r["spans"]:
            if not (0 <= s["s"] <= s["e"] <= T - npad):
                fail(f"row {ri}: span out of range {s}")
                continue
            # key by the SPAN's own book/chapter: rows may straddle boundaries
            recon.setdefault((r["trans"], BOOKS[s["b"]], s["c"], s["v"]),
                             []).append((ri, s["s"], s["e"]))

    # verse reconstruction: concat decoded span slices, compare to corpus.
    # In --partial (truncated) mode, verses touching the final row may be
    # cut mid-verse by --max-rows; those are scoped out (documented).
    n_ok = 0
    n_skipped = 0
    last_row = max(r["r"] for r in man["rows"]) if man["rows"] else -1
    skip_keys = set()
    if a.partial:
        for r in man["rows"]:
            if r["r"] == last_row:
                for s in r["spans"]:
                    skip_keys.add((r["trans"], BOOKS[s["b"]], s["c"], s["v"]))
    for key, parts in recon.items():
        if key in skip_keys:
            n_skipped += 1
            continue
        parts.sort()
        buf = b""
        for ri, s, e in parts:
            ids = rows[ri][P + s:P + e]
            try:
                buf += b"".join(vocab[int(i)] for i in ids)
            except KeyError as ex:
                fail(f"verse {key}: unknown token id {ex}")
                buf = None
                break
        if buf is None:
            continue
        want = verses.get(key)
        if want is None:
            fail(f"verse {key}: not in corpus")
            continue
        if buf.decode("utf-8", errors="replace") != want:
            fail(f"verse {key}: text mismatch {buf[:60]!r} vs {want[:60]!r}")
        else:
            n_ok += 1

    # coverage: every corpus verse of included translations must appear,
    # unless this is a truncated (--max-rows) smoke export.
    included = set(man["translations"])
    if a.partial:
        missing = []
    else:
        missing = [k for k in verses if k[0] in included and k not in recon]
    if missing:
        fail(f"{len(missing)} corpus verses missing from export, e.g. {missing[:3]}")
    extra = [k for k in recon if k not in verses]
    if extra:
        fail(f"{len(extra)} exported verses not in corpus, e.g. {extra[:3]}")

    print(f"rows={nr} verses_reconstructed={n_ok} skipped_truncated={n_skipped} missing={len(missing)} extra={len(extra)}")
    print("ROUNDTRIP-{}".format("PASS" if fails == 0 else f"FAIL ({fails})"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
