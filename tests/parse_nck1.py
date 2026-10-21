#!/usr/bin/env python3
"""Independent NCK1 reader (no C code shared): validates the documented
format, prints summaries, and diffs two checkpoints numerically.
Usage:
  parse_nck1.py info FILE
  parse_nck1.py diff A B   -> max abs diff over weights and moments
"""
import struct
import sys

MAGIC = b"NCK1"
FNV_BASIS = 14695981039346656037
FNV_PRIME = 1099511628211


def fnv1a(data, h=FNV_BASIS):
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


class Cur:
    def __init__(self, buf):
        self.b = buf
        self.o = 0

    def take(self, n):
        if self.o + n > len(self.b):
            raise ValueError("truncated at offset %d (need %d)" % (self.o, n))
        r = self.b[self.o:self.o + n]
        self.o += n
        return r

    def u8(self):
        return self.take(1)[0]

    def u16(self):
        return struct.unpack("<H", self.take(2))[0]

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def i64(self):
        return struct.unpack("<q", self.take(8))[0]

    def i32(self):
        return struct.unpack("<i", self.take(4))[0]

    def f64(self):
        return struct.unpack("<d", self.take(8))[0]

    def s(self):
        n = self.u16()
        return self.take(n).decode("utf-8", "replace")

    def f32arr(self, nbytes):
        import array
        raw = self.take(nbytes)
        a = array.array("f")
        a.frombytes(raw)
        if sys.byteorder != "little":
            a.byteswap()
        return a


def parse(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 12:
        raise ValueError("file too small")
    body, stored = data[:-8], struct.unpack("<Q", data[-8:])[0]
    calc = fnv1a(body)
    c = Cur(body)
    if c.take(4) != MAGIC:
        raise ValueError("bad magic")
    ver = c.u32()
    if ver != 1:
        raise ValueError("unsupported version %d" % ver)
    hdr = {}
    hdr["global_step"] = c.i64()
    hdr["batch_rng"] = c.u64()
    hdr["base_lr"], hdr["warmup"], hdr["total"] = c.f64(), c.f64(), c.f64()
    hdr["B"] = c.i32()
    hdr["accum"] = c.i32()
    hdr["n_rows"], hdr["row_len"] = c.i64(), c.i32()
    hdr["data_hash"] = c.u64()
    hdr["data_path"] = c.s()
    hdr["conf_hash"] = c.u64()
    hdr["tokenizer_id"] = c.s()
    hdr["cfg_json"] = c.s()
    n = c.u32()
    tensors = []
    for _ in range(n):
        nl = c.u8()
        name = c.take(nl).decode("utf-8", "replace")
        nd = c.u8()
        shape = [c.u32() for _ in range(nd)]
        nb = c.u64()
        arr = c.f32arr(nb)
        tensors.append((name, shape, arr))
    if c.take(4) != b"ADAM":
        raise ValueError("bad optimizer section")
    opt = {}
    opt["b1"], opt["b2"], opt["eps"], opt["wd"] = (c.f64(), c.f64(), c.f64(), c.f64())
    opt["n"] = c.u32()
    opt["step"] = c.i64()
    params = []
    for _ in range(opt["n"]):
        idx = c.u32()
        lrm = c.f64()
        mnb = c.u64()
        m = c.f32arr(mnb)
        vnb = c.u64()
        v = c.f32arr(vnb)
        params.append((idx, lrm, m, v))
    if c.o != len(body):
        raise ValueError("trailing garbage (%d bytes)" % (len(body) - c.o))
    return {"hdr": hdr, "tensors": tensors, "opt": opt,
            "params": params, "checksum_ok": calc == stored,
            "size": len(data)}


def cmd_info(path):
    p = parse(path)
    h, o = p["hdr"], p["opt"]
    print("file: %s (%d bytes)" % (path, p["size"]))
    print("checksum: %s" % ("OK" if p["checksum_ok"] else "MISMATCH"))
    print("global_step=%d opt_step=%d accum=%d" % (h["global_step"], o["step"], h["accum"]))
    print("schedule: lr=%.6g warmup=%.6g total=%.6g" % (h["base_lr"], h["warmup"], h["total"]))
    print("batch: B=%d rng=0x%016x rows=%d row_len=%d" % (h["B"], h["batch_rng"], h["n_rows"], h["row_len"]))
    print("tensors=%d params=%d" % (len(p["tensors"]), o["n"]))


def maxdiff(a, b):
    m = 0.0
    for x, y in zip(a, b):
        d = abs(float(x) - float(y))
        if d > m:
            m = d
    return m


def cmd_diff(pa, pb):
    a, b = parse(pa), parse(pb)
    for p in (a, b):
        if not p["checksum_ok"]:
            print("FAIL: checksum mismatch in input")
            return 2
    ha, hb, oa, ob = a["hdr"], b["hdr"], a["opt"], b["opt"]
    rc = 0
    for k in ("global_step", "batch_rng", "base_lr", "warmup", "total", "B",
              "accum", "n_rows", "row_len", "data_hash", "data_path",
              "conf_hash", "tokenizer_id", "cfg_json"):
        if ha[k] != hb[k]:
            print("SCALAR-DIFF %s: %r vs %r" % (k, ha[k], hb[k]))
            rc = 1
    if oa["step"] != ob["step"]:
        print("SCALAR-DIFF opt_step: %r vs %r" % (oa["step"], ob["step"]))
        rc = 1
    if len(a["tensors"]) != len(b["tensors"]) or oa["n"] != ob["n"]:
        print("FAIL: tensor/param count differs")
        return 2
    wmax, mmax, vmax = 0.0, 0.0, 0.0
    for (na, sa, aa), (nb, sb, ab) in zip(a["tensors"], b["tensors"]):
        if na != nb or sa != sb or len(aa) != len(ab):
            print("FAIL: tensor layout differs at %s" % na)
            return 2
        wmax = max(wmax, maxdiff(aa, ab))
    for (ia, la, ma, va), (ib, lb, mb, vb) in zip(a["params"], b["params"]):
        if ia != ib or la != lb or len(ma) != len(mb) or len(va) != len(vb):
            print("FAIL: param map differs")
            return 2
        mmax = max(mmax, maxdiff(ma, mb))
        vmax = max(vmax, maxdiff(va, vb))
    print("max-abs-diff weights=%.3g moments-m=%.3g moments-v=%.3g" % (wmax, mmax, vmax))
    return rc


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "info":
        cmd_info(sys.argv[2])
    elif sys.argv[1] == "diff" and len(sys.argv) == 4:
        sys.exit(cmd_diff(sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)
        sys.exit(2)
