"""
Python port of Extract_CSI.m: Nexmon CSI pcap -> complex CSI matrix.

Follows the MATLAB reader (Extract_CSI.m + readpcap.m) and the Nexmon MEX
function unpack_float.c, but runs in batch without MATLAB and writes
<stem>_csi.npz directly, with the same keys as convert_mat_to_npz.py
(preprocess_csi.py reads either).

What is ported
  * pcap records are read whole. Frames whose length is not 60 + 4*NFFT bytes
    (1084 for 80 MHz) are skipped, like the orig_len check in Extract_CSI.m.
  * header word = frame bytes 52..55 (MATLAB payload(14)), little-endian:
        seq_num  = low 16 bits        core_num = top byte
  * CSI = 4*NFFT bytes from byte 60 (MATLAB payload(16:...)), little-endian words.
  * bcm4339 / bcm43455c0: interleaved int16 real/imag.
  * bcm4358 / bcm4366c0: each 32-bit word packs a sign + mantissa for I and Q and a
    shared signed exponent. By default (decode_float) these are decoded exactly:
        value = +-mantissa * 2^exponent.
    The original C code (unpack_float, available with --matlab-compat) instead
    shifts the mantissas left by exponent + 10 + e_p bits in 32-bit integers
    (e_p = 16 for the 4358, 32 for the 4366c0). On real captures a large share of
    values needs a shift of 32 or more bits, which is undefined in C, so that
    output depends on the compiler. Wherever the C result is well defined and does
    not overflow int32, decode_float(...) * 2^(10 + e_p) equals it exactly; the
    scale is a constant, so per-subcarrier normalisation removes it.
  * subcarriers kept: 20 MHz -> 56, 80 MHz -> 242 (null/pilot bins dropped).

Added on top of the MATLAB script
  * css: the raw 16-bit core/spatial-stream field (frame bytes 54..55). The
    MATLAB core_num is only the high byte of it. Per the Nexmon README the low
    three bits give the core and the next three the spatial stream, so the
    number of distinct css values in a file tells you how many streams are mixed.
  * labels parsed from the file name (X_D_YY_ZZ, e.g. D_1_M1_P2) and a
    manifest.csv.

Keys saved: csi (N, n_sub) complex, ts (capture time in seconds from the first
kept frame, from the pcap record headers), seq_num, core_num, css, activity,
activity_id, day, device, subject.
"""
import argparse
import csv
import re
import struct
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HEADER_WORDS = 15          # words before the CSI: Eth+IP+UDP (42 B) + Nexmon header (18 B) = 60 B
HDR_WORD_IDX = 13          # MATLAB payload(14) -> 0-based word 13 -> bytes 52..55
MAX_PACKETS = 1_000_000    # same cap as Extract_CSI.m
CHUNK = 20_000             # frames decoded at a time (bounds memory)
CHIP_FORMATS = {"4358": (9, 5), "4366c0": (12, 6)}   # (nman, nexp) for unpack_float(0 | 1, ...)
INT16_CHIPS = {"4339", "43455c0"}

EXPECTED_SUBCARRIERS = 242
DUP_WARN_FRACTION = 0.05

ACTIVITIES = {
    "A": "jogging", "B": "clapping", "C": "push forward", "D": "boxing",
    "E": "writing", "F": "brushing teeth", "G": "rotating", "H": "standing",
    "I": "eating", "J": "reading a book", "K": "waving", "L": "walking",
    "M": "browsing phone", "N": "drinking", "O": "hands-up-down",
    "P": "phone call", "Q": "side bends", "R": "check wrist (watch)",
    "S": "washing hands", "T": "browsing laptop",
}
NAME_RE = re.compile(r"^([A-T])_(\d+)_(M\d+)_P(\d+)")
MANIFEST_FIELDS = [
    "file", "npz", "status", "activity", "activity_id", "day", "device",
    "subject", "packets", "frames", "subcarriers", "fs_est_hz", "seq_dup_frac",
    "css_values", "message",
]


def parse_name(stem: str):
    m = NAME_RE.match(stem)
    if m is None:
        return None
    letter, day, device, subject = m.groups()
    return {
        "activity": ACTIVITIES[letter],
        "activity_id": ord(letter) - ord("A"),
        "day": int(day),
        "device": device,
        "subject": int(subject),
    }


def non_zero_indices(bw: int) -> np.ndarray:
    """0-based version of Extract_CSI.m's non_zero ([5:32, 34:61] / [7:128, 132:251])."""
    if bw == 20:
        return np.concatenate([np.arange(4, 32), np.arange(33, 61)])
    if bw == 80:
        return np.concatenate([np.arange(6, 128), np.arange(131, 251)])
    raise ValueError("bw must be 20 or 80 (Extract_CSI.m only defines these two index sets)")


# ---------------------------------------------------------------------------
# pcap reading (readpcap.m)
# ---------------------------------------------------------------------------

def scan_pcap(path, frame_len: int, max_packets: int):
    """
    Walk the pcap records. Returns (byte buffer, offsets of the frames whose
    length equals frame_len, their capture times in seconds from the first of
    them, number of records read, Counter of record lengths).
    """
    data = Path(path).read_bytes()
    if len(data) < 24:
        raise ValueError("file too short to be a pcap")
    magic = data[:4]
    if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        endian = "<"
    elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
        endian = ">"
    elif magic == b"\x0a\x0d\x0d\x0a":
        raise ValueError("pcapng file; convert it first: editcap -F pcap in.pcapng out.pcap")
    else:
        raise ValueError("not a pcap file (bad magic number)")

    ns_per_tick = 1 if magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d") else 1000   # ns or us pcap
    rec = struct.Struct(endian + "IIII")   # ts_sec, ts_frac, incl_len, orig_len
    size = len(data)
    pos, n_packets = 24, 0
    offsets, stamps, lengths = [], [], Counter()
    while pos + 16 <= size and n_packets < max_packets:
        ts_sec, ts_frac, incl, orig = rec.unpack_from(data, pos)
        pos += 16
        if pos + incl > size:              # truncated last record
            break
        n_packets += 1
        lengths[orig] += 1
        if orig == frame_len and incl == frame_len:
            offsets.append(pos)
            stamps.append(ts_sec * 1_000_000_000 + ts_frac * ns_per_tick)   # exact integer ns
        pos += incl
    if stamps:
        ts = (np.array(stamps, dtype=np.int64) - stamps[0]) * 1e-9
    else:
        ts = np.empty(0, dtype=np.float64)
    return np.frombuffer(data, dtype=np.uint8), np.array(offsets, dtype=np.int64), ts, n_packets, lengths


# ---------------------------------------------------------------------------
# CSI unpacking (unpack_float.c)
# ---------------------------------------------------------------------------

def decode_float(words: np.ndarray, nman: int, nexp: int):
    """
    Exact decode of the Broadcom block-floating-point words (default).

    Per 32-bit word (little-endian), nexp exponent bits at the bottom, then an
    (nman-1)-bit magnitude and a sign bit for each of Q and I:
        bcm4358   (nman=9,  nexp=5):  exp[0:5]  q[5:13]  q_sign[13]  i[14:22]  i_sign[22]
        bcm4366c0 (nman=12, nexp=6):  exp[0:6]  q[6:17]  q_sign[17]  i[18:29]  i_sign[29]
    The exponent is a signed nexp-bit number shared by I and Q. The value is
        +-magnitude * 2^exponent
    with no integer shifting, so nothing wraps or depends on the compiler.
    Magnitudes have at most 11 significant bits, so complex64 stores them exactly.
    Returns (real, imag) as float64.
    """
    w = words.astype(np.uint32, copy=False)
    iq_mask = (1 << (nman - 1)) - 1
    e_p = 1 << (nexp - 1)
    sgn_i = 1 << (nexp + 2 * nman - 1)
    sgn_q = sgn_i >> nman

    he = (w & ((1 << nexp) - 1)).astype(np.int64)
    he = np.where(he >= e_p, he - 2 * e_p, he).astype(np.int32)

    def part(shift, sign_mask):
        mag = ((w >> shift) & iq_mask).astype(np.float64)
        return np.ldexp(np.where((w & sign_mask) != 0, -mag, mag), he)

    return part(nexp + nman, sgn_i), part(nexp, sgn_q)


def unpack_float(words: np.ndarray, nman: int, nexp: int):
    """
    Vectorised unpack_float_acphy() from nexmon_csi/utils/matlab/unpack_float.c,
    called the way the MEX wrapper calls it: nbits=10, autoscale=0 (--matlab-compat).

    Same bit layout as decode_float. With autoscale off, maxbit stays at -e_p,
    so every magnitude is shifted left by exponent + 10 + e_p bits in int32 and
    the result wraps like the C code.

    For shifts >= 32 the C code is undefined behaviour; x86 masks the count to
    5 bits and this function does the same. Returns (real, imag, n_ub) where
    n_ub counts the nonzero values that hit such a shift.
    """
    w = words.astype(np.uint32, copy=False)
    iq_mask = (1 << (nman - 1)) - 1
    e_p = 1 << (nexp - 1)
    sgn_i = 1 << (nexp + 2 * nman - 1)
    sgn_q = sgn_i >> nman

    vi = ((w >> (nexp + nman)) & iq_mask).astype(np.int64)
    vq = ((w >> nexp) & iq_mask).astype(np.int64)
    e = (w & ((1 << nexp) - 1)).astype(np.int64)
    e = np.where(e >= e_p, e - 2 * e_p, e) + (10 + e_p)    # shft = nbits - maxbit = 10 + e_p

    left = e >= 0
    right = (e < 0) & (e >= -nman)
    shl = np.where(left, e & 31, 0)
    shr = np.where(right, -e, 0)
    n_ub = int(np.count_nonzero(left & (e >= 32) & ((vi != 0) | (vq != 0))))

    def scale(v, negative):
        out = np.where(right, v >> shr, 0)
        out = np.where(left, v << shl, out).astype(np.int32)   # wrap to int32
        return np.where(negative, -out, out)

    return scale(vi, (w & sgn_i) != 0), scale(vq, (w & sgn_q) != 0), n_ub


def decode_frames(buf, offsets, chip: str, bw: int, dtype, matlab_compat: bool = False):
    nfft = round(bw * 3.2)
    keep = non_zero_indices(bw)
    hdr_idx = np.arange(4 * HDR_WORD_IDX, 4 * HDR_WORD_IDX + 4)
    csi_idx = np.arange(4 * HEADER_WORDS, 4 * HEADER_WORDS + 4 * nfft)

    n = len(offsets)
    csi = np.empty((n, len(keep)), dtype=dtype)
    hdr = np.empty(n, dtype=np.uint32)
    n_ub = 0
    for s in range(0, n, CHUNK):
        o = offsets[s:s + CHUNK, None]
        hdr[s:s + CHUNK] = np.ascontiguousarray(buf[o + hdr_idx]).view("<u4")[:, 0]
        body = np.ascontiguousarray(buf[o + csi_idx])
        if chip in INT16_CHIPS:
            iq = body.view("<i2").reshape(len(body), nfft, 2)
            re, im = iq[..., 0], iq[..., 1]
        else:
            nman, nexp = CHIP_FORMATS[chip]
            if matlab_compat:
                re, im, ub = unpack_float(body.view("<u4"), nman, nexp)
                n_ub += ub
            else:
                re, im = decode_float(body.view("<u4"), nman, nexp)
        full = re.astype(np.float64) + 1j * im.astype(np.float64)
        csi[s:s + CHUNK] = full[:, keep]
    return csi, hdr, n_ub


# ---------------------------------------------------------------------------
# batch pipeline
# ---------------------------------------------------------------------------

def extract_one(args):
    pcap_path, in_root, out_root, chip, bw, dtype, max_packets, matlab_compat = args
    pcap_path = Path(pcap_path)
    rel = pcap_path.relative_to(in_root)
    out_path = Path(out_root) / rel.parent / f"{pcap_path.stem}_csi.npz"
    row = {k: "" for k in MANIFEST_FIELDS}
    row.update(file=str(rel), npz=str(out_path), status="failed")

    try:
        labels = parse_name(pcap_path.stem)
        if labels is None:
            raise ValueError("filename does not match X_D_YY_ZZ (e.g. D_1_M1_P2)")
        row.update(labels)

        frame_len = 4 * HEADER_WORDS + 4 * round(bw * 3.2)
        buf, offsets, ts, n_packets, lengths = scan_pcap(pcap_path, frame_len, max_packets)
        row["packets"] = n_packets
        if len(offsets) == 0:
            common = ", ".join(f"{l} B x{c}" for l, c in lengths.most_common(3)) or "none"
            raise ValueError(
                f"no frames of the expected {frame_len} B among {n_packets} packets "
                f"(most common sizes: {common}); check --bw and the capture settings"
            )

        csi, hdr, n_ub = decode_frames(buf, offsets, chip, bw, dtype, matlab_compat)
        n_frames, n_sub = csi.shape
        seq = (hdr & 0xFFFF).astype(np.int32)
        core = (hdr >> 24).astype(np.int16)
        css = ((hdr >> 16) & 0xFFFF).astype(np.uint16)

        css_values = np.unique(css)
        dup_frac = float(np.mean(seq[1:] == seq[:-1])) if n_frames > 1 else 0.0
        notes = []
        if n_sub != EXPECTED_SUBCARRIERS and bw == 80:
            notes.append(f"{n_sub} subcarriers (expected {EXPECTED_SUBCARRIERS})")
        if len(css_values) > 1:
            notes.append(
                f"{len(css_values)} distinct core/stream values: frames from several "
                "cores/streams are interleaved in time"
            )
        elif dup_frac > DUP_WARN_FRACTION:
            notes.append(f"{dup_frac:.0%} of consecutive frames share a seq_num")
        if n_ub:
            notes.append(
                f"{n_ub} values need a 32-bit shift >= 32, which is undefined in the "
                "original C code (x86 behaviour emulated); drop --matlab-compat for exact values"
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            csi=csi,
            ts=ts,
            seq_num=seq,
            core_num=core,
            css=css,
            activity=np.array(labels["activity"]),
            activity_id=np.int16(labels["activity_id"]),
            day=np.int16(labels["day"]),
            device=np.array(labels["device"]),
            subject=np.int16(labels["subject"]),
        )
        dts = np.diff(ts)
        dts = dts[dts > 0]
        fs_est = 1.0 / float(np.median(dts)) if len(dts) else 0.0   # frames/s over all streams
        row.update(
            status="warning" if notes else "ok",
            frames=n_frames,
            subcarriers=n_sub,
            fs_est_hz=f"{fs_est:.1f}",
            seq_dup_frac=f"{dup_frac:.4f}",
            css_values="|".join(f"0x{v:04x}" for v in css_values),
            message="; ".join(notes),
        )
    except Exception as e:
        row["message"] = str(e)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/HAR-3/CSI/M1", help="folder searched recursively for .pcap files")
    ap.add_argument("--output", default="data/HAR-3/CSI_processed/M1", help="folder for the *_csi.npz files")
    ap.add_argument("--chip", choices=["4339", "4358", "43455c0", "4366c0"], default="4366c0")
    ap.add_argument("--bw", type=int, choices=[20, 80], default=80, help="bandwidth in MHz")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64",
                    help="complex64 is lossless here (values have <= 12 significant bits)")
    ap.add_argument("--max-packets", type=int, default=MAX_PACKETS,
                    help="max pcap records read per file (Extract_CSI.m uses 1,000,000)")
    ap.add_argument("--matlab-compat", action="store_true",
                    help="reproduce the original C unpack_float (32-bit shifts, undefined behaviour "
                         "for shifts >= 32) instead of the exact decode; only needed to mimic "
                         "MATLAB output")
    a = ap.parse_args()

    in_root, out_root = Path(a.input), Path(a.output)
    pcaps = sorted(in_root.rglob("*.pcap"))
    if not pcaps:
        print(f"No .pcap files found in {in_root}")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    mode = "MATLAB-compatible shifts" if a.matlab_compat else "exact decode"
    print(f"Extracting {len(pcaps)} files (chip {a.chip}, {a.bw} MHz, {mode}) with {a.workers} workers...\n")

    tasks = [(str(p), str(in_root), str(out_root), a.chip, a.bw, a.dtype, a.max_packets, a.matlab_compat)
             for p in pcaps]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures = [ex.submit(extract_one, t) for t in tasks]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            tag = {"ok": "OK     ", "warning": "WARNING", "failed": "FAILED "}[row["status"]]
            detail = (f"{row['frames']} x {row['subcarriers']} (~{row['fs_est_hz']} frames/s)"
                      if row["status"] != "failed" else "")
            print(f"[{tag}] {row['file']} {detail} {row['message']}".rstrip())

    rows.sort(key=lambda r: r["file"])
    with open(out_root / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    counts = {s: sum(r["status"] == s for r in rows) for s in ("ok", "warning", "failed")}
    print(f"\nDone: {counts['ok']} ok, {counts['warning']} with warnings, {counts['failed']} failed.")
    print(f"Manifest: {out_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()