"""
Preprocess extracted CSI (<stem>_csi.npz from extract_csi.py) into labelled,
fixed-length windows for a CNN, one <stem>_preprocessed.npz per recording.

Per recording (and per core/stream, if a file mixes several):
  1. drop frames with almost no energy (narrow-band control frames that leave
     most subcarriers empty) and frames that are not finite
  2. split into continuous segments wherever the capture has a gap, so no
     window ever spans a hole in the recording
  3. amplitude = |CSI|; phase = unwrapped phase minus its per-frame straight-line
     fit over the true subcarrier indices (removes the random per-packet offset
     and slope), computed vectorised
  4. Hampel filter along time (rolling median, replaces single-frame spikes
     without clipping real motion), then a zero-phase Butterworth low-pass
  5. resample to a uniform --target-fs grid using the capture timestamps
     (the filters and windows assume uniform sampling; real captures are not)
  6. centre each subcarrier per recording: amplitude as amp / median - 1,
     phase minus its median. Motion intensity is kept, so a still window stays
     flat and a jogging window does not
  7. cut windows of --window-sec seconds every --stride-sec seconds

Output layout:   windows  (n_windows, 2, T, n_subcarriers)   channel 0 = amplitude,
                 channel 1 = phase, T = window_sec * target_fs   (2D CNN friendly)
Per window:      t_start, stream (the css value), segment
Per recording:   subject, activity_id, day, activity, device, fs, fs_in, ...

Normalisation is deliberately NOT global. For leave-one-subject-out, compute
the per-channel mean/std from the TRAINING subjects' windows only, in your
training code, and apply it to the validation and test subjects.
load_windows() and loso_folds() below do the loading and the splitting:

    data = load_windows("data/HAR-3/CSI_preprocessed/M1")
    for fold in loso_folds(data["subject"], n_val=1):
        test = np.isin(data["subject"], fold["test"])
        val = np.isin(data["subject"], fold["val"])
        train = np.isin(data["subject"], fold["train"])
        mean = data["X"][train].mean(axis=(0, 2, 3), keepdims=True)
        std = data["X"][train].std(axis=(0, 2, 3), keepdims=True) + 1e-8
        ...   # fit on train, early-stop on val, report on test

Windows from one recording overlap, so never split windows of the same
recording across train/test: always split by subject, as above.

Memory: a long recording is held in memory (about 1.5 GB for 400k frames x 242
subcarriers), times --workers. Lower --workers if you run out of RAM.
"""
import argparse
import csv
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

NAME_RE = re.compile(r"^([A-T])_(\d+)_(M\d+)_P(\d+)")
OUT_SUFFIX = "_preprocessed.npz"

DEFAULTS = dict(
    target_fs=100.0,        # Hz, rate of the output windows
    assumed_fs=100.0,       # Hz, only used if a file has no timestamps
    cutoff=10.0,            # Hz, low-pass cutoff
    window_sec=2.0,
    stride_sec=1.0,
    hampel_window=7,        # frames, odd; 0 disables the Hampel filter
    hampel_sigma=3.0,
    min_energy_ratio=0.1,   # drop frames below this fraction of the median frame energy; 0 disables
    gap_sec=0.5,            # minimum gap that splits a recording (at least 10x the median frame interval)
    streams="all",          # "all": every css value separately, "main": only the most frequent one
    dtype="float32",
    compress=False,
)

MANIFEST_FIELDS = [
    "file", "out", "status", "subject", "activity", "day", "frames_in",
    "frames_dropped", "fs_in_hz", "streams", "segments", "windows", "message",
]


# ---------------------------------------------------------------------------
# signal processing
# ---------------------------------------------------------------------------

def subcarrier_index(n_sub: int) -> np.ndarray:
    """True OFDM subcarrier indices of the kept columns (same sets as Extract_CSI.m, 0-based)."""
    if n_sub == 242:    # 80 MHz
        return np.concatenate([np.arange(6, 128), np.arange(131, 251)])
    if n_sub == 56:     # 20 MHz
        return np.concatenate([np.arange(4, 32), np.arange(33, 61)])
    return np.arange(n_sub)


def sanitize_phase(csi: np.ndarray, idx: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """
    Unwrap along subcarriers, then remove each frame's least-squares line
    (slope + intercept) over the true subcarrier indices. Vectorised.
    """
    x = idx.astype(np.float64)
    x -= x.mean()
    sxx = float((x ** 2).sum())
    out = np.empty(csi.shape, dtype=np.float32)
    for s in range(0, len(csi), chunk):
        ph = np.unwrap(np.angle(csi[s:s + chunk]), axis=1).astype(np.float64)
        slope = (ph * x).sum(axis=1) / sxx
        out[s:s + chunk] = ph - ph.mean(axis=1, keepdims=True) - slope[:, None] * x
    return out


def hampel(x: np.ndarray, k: int, sigma: float) -> np.ndarray:
    """Replace samples further than sigma*1.4826*MAD from the rolling median (along axis 0)."""
    med = median_filter(x, size=(k, 1), mode="nearest")
    dev = np.abs(x - med)
    mad = median_filter(dev, size=(k, 1), mode="nearest")
    thr = sigma * 1.4826 * np.maximum(mad, 0.005 * np.abs(med))   # floor: flat stretches must not trigger
    return np.where(dev > thr, med, x)


def apply_chunked(fn, x: np.ndarray, chunk: int, overlap: int) -> np.ndarray:
    """Apply fn along axis 0 in chunks with overlap so edge effects stay out of the kept part."""
    n = len(x)
    if n <= chunk + 2 * overlap:
        return fn(x).astype(x.dtype, copy=False)
    out = np.empty_like(x)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        a, b = max(0, s - overlap), min(n, e + overlap)
        out[s:e] = fn(x[a:b])[s - a:s - a + (e - s)]
    return out


def clean_series(x: np.ndarray, fs_in: float, cfg: dict) -> np.ndarray:
    """Hampel + zero-phase Butterworth low-pass along time. x: (n, n_sub) float32."""
    cutoff = min(cfg["cutoff"], 0.4 * fs_in)
    sos = butter(4, cutoff / (0.5 * fs_in), btype="low", output="sos")
    min_len = 3 * (2 * len(sos) + 1) + 1
    k = int(cfg["hampel_window"])
    if k % 2 == 0 and k > 0:
        k += 1

    def fn(a):
        if k > 0:
            a = hampel(a, k, cfg["hampel_sigma"])
        return sosfiltfilt(sos, a, axis=0) if len(a) >= min_len else a

    return apply_chunked(fn, x, chunk=40_000, overlap=max(200, int(2 * fs_in)))


def resample_uniform(x: np.ndarray, t: np.ndarray, fs: float):
    """Linear interpolation of every column onto a uniform grid. Returns (values, grid times)."""
    n_out = int(np.floor((t[-1] - t[0]) * fs)) + 1
    grid = t[0] + np.arange(n_out) / fs
    out = np.empty((n_out, x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(grid, t, x[:, c])
    return out, grid


def make_windows(amp, ph, t, T, stride, dtype):
    n = len(amp)
    starts = np.arange(0, n - T + 1, stride)
    if len(starts) == 0:
        return None, starts
    idx = starts[:, None] + np.arange(T)
    w = np.stack([amp[idx], ph[idx]], axis=1).astype(dtype, copy=False)
    return w, t[starts]


def process_stream(csi, t, idx, cfg):
    """
    One core/stream of one recording -> dict with windows (n, 2, T, F), t_start,
    segment and bookkeeping. csi: (n, F) complex, t: (n,) seconds.
    """
    n_in = len(csi)
    amp = np.abs(csi).astype(np.float32)
    energy = amp.mean(axis=1)
    keep = np.isfinite(energy)
    if cfg["min_energy_ratio"] > 0 and keep.any():
        keep &= energy >= cfg["min_energy_ratio"] * np.median(energy[keep])
    if not keep.all():
        csi, t, amp = csi[keep], t[keep], amp[keep]
    n_dropped = n_in - len(csi)
    if len(csi) < 2:
        raise ValueError("fewer than 2 usable frames")

    if np.any(np.diff(t) < 0):                       # timestamps out of order: sort
        order = np.argsort(t, kind="stable")
        csi, t, amp = csi[order], t[order], amp[order]
    ph = sanitize_phase(csi, idx)
    del csi

    dt = np.diff(t)
    dt_pos = dt[dt > 0]
    if len(dt_pos) == 0:
        raise ValueError("all timestamps identical")
    fs_in = 1.0 / float(np.median(dt_pos))
    gap_thr = max(10.0 / fs_in, cfg["gap_sec"])
    bounds = [0, *(np.flatnonzero(dt > gap_thr) + 1), len(t)]

    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 10:
            continue
        a_c = clean_series(amp[a:b], fs_in, cfg)
        p_c = clean_series(ph[a:b], fs_in, cfg)
        a_u, grid = resample_uniform(a_c, t[a:b], cfg["target_fs"])
        p_u, _ = resample_uniform(p_c, t[a:b], cfg["target_fs"])
        segs.append((a_u, p_u, grid))
    del amp, ph
    if not segs:
        raise ValueError("no continuous segment long enough")

    # centre per subcarrier over the whole recording (keeps motion intensity)
    amp_med = np.median(np.concatenate([s[0] for s in segs]), axis=0)
    amp_med = np.maximum(amp_med, 1e-3 * float(amp_med.mean()))
    ph_med = np.median(np.concatenate([s[1] for s in segs]), axis=0)

    T = int(round(cfg["window_sec"] * cfg["target_fs"]))
    stride = max(1, int(round(cfg["stride_sec"] * cfg["target_fs"])))
    wins, starts, seg_ids = [], [], []
    for i, (a_u, p_u, grid) in enumerate(segs):
        w, ts_ = make_windows(a_u / amp_med - 1.0, p_u - ph_med, grid, T, stride, cfg["dtype"])
        if w is not None:
            wins.append(w)
            starts.append(ts_)
            seg_ids.append(np.full(len(ts_), i, dtype=np.int32))
    return dict(
        windows=np.concatenate(wins) if wins else None,
        t_start=np.concatenate(starts).astype(np.float32) if wins else None,
        segment=np.concatenate(seg_ids) if wins else None,
        n_in=n_in, n_dropped=n_dropped, fs_in=fs_in, n_segments=len(segs),
    )


# ---------------------------------------------------------------------------
# per-file worker and batch runner
# ---------------------------------------------------------------------------

def read_labels(d, stem: str) -> dict:
    if "subject" in d.files:
        return dict(subject=int(d["subject"]), activity_id=int(d["activity_id"]), day=int(d["day"]),
                    activity=str(d["activity"]), device=str(d["device"]))
    m = NAME_RE.match(stem)
    if m is None:
        raise ValueError("no labels in the file and the name does not match X_D_YY_ZZ")
    letter, day, device, subject = m.groups()
    return dict(subject=int(subject), activity_id=ord(letter) - ord("A"), day=int(day),
                activity=letter, device=device)


def process_file(args):
    path, in_root, out_root, cfg = args
    path = Path(path)
    rel = path.relative_to(in_root)
    stem = path.stem[:-4] if path.stem.endswith("_csi") else path.stem
    out_path = Path(out_root) / rel.parent / f"{stem}{OUT_SUFFIX}"
    row = {k: "" for k in MANIFEST_FIELDS}
    row.update(file=str(rel), out=str(out_path), status="failed")

    try:
        d = np.load(path)
        labels = read_labels(d, stem)
        row.update(subject=labels["subject"], activity=labels["activity"], day=labels["day"])

        csi = d["csi"]
        n, n_sub = csi.shape
        css = d["css"] if "css" in d.files else np.zeros(n, dtype=np.uint16)
        notes = []
        if "ts" in d.files:
            t_all = d["ts"].astype(np.float64)
        else:
            t_all = np.arange(n) / cfg["assumed_fs"]
            notes.append(f"no timestamps in file: assumed a uniform {cfg['assumed_fs']:g} Hz "
                         "(re-run extract_csi.py to get real timing)")
        idx = subcarrier_index(n_sub)

        values, counts = np.unique(css, return_counts=True)
        if cfg["streams"] == "main" and len(values) > 1:
            values = values[[int(np.argmax(counts))]]

        parts, dropped, seg_offset, fs_list = [], 0, 0, []
        for v in values:
            sel = css == v
            res = process_stream(csi if sel.all() else csi[sel], t_all[sel], idx, cfg)
            dropped += res["n_dropped"]
            fs_list.append(res["fs_in"])
            if res["windows"] is not None:
                res["stream"] = np.full(len(res["t_start"]), v, dtype=np.uint16)
                res["segment"] = res["segment"] + seg_offset
                parts.append(res)
            seg_offset += res["n_segments"]
        del csi

        if not parts:
            raise ValueError(f"no windows: recordings shorter than {cfg['window_sec']:g} s "
                             "or fully fragmented by gaps")
        windows = np.concatenate([p["windows"] for p in parts])
        saver = np.savez_compressed if cfg["compress"] else np.savez
        out_path.parent.mkdir(parents=True, exist_ok=True)
        saver(
            out_path,
            windows=windows,
            t_start=np.concatenate([p["t_start"] for p in parts]),
            stream=np.concatenate([p["stream"] for p in parts]),
            segment=np.concatenate([p["segment"] for p in parts]),
            subject=np.int16(labels["subject"]),
            activity_id=np.int16(labels["activity_id"]),
            day=np.int16(labels["day"]),
            activity=np.array(labels["activity"]),
            device=np.array(labels["device"]),
            fs=np.float32(cfg["target_fs"]),
            window_sec=np.float32(cfg["window_sec"]),
            stride_sec=np.float32(cfg["stride_sec"]),
            fs_in=np.float32(np.median(fs_list)),
        )
        if min(fs_list) < cfg["target_fs"]:
            notes.append(f"input rate {min(fs_list):.0f} Hz is below --target-fs: upsampled by interpolation")
        if len(values) > 1:
            notes.append(f"{len(values)} core/stream values processed separately (tagged in 'stream')")
        row.update(
            status="warning" if notes else "ok",
            frames_in=n,
            frames_dropped=dropped,
            fs_in_hz=f"{np.median(fs_list):.1f}",
            streams=len(values),
            segments=seg_offset,
            windows=len(windows),
            message="; ".join(notes),
        )
    except Exception as e:
        row["message"] = str(e)
    return row


def batch_preprocess(input_dir, output_dir, num_workers=2, **overrides):
    cfg = {**DEFAULTS, **overrides}
    in_root, out_root = Path(input_dir), Path(output_dir)
    files = sorted(in_root.rglob("*_csi.npz"))
    if not files:
        print(f"No *_csi.npz files found in {in_root}")
        return []
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Preprocessing {len(files)} recordings -> windows of {cfg['window_sec']:g} s at "
          f"{cfg['target_fs']:g} Hz ({num_workers} workers)\n")

    rows = []
    tasks = [(str(p), str(in_root), str(out_root), cfg) for p in files]
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(process_file, t) for t in tasks]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            tag = {"ok": "OK     ", "warning": "WARNING", "failed": "FAILED "}[row["status"]]
            detail = (f"{row['windows']} windows, input ~{row['fs_in_hz']} Hz, "
                      f"{row['frames_dropped']} frames dropped") if row["status"] != "failed" else ""
            print(f"[{tag}] {row['file']} {detail} {row['message']}".rstrip())

    rows.sort(key=lambda r: r["file"])
    with open(out_root / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    counts = {s: sum(r["status"] == s for r in rows) for s in ("ok", "warning", "failed")}
    print(f"\nDone: {counts['ok']} ok, {counts['warning']} with warnings, {counts['failed']} failed.")
    per_subject = {}
    for r in rows:
        if r["status"] != "failed":
            per_subject[int(r["subject"])] = per_subject.get(int(r["subject"]), 0) + int(r["windows"])
    if per_subject:
        print("Windows per subject: " + ", ".join(f"P{s}: {n}" for s, n in sorted(per_subject.items())))
    print(f"Manifest: {out_root / 'manifest.csv'}")
    return rows


# ---------------------------------------------------------------------------
# loading and leave-one-subject-out splits
# ---------------------------------------------------------------------------

def load_windows(directory, subjects=None, activities=None, days=None):
    """
    Concatenate the *_preprocessed.npz files of a folder. Optional filters by
    subject / activity_id / day (iterables of ints) skip files before reading
    their windows. Returns a dict:
        X (N, 2, T, F), y (activity_id), subject, day, stream, t_start,
        file (index into the 'files' list), files (list of paths)
    """
    subjects = None if subjects is None else {int(s) for s in subjects}
    activities = None if activities is None else {int(a) for a in activities}
    days = None if days is None else {int(x) for x in days}
    files = sorted(Path(directory).rglob(f"*{OUT_SUFFIX}"))
    keys = ("X", "y", "subject", "day", "stream", "t_start", "file")
    parts = {k: [] for k in keys}
    used = []
    for path in files:
        d = np.load(path)
        s, a, dy = int(d["subject"]), int(d["activity_id"]), int(d["day"])
        if (subjects is not None and s not in subjects) or \
           (activities is not None and a not in activities) or \
           (days is not None and dy not in days):
            continue
        n = len(d["windows"])
        parts["X"].append(d["windows"])
        parts["y"].append(np.full(n, a, dtype=np.int16))
        parts["subject"].append(np.full(n, s, dtype=np.int16))
        parts["day"].append(np.full(n, dy, dtype=np.int16))
        parts["stream"].append(d["stream"])
        parts["t_start"].append(d["t_start"])
        parts["file"].append(np.full(n, len(used), dtype=np.int32))
        used.append(str(path))
    if not used:
        raise ValueError(f"no matching *{OUT_SUFFIX} files in {directory}")
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["files"] = used
    return out


def loso_folds(subjects, n_val=1):
    """
    Leave-one-subject-out folds. For every subject: test = that subject,
    val = n_val other subjects (rotating, so each subject serves as validation
    in turn), train = everyone else. Returns a list of dicts of subject ids.
    """
    subs = sorted({int(s) for s in np.asarray(subjects).ravel()})
    if len(subs) < n_val + 2:
        raise ValueError(f"need at least {n_val + 2} subjects for n_val={n_val}, got {len(subs)}")
    folds = []
    for i, test in enumerate(subs):
        others = [s for s in subs if s != test]
        val = [others[(i + j) % len(others)] for j in range(n_val)]
        folds.append(dict(test=[test], val=val, train=[s for s in others if s not in val]))
    return folds


# ---------------------------------------------------------------------------

def main():
    D = DEFAULTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/HAR-3/CSI_processed/M1", help="folder with *_csi.npz (from extract_csi.py)")
    ap.add_argument("--output", default="data/HAR-3/CSI_preprocessed/M1", help="folder for *_preprocessed.npz")
    ap.add_argument("--workers", type=int, default=2, help="each worker holds a whole recording in memory")
    ap.add_argument("--target-fs", type=float, default=D["target_fs"], help="Hz of the output windows")
    ap.add_argument("--fs", type=float, default=D["assumed_fs"], dest="assumed_fs",
                    help="Hz to assume for files without timestamps")
    ap.add_argument("--cutoff", type=float, default=D["cutoff"], help="low-pass cutoff in Hz")
    ap.add_argument("--window-sec", type=float, default=D["window_sec"])
    ap.add_argument("--stride-sec", type=float, default=D["stride_sec"])
    ap.add_argument("--hampel-window", type=int, default=D["hampel_window"], help="frames; 0 disables")
    ap.add_argument("--hampel-sigma", type=float, default=D["hampel_sigma"])
    ap.add_argument("--min-energy-ratio", type=float, default=D["min_energy_ratio"],
                    help="drop frames with less than this fraction of the median energy; 0 disables")
    ap.add_argument("--gap-sec", type=float, default=D["gap_sec"], help="minimum gap that splits a recording")
    ap.add_argument("--streams", choices=["all", "main"], default=D["streams"],
                    help="all: every core/stream value separately; main: only the most frequent one")
    ap.add_argument("--dtype", choices=["float32", "float16"], default=D["dtype"])
    ap.add_argument("--compress", action="store_true", help="compress the output (slower, ~10%% smaller)")
    a = ap.parse_args()

    batch_preprocess(
        a.input, a.output, num_workers=a.workers,
        target_fs=a.target_fs, assumed_fs=a.assumed_fs, cutoff=a.cutoff,
        window_sec=a.window_sec, stride_sec=a.stride_sec, hampel_window=a.hampel_window,
        hampel_sigma=a.hampel_sigma, min_energy_ratio=a.min_energy_ratio, gap_sec=a.gap_sec,
        streams=a.streams, dtype=a.dtype, compress=a.compress,
    )


if __name__ == "__main__":
    main()