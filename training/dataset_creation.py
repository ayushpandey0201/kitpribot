#!/usr/bin/env python3
"""
KitPri v4 — complete synthetic dataset build pipeline.

Converted from the Colab notebook `kitpri_v3_rebuild_dataset.ipynb`.
NOTE ON NAMING: the notebook was reused across dataset versions and its "v3"
labels were never bumped — it IS the v4 builder. Version strings are corrected
to v4.0 here. (The live Kaggle dataset's manifest may still say v3.0; the
audio and metadata are the v4 build.)

Pipeline stages (one function per original notebook cell):
  1. inventory  — walk soundbank, probe + SHA256 every stem
  2. grouping   — merge stems that are segments of one recording (leak guard)
  3. partition  — group-level 70/15/15 split, leakage gates
  4. plan       — decide every mix on paper, pre-render balance gates
  5. render     — 10 s / 32 kHz / mono wavs, RMS-normalized to -20 dBFS
  6. augment    — train-only time_stretch + pitch_shift (x2 per clip)
  7. metadata   — master_metadata.csv + train/val/test.csv with full lineage
  8. gates      — 20 integrity gates; any red gate = DO NOT TRAIN
  9. package    — SHA256SUMS.txt, build_config.json, single zip for Kaggle

Label rule: a clip is COOKING iff it contains >=1 CORE COOKING sound.
water_tap and dishes are NEUTRAL — they never determine the label.

Expected output with defaults (seed 1337, 3000 base clips):
  2100/450/450 base -> +2 augmentations per train clip -> 6300/450/450 = 7200.

REPRODUCIBILITY: --seed default 1337 and the derived streams (SEED, SEED+7,
SEED+8, SEED+9, SEED+100) are exactly the notebook's. Changing any of them
produces a dataset that does NOT match the reported metrics.

Omitted from the notebook (Colab-only): drive.mount, the Drive->local copy
stage (the script reads the soundbank directly), !pip install, and dev
scratch/diagnostic cells.

Usage:
    python training/dataset_creation.py \
        --soundbank /path/to/raw_sources --out /path/to/kitpri_v4_build
    # requires: soundfile pandas numpy librosa   (pip install 'kitpri[datagen]')
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore")

BUILD_VERSION = "v4.0"   # notebook said "v3.0" — stale label, corrected.

# ---- audio spec (VERIFIED — do not change) ----------------------------------
SR          = 32000
CLIP_SEC    = 10.0
CLIP_LEN    = int(SR * CLIP_SEC)
PEAK_DBFS   = -1.0
MIN_STEM_S  = 1.0
XFADE_MS    = 50
TARGET_RMS_DBFS = -20.0          # v3.1 fix: equal loudness kills the loudness cheat
TARGET_RMS  = 10 ** (TARGET_RMS_DBFS / 20.0)

# ---- build (VERIFIED) --------------------------------------------------------
DEFAULT_SEED    = 1337
N_BASE_CLIPS    = 3000
SPLIT_FRACS     = {"train": 0.70, "val": 0.15, "test": 0.15}
AUG_TRAIN_ONLY  = True
AUG_PER_CLIP    = 2

# ---- label rulebook (D1 committed) -------------------------------------------
CORE_COOKING = {"frying", "boiling", "chopping", "stove",
                "stirring", "blender", "pressure_cooker", "microwave"}
NEUTRAL      = {"water_tap", "dishes"}
NONCOOKING   = {"speech", "TV_audio", "music", "footsteps", "door_knock",
                "phone_ringing", "keyboard_typing", "vacuum_cleaner",
                "dog_barking"}

# ---- clip taxonomy: (label, n_core, n_noncook, n_neutral, fraction) -----------
CLIP_TYPES = {
    "A": (1, 1, 0, 0, 0.15),
    "B": (1, 2, 0, 0, 0.08),
    "C": (1, 1, 1, 0, 0.17),   # cooking + competing non-cooking sound — hard cases
    "D": (1, 1, 0, 1, 0.10),
    "E": (0, 0, 1, 0, 0.08),
    "F": (0, 0, 2, 0, 0.22),
    "G": (0, 0, 1, 1, 0.06),
    "H": (0, 0, 0, 1, 0.04),
    "I": (0, 0, 0, 0, 0.10),
}

SNR_RANGE = (-5.0, 20.0)
MIN_COOK_VS_NONCOOK_DB = -5.0
BALANCE_TOL = 0.05
NUMSTEM_TOL = 0.10

AUDIO_EXTS = {".wav", ".WAV", ".flac", ".FLAC"}
TRAIL_ID = re.compile(r"^(?P<prefix>.*?)[_\-]?(?P<num>\d{3,})$")
MIN_GROUPS_PER_CLASS = 20
DROP_CROSS_ROLE_DUPES = True


# ── helpers ────────────────────────────────────────────────────────────────────

def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def role_of(audio_class):
    if audio_class in CORE_COOKING: return "core_cooking"
    if audio_class in NEUTRAL:      return "neutral"
    if audio_class in NONCOOKING:   return "noncooking"
    return "background"


def corpus_of(name):
    n = name.lower()
    if n.startswith("noise-free-sound"): return "musan"
    if re.match(r"^\d+-\d+-[a-z]-\d+", n): return "esc50"
    return "freesound"


def rms(x): return float(np.sqrt(np.mean(np.square(x)) + 1e-12))
def db(x):  return 20 * math.log10(max(x, 1e-12))


def _librosa():
    try:
        import librosa
        return librosa
    except ImportError as e:
        raise SystemExit(
            "ERROR: librosa is required for rendering/augmentation. "
            "Install with: pip install 'kitpri[datagen]'"
        ) from e


# ── STAGE 1: inventory + hash ──────────────────────────────────────────────────

def stage_inventory(src: Path, meta_out: Path, manif_out: Path) -> pd.DataFrame:
    files = [p for p in src.rglob("*") if p.suffix in AUDIO_EXTS]
    if not files:
        raise SystemExit(f"ERROR: no audio stems found under {src}")
    print(f"Scanning {len(files)} stems...")

    rows = []
    for i, p in enumerate(files):
        if i and i % 500 == 0:
            print(f"  {i}/{len(files)}")
        parts = p.relative_to(src).parts
        # raw_sources/background/<class>/f.wav
        # raw_sources/foreground/<cooking|noncooking>/<class>/f.wav
        if parts[0] == "background":
            audio_class, src_dir = parts[1], "background"
        elif parts[0] == "foreground":
            audio_class, src_dir = parts[2], f"foreground/{parts[1]}"
        else:
            audio_class, src_dir = parts[-2], parts[0]

        rec = {"stem_path": str(p), "rel_path": str(p.relative_to(src)),
               "file_name": p.name, "audio_class": audio_class,
               "src_dir": src_dir, "stem_role": role_of(audio_class)}
        try:
            info = sf.info(str(p))
            rec.update(duration_sec=round(info.frames / info.samplerate, 3),
                       native_sr=info.samplerate, channels=info.channels,
                       readable=True, probe_error="")
        except Exception as e:
            rec.update(duration_sec=0.0, native_sr=0, channels=0,
                       readable=False, probe_error=repr(e)[:120])
        rec["sha256"] = sha256_of(p)
        rows.append(rec)

    inv = pd.DataFrame(rows)
    inv["is_usable"] = inv["readable"] & (inv["duration_sec"] >= MIN_STEM_S)
    inv["source_dataset"] = inv["file_name"].apply(corpus_of)

    print(f"\nstems={len(inv)}  usable={inv.is_usable.sum()}  "
          f"exact_dupes={len(inv) - inv.sha256.nunique()}")
    print("\nby stem_role:"); print(inv.stem_role.value_counts().to_string())
    unknown = inv[inv.stem_role == "background"]["audio_class"].unique()
    print(f"\nclasses treated as background: {sorted(unknown)}")
    print(">>> verify none of these should have been a foreground class")

    inv.to_csv(meta_out / "stem_inventory.csv", index=False)
    with open(manif_out / "SOUNDBANK_SHA256.txt", "w") as f:
        for _, r in inv.sort_values("rel_path").iterrows():
            f.write(f"{r.sha256}  {r.rel_path}\n")
    return inv


# ── STAGE 2: source grouping ───────────────────────────────────────────────────

def _build_groups(df, window):
    out, buckets = {}, defaultdict(list)
    for idx, r in df.iterrows():
        stem = Path(r["file_name"]).stem
        m = TRAIL_ID.match(stem)
        if m:
            buckets[(r["audio_class"], m.group("prefix"))].append(
                (int(m.group("num")), idx))
        else:
            out[idx] = f"{r['audio_class']}::{stem}"
    for (cls, pre), items in buckets.items():
        items.sort()
        gid, prev = 0, None
        for num, idx in items:
            if prev is not None and num - prev > window:
                gid += 1
            out[idx] = f"{cls}::{pre}::g{gid}"
            prev = num
    return pd.Series(out).reindex(df.index)


def stage_grouping(inv: pd.DataFrame) -> pd.DataFrame:
    candidates = {}
    for w in [0, 10, 50, 200, 1000]:
        g = _build_groups(inv, w)
        per_class = pd.DataFrame({"c": inv.audio_class, "g": g}) \
                      .groupby("c")["g"].nunique()
        candidates[w] = {"groups": g.nunique(),
                         "min_groups_in_a_class": int(per_class.min()),
                         "classes_too_thin": int((per_class < MIN_GROUPS_PER_CLASS).sum()),
                         "series": g}
    print("GROUPING (window = max ID gap to still merge):")
    print(pd.DataFrame({w: {k: v for k, v in d.items() if k != "series"}
                        for w, d in candidates.items()}).T.to_string())

    viable = [w for w, d in candidates.items() if d["classes_too_thin"] == 0]
    chosen = max(viable) if viable else 0
    if not viable:
        print("*** No window keeps every class splittable — falling back to file-level (0)")
    print(f"CHOSEN WINDOW: {chosen}")

    inv = inv.copy()
    inv["source_group"] = candidates[chosen]["series"].values
    dup_map = inv.groupby("sha256")["source_group"].transform("first")
    inv["source_group"] = np.where(inv.duplicated("sha256", keep=False),
                                   dup_map, inv["source_group"])
    inv.attrs["grouping_window"] = chosen
    print(f"final groups: {inv.source_group.nunique()} (from {len(inv)} stems)")
    return inv


# ── STAGE 3: partition ─────────────────────────────────────────────────────────

def stage_partition(inv: pd.DataFrame, meta_out: Path, seed: int):
    assign_path = meta_out / "stem_assignment.csv"
    assign_path.unlink(missing_ok=True)
    usable = inv[inv.is_usable].copy()

    role_span = usable.groupby("sha256")["stem_role"].nunique()
    bad_hashes = set(role_span[role_span > 1].index)
    print(f"identical files carrying >1 ROLE (label noise): {len(bad_hashes)}")
    if DROP_CROSS_ROLE_DUPES and bad_hashes:
        before = len(usable)
        usable = usable[~usable.sha256.isin(bad_hashes)]
        print(f"dropped {before - len(usable)} stems whose audio carried two roles")

    grp = (usable.groupby("source_group")
                 .agg(audio_class=("audio_class", lambda s: s.mode().iat[0]),
                      n_files=("source_group", "size"))
                 .reset_index())
    print(f"unique source groups: {len(grp)} covering {len(usable)} stems")

    rng = np.random.default_rng(seed)                     # notebook: SEED
    recs = []
    for cls, sub in grp.groupby("audio_class"):
        g = sub.source_group.to_numpy().copy()
        rng.shuffle(g)
        n = len(g)
        n_tr = int(round(n * SPLIT_FRACS["train"]))
        n_va = int(round(n * SPLIT_FRACS["val"]))
        for i, gid in enumerate(g):
            sp = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
            recs.append({"source_group": gid, "audio_class": cls, "split": sp})

    assign = pd.DataFrame(recs).drop_duplicates("source_group")
    assert assign.source_group.is_unique, "assignment not unique"
    assign.to_csv(assign_path, index=False)
    print(f"wrote {assign_path} (IMMUTABLE from now on)")

    split_map = dict(zip(assign.source_group, assign.split))
    usable["split"] = usable.source_group.map(split_map)
    pool = usable[usable.split.notna()].copy()

    g2s = pool.groupby("source_group")["split"].nunique()
    assert (g2s == 1).all(), f"GATE FAIL: {(g2s > 1).sum()} groups span splits"
    h2s = pool.groupby("sha256")["split"].nunique()
    assert (h2s == 1).all(), f"GATE FAIL: {(h2s > 1).sum()} identical files span splits"
    print("stems per split:"); print(pool.split.value_counts().to_string())
    print("GATES PASSED: every group and unique file lives in exactly one split.")
    return pool


# ── STAGE 4: generation plan ───────────────────────────────────────────────────

class _Deck:
    """Cycles a shuffled pool — gives every background class ~50/50 label
    balance BY CONSTRUCTION (kills v2's F1 0.719 background shortcut)."""

    def __init__(self, df, rng):
        self.df, self.rng, self.order, self.i = df, rng, [], 0
        self._reshuffle()

    def _reshuffle(self):
        self.order = list(range(len(self.df)))
        self.rng.shuffle(self.order)
        self.i = 0

    def draw(self, n):
        out = []
        while len(out) < n:
            if self.i >= len(self.order):
                self._reshuffle()
            out.append(self.df.iloc[self.order[self.i]])
            self.i += 1
        return out


def _pick(df, rng, n):
    if len(df) == 0 or n == 0:
        return []
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    return [df.iloc[i] for i in idx]


def stage_plan(pool: pd.DataFrame, meta_out: Path, seed: int, n_base: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 7)                 # notebook: SEED+7

    pools = {}
    for sp in ["train", "val", "test"]:
        s = pool[pool.split == sp]
        pools[sp] = {r: s[s.stem_role == r].reset_index(drop=True)
                     for r in ["core_cooking", "neutral", "noncooking", "background"]}
        print(f"{sp}: " + "  ".join(f"{k}={len(v)}" for k, v in pools[sp].items()))

    bg_decks = {sp: _Deck(pools[sp]["background"], np.random.default_rng(seed + 100))
                for sp in ["train", "val", "test"]}       # notebook: SEED+100

    plan_rows, clip_no = [], 0
    for sp, frac in SPLIT_FRACS.items():
        n_clips = int(round(n_base * frac))
        counts = {t: int(round(n_clips * v[4])) for t, v in CLIP_TYPES.items()}
        order = []
        for t, c in counts.items():
            order += [t] * c
        rng.shuffle(order)

        for t in order:
            label, n_core, n_non, n_neu, _ = CLIP_TYPES[t]
            n_fg = n_core + n_non + n_neu
            n_bg = 1 if rng.random() < 0.5 else 2

            P = pools[sp]
            stems = []
            stems += [("core_cooking", s) for s in _pick(P["core_cooking"], rng, n_core)]
            stems += [("noncooking",   s) for s in _pick(P["noncooking"],   rng, n_non)]
            stems += [("neutral",      s) for s in _pick(P["neutral"],      rng, n_neu)]
            stems += [("background",   s) for s in bg_decks[sp].draw(n_bg)]

            got_core = sum(1 for r, _ in stems if r == "core_cooking")
            if label == 1 and got_core == 0:
                continue
            if label == 0 and got_core > 0:
                continue

            clip_no += 1
            fid = f"{'c' if label == 1 else 'n'}_{clip_no:05d}"
            row = {"file_id": fid, "label": label, "split": sp, "clip_type": t,
                   "num_stems": len(stems), "n_fg": n_fg, "n_bg": n_bg,
                   "SNR_dB": round(float(rng.uniform(*SNR_RANGE)), 2),
                   "has_cross_class_overlap": bool(n_core > 0 and n_non > 0),
                   "has_neutral": bool(n_neu > 0)}
            for i, (role, s) in enumerate(stems, start=1):
                row[f"stem{i}_role"] = role
                row[f"stem{i}_class"] = s["audio_class"]
                row[f"stem{i}_file"] = s["stem_path"]
                row[f"stem{i}_group"] = s["source_group"]
                row[f"stem{i}_dataset"] = s["source_dataset"]
                row[f"stem{i}_sha256"] = s["sha256"]
            plan_rows.append(row)

    plan = pd.DataFrame(plan_rows)
    print(f"\nplanned clips: {len(plan)}")
    print(pd.crosstab(plan.label, plan.split).to_string())

    bgmask = pd.concat([plan[[f"stem{i}_role", f"stem{i}_class", "label"]]
                        .rename(columns={f"stem{i}_role": "r", f"stem{i}_class": "v"})
                        for i in range(1, 5) if f"stem{i}_role" in plan.columns],
                       ignore_index=True).dropna()
    bgonly = bgmask[bgmask.r == "background"]
    bct = pd.crosstab(bgonly.v, bgonly.label)
    if 1 not in bct: bct[1] = 0
    worst = (bct[1] / bct.sum(axis=1) - 0.5).abs().max()
    print(f"worst background skew: {worst:.3f} (tolerance {BALANCE_TOL})")

    plan.to_csv(meta_out / "generation_plan.csv", index=False)
    print(">>> Review balance BEFORE rendering.")
    return plan


# ── STAGE 5: render ────────────────────────────────────────────────────────────

def _load_stem(path):
    librosa = _librosa()
    x, native = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if native != SR:
        x = librosa.resample(x, orig_sr=native, target_sr=SR)
    return np.ascontiguousarray(x, dtype=np.float32)


def _fit_length(x, n=CLIP_LEN, rng=None):
    if len(x) == 0:
        return np.zeros(n, np.float32), "zeros", 0
    if len(x) >= n:
        start = 0 if rng is None else int(rng.integers(0, len(x) - n + 1))
        return x[start:start + n].copy(), "crop", 1
    fade = min(int(SR * XFADE_MS / 1000), len(x) // 4)
    out, loops, cur = [], 0, x.copy()
    while sum(len(s) for s in out) < n:
        if out and fade > 0:
            prev = out[-1]
            w = np.linspace(0, 1, fade, dtype=np.float32)
            prev[-fade:] = prev[-fade:] * (1 - w) + cur[:fade] * w
            out.append(cur[fade:].copy())
        else:
            out.append(cur.copy())
        loops += 1
        if loops > 200:
            break
    y = np.concatenate(out)[:n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y.astype(np.float32), "loop_xfade", loops


def stage_render(plan: pd.DataFrame, audio_out: Path, seed: int) -> pd.DataFrame:
    rng_r = np.random.default_rng(seed + 8)               # notebook: SEED+8
    log, t0 = [], time.time()

    for k, (_, r) in enumerate(plan.iterrows()):
        out_dir = audio_out / ("cooking" if r["label"] == 1 else "noncooking")
        out_path = out_dir / f"{r['file_id']}.wav"
        if out_path.exists():
            continue
        if k and k % 250 == 0:
            print(f"  {k}/{len(plan)}  ({time.time() - t0:.0f}s)")

        fg_parts, bg_parts, det = [], [], {}
        cook_sig, noncook_sig = None, None
        for i in range(1, 5):
            rc = f"stem{i}_role"
            if rc not in r or pd.isna(r.get(rc)):
                continue
            role = r[rc]
            y, pol, loops = _fit_length(_load_stem(r[f"stem{i}_file"]), rng=rng_r)
            det[f"stem{i}_pad_policy"] = pol
            det[f"stem{i}_n_loops"] = loops
            det[f"stem{i}_in_rms_dB"] = round(db(rms(y)), 2)
            if role == "background":
                bg_parts.append(y)
            else:
                fg_parts.append(y)
                if role == "core_cooking":
                    cook_sig = y if cook_sig is None else cook_sig + y
                elif role == "noncooking":
                    noncook_sig = y if noncook_sig is None else noncook_sig + y

        fg = np.sum(fg_parts, axis=0) if fg_parts else np.zeros(CLIP_LEN, np.float32)
        bg = np.sum(bg_parts, axis=0) if bg_parts else np.zeros(CLIP_LEN, np.float32)
        if fg_parts and bg_parts:
            target = 10 ** (r["SNR_dB"] / 20.0)
            bg = bg * (rms(fg) / (rms(bg) * target + 1e-12))
        mix = fg + bg

        cvn = None
        if cook_sig is not None and noncook_sig is not None:
            cvn = round(db(rms(cook_sig)) - db(rms(noncook_sig)), 2)

        # RMS normalize, THEN peak-limit (order matters — equal loudness for all)
        mix = mix * (TARGET_RMS / (rms(mix) + 1e-12))
        peak = float(np.max(np.abs(mix)))
        if peak > 0.99:
            mix = mix * (0.99 / peak)
        mix = np.clip(mix, -1.0, 1.0).astype(np.float32)

        sf.write(str(out_path), mix, SR, subtype="PCM_16")
        det.update(file_id=r["file_id"], wav_path=str(out_path),
                   measured_rms_dBFS=round(db(rms(mix)), 3),
                   measured_peak_dBFS=round(db(float(np.max(np.abs(mix)))), 3),
                   measured_snr_dB=(round(db(rms(fg)) - db(rms(bg)), 2)
                                    if fg_parts and bg_parts else None),
                   cook_vs_noncook_dB=cvn,
                   duration_s=round(len(mix) / SR, 3), sr=SR)
        log.append(det)

    render = pd.DataFrame(log)
    print(f"rendered {len(render)} clips in {time.time() - t0:.0f}s")
    if len(render):
        print(f"RMS spread (should be ~0): std={render.measured_rms_dBFS.std():.3f} dB")
    return render


# ── STAGE 6: augmentation (train only) ─────────────────────────────────────────

def stage_augment(plan: pd.DataFrame, audio_out: Path, seed: int) -> pd.DataFrame:
    librosa = _librosa()
    rng_a = np.random.default_rng(seed + 9)               # notebook: SEED+9
    targets = plan[plan.split == "train"] if AUG_TRAIN_ONLY else plan
    rows = []
    print(f"augmenting {len(targets)} clips x {AUG_PER_CLIP}...")

    for k, (_, r) in enumerate(targets.iterrows()):
        if k and k % 250 == 0:
            print(f"  {k}/{len(targets)}")
        src = audio_out / ("cooking" if r["label"] == 1 else "noncooking") / f"{r['file_id']}.wav"
        if not src.exists():
            continue
        y, _ = sf.read(str(src), dtype="float32")

        for kind in ["time_stretch", "pitch_shift"][:AUG_PER_CLIP]:
            fid = f"aug_{kind[:2]}_{r['file_id']}"
            dst = src.parent / f"{fid}.wav"
            if dst.exists():
                continue
            if kind == "time_stretch":
                rate = float(rng_a.uniform(0.9, 1.1))
                z = librosa.effects.time_stretch(y, rate=rate)
                param = round(rate, 4)
            else:
                steps = float(rng_a.uniform(-2, 2))
                z = librosa.effects.pitch_shift(y, sr=SR, n_steps=steps)
                param = round(steps, 4)
            z = z[:CLIP_LEN] if len(z) >= CLIP_LEN else np.pad(z, (0, CLIP_LEN - len(z)))
            # NOTE (faithful to the original build): augmented clips are
            # PEAK-normalized to -1 dBFS, unlike base clips which are
            # RMS-normalized. This asymmetry exists in the shipped dataset;
            # do not "fix" it here or regeneration won't match.
            p = float(np.max(np.abs(z))) + 1e-12
            z = (z * (10 ** (PEAK_DBFS / 20.0)) / p).astype(np.float32)
            sf.write(str(dst), z, SR, subtype="PCM_16")
            rows.append({"file_id": fid, "parent_id": r["file_id"],
                         "label": r["label"], "split": r["split"],
                         "clip_type": r["clip_type"], "aug_type": kind,
                         "aug_param": param, "wav_path": str(dst)})

    aug = pd.DataFrame(rows)
    print(f"created {len(aug)} augmented clips")
    return aug


# ── STAGE 7: master metadata + split CSVs ──────────────────────────────────────

def stage_metadata(plan, render, aug, meta_out: Path, seed: int) -> pd.DataFrame:
    master = plan.merge(render, on="file_id", how="inner")
    master["parent_id"] = master["file_id"]
    master["aug_type"] = None
    master["aug_param"] = None
    master["is_ambiguous_class"] = master.apply(
        lambda r: any(str(r.get(f"stem{i}_class")) in NEUTRAL for i in range(1, 5)),
        axis=1)
    master["below_audibility_floor"] = (
        master.cook_vs_noncook_dB.notna() &
        (master.cook_vs_noncook_dB < MIN_COOK_VS_NONCOOK_DB))
    print(f"flagged (not dropped) {master.below_audibility_floor.sum()} quiet type-C clips")

    if len(aug):
        inherit = master.set_index("file_id")
        extra = []
        for _, a in aug.iterrows():
            if a["parent_id"] not in inherit.index:
                continue
            base = inherit.loc[a["parent_id"]].to_dict()
            base.update(file_id=a["file_id"], parent_id=a["parent_id"],
                        aug_type=a["aug_type"], aug_param=a["aug_param"],
                        wav_path=a["wav_path"])
            extra.append(base)
        master = pd.concat([master, pd.DataFrame(extra)], ignore_index=True)

    master["rel_path"] = master.apply(
        lambda r: f"audio_32k/{'cooking' if r['label'] == 1 else 'noncooking'}/{r['file_id']}.wav",
        axis=1)
    master["sha256"] = master["wav_path"].apply(
        lambda p: sha256_of(p) if os.path.exists(p) else "")
    master["build_seed"] = seed
    master["build_version"] = BUILD_VERSION           # notebook wrote "v3.0" — stale

    master.to_csv(meta_out / "master_metadata.csv", index=False)
    for sp in ["train", "val", "test"]:
        sub = master[master.split == sp][
            ["rel_path", "label", "file_id", "split", "clip_type",
             "aug_type", "is_ambiguous_class", "parent_id"]]
        sub.to_csv(meta_out / f"{sp}.csv", index=False)
        print(f"{sp}: {len(sub)} clips  (cooking {sub.label.sum()})")

    # Reference build (seed 1337, 3000 base): 6300/450/450. Divergence is not
    # fatal (soundbank may differ) but means metrics are not comparable.
    expected = {"train": 6300, "val": 450, "test": 450}
    got = master.split.value_counts().to_dict()
    if got != expected:
        print(f"*** WARNING: split counts {got} != reference {expected} — "
              "this build will NOT match the reported metrics.")
    return master


# ── STAGE 8: integrity gates ───────────────────────────────────────────────────

def stage_gates(master, pool, audio_out: Path, report_out: Path) -> bool:
    results, failures = {}, []

    def gate(name, ok, detail=""):
        results[name] = {"pass": bool(ok), "detail": str(detail)}
        if not ok:
            failures.append(name)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    print("LEAKAGE GATES")
    stem_split = defaultdict(set)
    for _, r in master.iterrows():
        for i in range(1, 5):
            v = r.get(f"stem{i}_file")
            if isinstance(v, str) and v:
                stem_split[v].add(r["split"])
    gate("1_no_stem_across_splits",
         sum(1 for v in stem_split.values() if len(v) > 1) == 0)

    grp_split = defaultdict(set)
    for _, r in master.iterrows():
        for i in range(1, 5):
            v = r.get(f"stem{i}_group")
            if isinstance(v, str) and v:
                grp_split[v].add(r["split"])
    gate("2_no_group_across_splits",
         sum(1 for v in grp_split.values() if len(v) > 1) == 0)

    train_stems = {k for k, v in stem_split.items() if "train" in v}
    te = master[master.split == "test"]
    contam = te.apply(lambda r: any(isinstance(r.get(f"stem{i}_file"), str)
                                    and r.get(f"stem{i}_file") in train_stems
                                    for i in range(1, 5)), axis=1).sum()
    gate("3_test_clean_of_train_stems", contam == 0, f"{contam}/{len(te)}")

    h = master[master.sha256 != ""].groupby("sha256")["split"].nunique()
    gate("4_no_duplicate_audio_across_splits", (h <= 1).all())
    p = master.groupby("parent_id")["split"].nunique()
    gate("5_aug_parents_single_split", (p <= 1).all())

    print("\nSHORTCUT GATES (cheat classifiers must score at chance)")

    def cheat(series, name):
        d = pd.DataFrame({"v": series, "label": master.label,
                          "split": master.split}).dropna()
        tr, tv = d[d.split == "train"], d[d.split == "test"]
        if len(tr) == 0 or len(tv) == 0:
            gate(name, True, "insufficient data"); return
        rule = tr.groupby("v")["label"].mean()
        pred = (tv["v"].map(rule).fillna(0.5) > 0.5).astype(int)
        acc = float((pred == tv["label"]).mean())
        gate(name, abs(acc - 0.5) <= 0.10, f"acc={acc:.3f}")

    cheat(master.apply(lambda r: next((r.get(f"stem{i}_class") for i in range(1, 5)
                                       if r.get(f"stem{i}_role") == "background"), None),
                       axis=1), "6_background_only_cheat")
    cheat(master.apply(lambda r: next((r.get(f"stem{i}_dataset") for i in range(1, 5)
                                       if r.get(f"stem{i}_role") == "background"), None),
                       axis=1), "7_source_dataset_cheat")
    cheat(master.num_stems.where(master.clip_type != "I"), "8_num_stems_cheat_excl_typeI")
    cheat(master.measured_rms_dBFS.round(0), "9_loudness_cheat")
    cheat(master.SNR_dB.round(-1), "10_snr_cheat")

    print("\nBALANCE GATES")
    bgall = pd.concat([master[[f"stem{i}_role", f"stem{i}_class", "label"]]
                       .rename(columns={f"stem{i}_role": "r", f"stem{i}_class": "v"})
                       for i in range(1, 5) if f"stem{i}_role" in master.columns],
                      ignore_index=True).dropna()
    bo = bgall[bgall.r == "background"]
    ct = pd.crosstab(bo.v, bo.label)
    if 1 not in ct: ct[1] = 0
    skew = (ct[1] / ct.sum(axis=1) - 0.5).abs()
    gate("11_background_label_balance", (skew <= BALANCE_TOL).all(),
         f"worst={skew.max():.3f}")

    nu = bgall[bgall.r == "neutral"]
    if len(nu):
        ctn = pd.crosstab(nu.v, nu.label)
        if 1 not in ctn: ctn[1] = 0
        sk = (ctn[1] / ctn.sum(axis=1) - 0.5).abs()
        gate("12_neutral_label_balance", (sk <= BALANCE_TOL).all(),
             f"worst={sk.max():.3f}")

    _m = master[master.clip_type != "I"]
    ns = pd.crosstab(_m.num_stems, _m.label, normalize="columns")
    gate("13_num_stems_balance_excl_typeI",
         (ns[0] - ns[1]).abs().max() <= NUMSTEM_TOL,
         f"max_diff={(ns[0] - ns[1]).abs().max():.3f}")

    print("\nCOVERAGE + FILE GATES")
    cls_split = pd.crosstab(pool.audio_class, pool.split)
    gate("14_all_classes_in_all_splits", (cls_split > 0).all().all())
    gate("15_min_stems_in_test", (cls_split["test"] >= 2).all(),
         f"min={cls_split['test'].min()}")

    on_disk = {p.stem for p in audio_out.rglob('*.wav')}
    gate("16_csv_matches_disk", set(master.file_id) == on_disk,
         f"csv={len(master)} disk={len(on_disk)}")
    gate("17_uniform_duration", master.duration_s.round(1).nunique() == 1)
    gate("18_uniform_sr", master.sr.nunique() == 1)
    gate("19_no_silent_clips", (master.measured_rms_dBFS > -60).all())
    gate("20_label_matches_folder", master.apply(
        lambda r: (r["label"] == 1) == ("/cooking/" in r["wav_path"]), axis=1).all())

    print("\n" + "=" * 60)
    if failures:
        print(f"BUILD FAILED — {len(failures)} red gates:")
        for f in failures:
            print(f"   {f}")
        print("DO NOT TRAIN ON THIS DATASET.")
    else:
        print("ALL GATES PASSED.")
    print("=" * 60)

    with open(report_out / "integrity_report.json", "w") as f:
        json.dump({"passed": len(failures) == 0, "failures": failures,
                   "gates": results}, f, indent=2)
    return len(failures) == 0


# ── STAGE 9: package ───────────────────────────────────────────────────────────

def stage_package(master, build: Path, manif_out: Path, seed: int,
                  grouping_window: int, make_zip: bool):
    with open(manif_out / "SHA256SUMS.txt", "w") as f:
        for _, r in master.sort_values("rel_path").iterrows():
            f.write(f"{r['sha256']}  {r['rel_path']}\n")

    with open(manif_out / "build_config.json", "w") as f:
        json.dump({
            "version": BUILD_VERSION,               # was "v3.0" in the notebook — stale
            "seed": seed, "sr": SR, "clip_sec": CLIP_SEC,
            "n_clips": int(len(master)),
            "core_cooking": sorted(CORE_COOKING), "neutral": sorted(NEUTRAL),
            "noncooking": sorted(NONCOOKING),
            "clip_types": {k: list(v) for k, v in CLIP_TYPES.items()},
            "grouping_window": int(grouping_window),
            "audibility_floor_dB": MIN_COOK_VS_NONCOOK_DB,
            "aug_train_only": AUG_TRAIN_ONLY,
            "D1": {"water_tap": "neutral", "dishes": "neutral",
                   "microwave": "core_cooking"},
        }, f, indent=2)

    if make_zip:
        zip_base = str(build.parent / "kitpri_v4")   # notebook: kitpri_v3 — stale label
        shutil.make_archive(zip_base, "zip", build)
        size_mb = os.path.getsize(zip_base + ".zip") / 1e6
        print(f"archive: {zip_base}.zip ({size_mb:.0f} MB)")
        print(f"""
NEXT:
  1. Upload the zip to Kaggle as a SINGLE archive (not a folder of files —
     v2 lost 1,492 of 5,000 files to a silent multi-file upload failure).
  2. On Kaggle, BEFORE training: count files (must equal {len(master)}),
     spot-check sha256 against manifests/SHA256SUMS.txt.
""")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="KitPri v4 synthetic dataset builder (converted from Colab notebook)")
    ap.add_argument("--soundbank", required=True,
                    help="path to raw_sources/ (background/ + foreground/ stem folders)")
    ap.add_argument("--out", default="kitpri_v4_build",
                    help="build output directory (default: ./kitpri_v4_build)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"build seed (default {DEFAULT_SEED} — the reference build; "
                         "changing it produces a dataset that won't match reported metrics)")
    ap.add_argument("--n-base-clips", type=int, default=N_BASE_CLIPS)
    ap.add_argument("--clean", action="store_true",
                    help="wipe previously rendered audio first (render is otherwise resumable)")
    ap.add_argument("--no-zip", action="store_true", help="skip the final zip")
    args = ap.parse_args()

    src = Path(args.soundbank).expanduser().resolve()
    build = Path(args.out).expanduser().resolve()
    audio_out, meta_out = build / "audio_32k", build / "metadata"
    manif_out, report_out = build / "manifests", build / "reports"
    for d in [audio_out / "cooking", audio_out / "noncooking",
              meta_out, manif_out, report_out]:
        d.mkdir(parents=True, exist_ok=True)

    assert abs(sum(v[4] for v in CLIP_TYPES.values()) - 1.0) < 1e-9
    cook_frac = sum(v[4] for v in CLIP_TYPES.values() if v[0] == 1)
    print(f"KitPri {BUILD_VERSION} dataset build  |  seed={args.seed}  "
          f"base_clips={args.n_base_clips}")
    print(f"label balance by design: cooking {cook_frac:.0%} / noncooking {1 - cook_frac:.0%}")

    if args.clean:
        for sub in ["cooking", "noncooking"]:
            d = audio_out / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        print("clean slate for render")

    inv = stage_inventory(src, meta_out, manif_out)
    inv = stage_grouping(inv)
    pool = stage_partition(inv, meta_out, args.seed)
    plan = stage_plan(pool, meta_out, args.seed, args.n_base_clips)
    render = stage_render(plan, audio_out, args.seed)
    aug = stage_augment(plan, audio_out, args.seed)
    master = stage_metadata(plan, render, aug, meta_out, args.seed)
    ok = stage_gates(master, pool, audio_out, report_out)
    stage_package(master, build, manif_out, args.seed,
                  inv.attrs.get("grouping_window", 0), make_zip=not args.no_zip)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
