# -*- coding: utf-8 -*-
"""
recent_models — rescore four recent (2024–2025) models under the existing protocol unchanged.

Uses the **same replicates** as bootstrap. Replicate b draws its training sample with SEED+b, so
the same b in both outputs is the same participant sample. That is what allows paired differences
within a replicate — drawing separately would only give the difference of two means.

  · Validation  train on Hospital A (25) -> test on Hospital B (18), fixed.
  · Input       physio = 18 physiological columns, combined = those 18 + 8 calendar columns. Same as before.
  · Metrics     macro-F1 primary, ov-F1 secondary.
  · Device      GPU. Each worker takes one card via OVUL_GPU.

  python3 src/train/recent_models.py --n-boot 1000 --workers 4

GPUs are restricted by OVUL_ALLOWED_GPUS (e.g. 3,4) — never spills onto other cards.
Output -> results/boot_sota/sota.csv
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
# Other experiments disable CUDA but a GPU is required here. On CPU, TabPFN refuses more than
# 1,000 rows and one TabM fit takes 60 s, so 8,000 fits are not feasible.
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import argparse
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model.recent import MODELS                                      # noqa: E402
from cohort import load, audit, macro, ovf1, OV_CLS, LABELS  # noqa: E402
from features import SEED                                        # noqa: E402
from phase_labels import phase5, collapse                       # noqa: E402

_rd = os.environ.get("OVUL_RESULT_DIR", "").strip()
OUT = Path(_rd) if _rd else ROOT / "results" / "boot_sota"
ORDER = ["TabPFNv2", "TabICL", "TabM", "Mamba"]
_S = {}


# Cards this project is allowed to use. **Never exceed them.**
#
# Automatically picking "empty" cards once spread 16 workers over ten GPUs,
# each holding 6–7 GB per card, and another user's training on the same machine
# died with OOM. 'Currently empty' does not mean 'OK to use'. Automatically grabbing
# spare capacity on a shared machine is the wrong design.
# Restrict usable cards on the shared machine. Empty means all visible cards.
ALLOWED_GPUS = tuple(int(x) for x in os.environ.get("OVUL_ALLOWED_GPUS", "").split(",") if x.strip()) or tuple(range(64))


def free_gpus(min_free_mb=15000):
    """Allowed cards that are usable right now.

    Never goes outside ALLOWED_GPUS. The memory check is there so our job does not fail
    silently, not to find more cards. One worker holds
    6–7 GB and two go on each card, so 14 GB must be free.
    """
    import subprocess
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    out, tight = [], []
    for ln in q:
        i, tot, used = (int(x) for x in ln.split(","))
        if i not in ALLOWED_GPUS:
            continue
        (out if tot - used >= min_free_mb else tight).append((i, tot - used))
    if tight:
        print("  Skipped cards with too little free memory: "
              + ", ".join(f"GPU {i}({f // 1024} GB)" for i, f in tight), flush=True)
    return [i for i, _ in out]


def init(gpus, tag):
    """**Pin** the worker to one card. CUDA_VISIBLE_DEVICES exposes only that card.

    This was gotten wrong twice.

    1) An mp.Value counter to give each worker one card -> deadlock. Workers are started
       with fork, and if someone is forked while holding that lock the child hangs on a futex. The parent
       sat in pipe_write and all sixteen workers in futex_wait, running no job for 30 minutes.
    2) So the device was assigned **per job** -> no shared state, but one worker
       **created a context on every card in the list** as it processed jobs. An empty context is 202 MiB
       and seemed fine, but the caching allocator held freed blocks on each card, so in practice
       it was 6–7 GB per card. 16 workers ate every GPU and **crashed another user's training with
       OOM.**

    The fix avoids both. Slots are claimed by file creation (O_EXCL) — atomic, so no lock
    is needed and it is unaffected by fork. And pinning via CUDA_VISIBLE_DEVICES makes it impossible
    by design to create a context on another card. We verify with the visible device count rather than trusting the env var.
    """
    import itertools
    import tempfile
    d = Path(tempfile.gettempdir()) / f"ovul_gpu_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    for k in itertools.count():
        try:
            os.close(os.open(str(d / f"w{k}"), os.O_CREAT | os.O_EXCL))
            break
        except FileExistsError:
            continue
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k % len(gpus)])
    os.environ["OVUL_GPU"] = "0"          # once pinned, that card is index 0
    import torch
    assert torch.cuda.device_count() == 1, (
        f"worker sees {torch.cuda.device_count()} devices — not pinned to one "
        "card. Left as is it would eat every GPU.")
    _S["D"] = load()[1]


# Dump per-night predictions of the original fit (b=0). Read by the participant-level tests (scripts/model_tests_testcohort.py).
# Empty writes nothing — default behavior is unchanged.
_PD = os.environ.get("OVUL_DUMP_PRED_DIR", "").strip()


def dump_pred(b, lab, arm, nm, D, ite, y, pred):
    if not _PD or b != 0:
        return
    d = Path(_PD)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(idx=ite, g=D["g"][ite], y_true=y[ite], y_pred=pred)).to_csv(
        d / f"{lab}_{arm}_{nm}.csv", index=False)


def one(job):
    b, lab, arm, names, _gpu = job
    D = _S["D"]
    g, cd, cyc = D["g"], D["cd"], D["cyc"]
    nc, ovc = int(lab[1]), OV_CLS[lab]
    y = collapse(phase5(cd, D["ovt"]), lab)
    X = np.column_stack([D["Xp"], D["C"]]) if arm == "combined" else D["Xp"]
    ite = np.where(~D["is_A"])[0]
    pats = np.unique(g[D["is_A"]])
    rng = np.random.RandomState(SEED + b)       # same draw as bootstrap
    draw = pats if b == 0 else rng.choice(pats, len(pats), replace=True)
    itr = np.concatenate([np.where(g == p)[0] for p in draw])
    if len(np.unique(y[itr])) < nc:
        return None
    out = dict(boot=b, label=f"{nc}-class", arm=arm)
    for nm in names:
        try:
            m = MODELS[nm](random_state=SEED + b)
            m.set_structure(cyc[itr], cd[itr])
            m.fit(X[itr], y[itr])
            m.set_structure(cyc[ite], cd[ite])
            pred = np.asarray(m.predict(X[ite]))
            dump_pred(b, lab, arm, nm, D, ite, y, pred)
            out[f"{nm}_macro"] = macro(y[ite], pred, nc)
            out[f"{nm}_ov"] = ovf1(y[ite], pred, ovc)
        except Exception as e:
            out[f"{nm}_err"] = f"{type(e).__name__}: {e}"[:120]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--per-gpu", type=int, default=2,
                    help="max workers per card. One worker holds 6–7 GB")
    ap.add_argument("--gpus", default="", help="comma-separated list. Empty auto-selects cards with free memory")
    ap.add_argument("--models", default=",".join(ORDER))
    ap.add_argument("--labels", default=",".join(LABELS))
    a = ap.parse_args()

    names = [n for n in a.models.split(",") if n.strip()]
    labs = [l for l in a.labels.split(",") if l.strip()]
    assert all(n in MODELS for n in names), f"unknown model: {set(names) - set(MODELS)}"

    gpus = ([int(x) for x in a.gpus.split(",") if x.strip()] if a.gpus
            else free_gpus())
    bad = set(gpus) - set(ALLOWED_GPUS)
    assert not bad, (f"GPU {sorted(bad)} is not allowed for this project. "
                     f"Usable cards are only {list(ALLOWED_GPUS)}.")
    assert gpus, (f"Allowed cards {list(ALLOWED_GPUS)} are both busy. "
                  "Wait for them to free up — will not move to other cards.")

    d, D = load()
    audit(d, D)
    print(f"== SOTA {names} · replicates {a.n_boot} · GPU {gpus} · worker {a.workers}",
          flush=True)

    jobs = [(b, l, arm, names, gpus[j % len(gpus)])
            for j, (l, arm, b) in enumerate(
                (l, arm, b) for l in labs for arm in ("physio", "combined")
                for b in range(a.n_boot))]
    t0, rows = time.time(), []
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "sota.csv"
    # One worker holds 6–7 GB per card. How many to put on each card is derived backward
    # from free memory — this machine is shared, and using every free slot kills other people's jobs.
    workers = min(a.workers, len(gpus) * a.per_gpu)
    if workers < a.workers:
        print(f"  worker {a.workers} -> {workers} (GPU {len(gpus)} cards x per card "
              f"{a.per_gpu})", flush=True)
    tag = str(os.getpid())
    with ProcessPoolExecutor(max_workers=workers, initializer=init,
                             initargs=(gpus, tag)) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=1), 1):
            if r:
                rows.append(r)
            if i % 50 == 0 or i == len(jobs):
                # Checkpoint. This runs overnight, so progress up to here survives an interruption.
                pd.DataFrame(rows).to_csv(f, index=False)
                s = pd.DataFrame(rows)
                done = f"[{time.time()-t0:6.0f}s] {i}/{len(jobs)}"
                bits = []
                for nm in names:
                    c = f"{nm}_macro"
                    if c in s and s[c].notna().any():
                        bits.append(f"{nm}={s[c].mean():.3f}")
                    e = f"{nm}_err"
                    if e in s and s[e].notna().any():
                        bits.append(f"{nm}!{s[e].dropna().iloc[0][:40]}")
                print(done + "  " + " ".join(bits), flush=True)
    pd.DataFrame(rows).to_csv(f, index=False)
    print(f"-> {f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
