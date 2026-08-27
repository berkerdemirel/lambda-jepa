"""THE pull instrument: per-term trunk-gradient norms, realized shares, and force geometry
at a held training state. One script replacing the eleven bespoke bridges that accumulated
between E12 and E24 (archived 2026-08-07 under D-083's housekeeping; see
experiments/archive/README.md for the map from each retired script to this one).

What it measures. For each named term of a method's loss, the gradient of that term ALONE
w.r.t. the trunk parameters (`g_enc`, trunk-module-only — the standing convention since
e12h_pull), its realized weighted pull `w·g`, and its `share` of the total. Shares are the
dose coordinate: E24-T1's recipe is stated in them, and E19-T1 is the reason nominal weights
are never transplanted across a frame, an aug family, a dataset, or an architecture.

Two corrections this consolidation carries, both already established but never both present
in one script:

1. THE RING IS WARMED (D-072). A checkpoint does not store the conditioner ring (`extras()`
   is not overridden), so a freshly loaded method starts cold: the conditioner sees n = bs
   rows instead of (q+1)·bs, n/d′ drops to 1, its gradient inflates, and its share is
   overstated — the artifact behind E24-T2's retracted vm4 "z-dominance". We therefore step
   a SEQUENCE of batches and record `qfill` (rows in the ring at read time) and `warm`
   (qfill >= queue_steps) on every row. **Warm rows are the answer; cold rows are kept
   because batch 0 reproduces the historical convention exactly.** For a cell with
   queue_steps = 0 (including every OAS cell) every row is warm and one batch suffices.
2. METHOD-AGNOSTIC. Term→weight mapping is read from the method's own `PULL_W` (the same
   attribute train.py's share logger uses), not hardcoded. A method that has not declared
   `PULL_W` fails loudly rather than being measured against a guessed weight map.

Deliberately NOT ported from `e12h_pull.py`: its Hydra path, which built a method from
train.py overrides with no checkpoint in order to dose a config that did not exist yet.
That job is now done better by the in-training share logger — launch a short pilot and read
its `[share] ep1` line at a real formation state (the E27 dose procedure). Init-state pulls
remain available here via `state=init`, with E19-T1's caveat standing: a random trunk maps
different aug families to near-identical term values, so init pulls cannot bridge an aug axis.

`--set k=v` overrides the checkpoint's stored method cfg before measuring — the dose-bridge
job every retired script existed to do: hold a certified formation state fixed and swap ONE
axis (an aug family, a payment mode, a conditioner stream), then re-dose per term by
w'_i = s*_i · T / g_i. Values parse as JSON when possible, else stay strings.

Usage:
    python experiments/pull.py <label> <ckpt> <state: ckpt|init> [...]      # triples repeat
    python experiments/pull.py ... --batches 6 --no-cos
    python experiments/pull.py lab ck.pt ckpt --set cond_stream=globals
Appends -> results/diag/pull.csv. Numbers land RAW.
"""
import csv
import json
import os
import sys

import torch
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.utils.data import DataLoader

from sslgap.data import seed_everything
from sslgap.methods import METHODS
from sslgap.methods.base import Frame
from sslgap.paths import DIAG

OUT = DIAG / "pull.csv"
BS = 128          # house convention; shares are ratios, so this is fixed, not tuned


def _to_dev(x, dev):
    return (type(x)(_to_dev(t, dev) for t in x) if isinstance(x, (list, tuple))
            else x.to(dev, non_blocking=True))


def measure(label, ck_path, state, nb, want_cos, overrides=None, dev="cuda"):
    base = torch.load(ck_path, map_location="cpu", weights_only=False)
    mcfg = dict(base["cfg"]["method"])
    for k, v in (overrides or {}).items():
        try:
            mcfg[k] = json.loads(v)
        except json.JSONDecodeError:
            mcfg[k] = v
    fr = base["cfg"]["frame"]
    name = mcfg["name"]
    frame = Frame(name=fr["name"], model_name=fr["model_name"], img_size=fr["img_size"],
                  dataset=fr["dataset"], data_root=fr["data_root"], epochs=fr["epochs"],
                  seed=0, grad_clip=1.0, num_workers=0, device=dev)
    seed_everything(0)
    method = METHODS[name](OmegaConf.create(mcfg), frame)
    pull_w = getattr(method, "PULL_W", None)
    assert pull_w, (f"{name} declares no PULL_W — add the term->cfg-weight map to the method "
                    f"class before measuring it (train.py's share logger needs it too)")
    modules = method.build_modules().to(dev)
    if state == "ckpt":
        for role, sd in base["modules"].items():
            if role != "probe":
                modules[role].load_state_dict(sd)
        method.load_extras(base.get("extras", {}))
    loader = DataLoader(method.build_train_dataset(), batch_size=BS, shuffle=True,
                        drop_last=True, num_workers=0,
                        generator=torch.Generator().manual_seed(0))
    params = [p for p in modules["backbone"].parameters() if p.requires_grad]
    q = int(mcfg.get("queue_steps", 0) or 0)
    anat = {"zb": mcfg.get("z_floor_batch", "pooled"), "hb": mcfg.get("h_floor_batch", "pooled"),
            "q": q, "shrink": mcfg.get("floor_shrink"),
            "cond_stream": mcfg.get("cond_stream") or "all"}
    ep = base.get("epoch", -1) if state == "ckpt" else -1
    it, rows = iter(loader), []
    for b in range(nb):
        qfill = len(getattr(method, "_zq", []))
        views, y = next(it)
        with autocast(dev, dtype=torch.bfloat16):
            terms, _, _ = method.training_step(modules, _to_dev(views, dev), dev,
                                               y=y.to(dev))
        names = [k for k in terms if k != "loss"]
        vec = {}
        for i, k in enumerate(names):
            g = torch.autograd.grad(terms[k], params, retain_graph=i < len(names) - 1,
                                    allow_unused=True)
            vec[k] = torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1)
                                for x, p in zip(g, params)]).float()
        gs = {k: vec[k].norm().item() for k in names}
        # PULL_W values: cfg-key string, or a literal number for weights fixed by the
        # loss code itself (dino/byol main terms = 1 by construction; 2026-08-25 — old
        # ckpts predate any cfg key for them; the weight is a property of the code)
        w = {k: (float(pull_w[k]) if isinstance(pull_w[k], (int, float))
                 else float(mcfg[pull_w[k]])) for k in names}
        tot = sum(w[k] * gs[k] for k in names) or 1.0
        # f64 dot: fp32 over ~21M dims drifts ~.5% (the e21_vmcos control lesson)
        cos = {}
        if want_cos:
            for i, a in enumerate(names):
                for c in names[i + 1:]:
                    va, vc = vec[a].double(), vec[c].double()
                    cos[f"cos_{a}_{c}"] = round(float(va @ vc / (va.norm() * vc.norm())), 4)
        del vec
        for k in names:
            rows.append({"label": label, "method": name, "state": state,
                         "ckpt": os.path.basename(ck_path), "ep": ep,
                         "aug": mcfg.get("aug", "byol"), "V": mcfg.get("V", 2), **anat,
                         "batch": b, "qfill": qfill, "warm": int(qfill >= q),
                         "term": k, "g_enc": round(gs[k], 6), "w": w[k],
                         "wg": round(w[k] * gs[k], 4), "share": round(w[k] * gs[k] / tot, 4),
                         "wg_total": round(tot, 4), "loss_value": round(float(terms[k]), 6),
                         **cos, "slurm_job": os.environ.get("SLURM_JOB_ID", "")})
        print(f"{label:14s} {state:4s} b{b} qfill={qfill}{'(warm)' if qfill >= q else '(COLD)'} "
              + " ".join(f"{k}={w[k] * gs[k] / tot:.3f}(g{gs[k]:.3f})" for k in names)
              + ("" if not cos else " | " + " ".join(f"{k}={v:+.3f}" for k, v in cos.items())),
              flush=True)
    return rows


def main():
    args = [a for a in sys.argv[1:]]
    nb, want_cos, over = 6, True, {}
    while "--set" in args:
        i = args.index("--set")
        k, v = args[i + 1].split("=", 1)
        over[k] = v
        del args[i:i + 2]
    if "--no-cos" in args:
        args.remove("--no-cos")
        want_cos = False
    if "--batches" in args:
        i = args.index("--batches")
        nb = int(args[i + 1])
        del args[i:i + 2]
    assert args and len(args) % 3 == 0, "triples: <label> <ckpt> <ckpt|init>"
    rows = []
    for i in range(0, len(args), 3):
        rows += measure(args[i], args[i + 1], args[i + 2], nb, want_cos, over)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Non-floorssl methods carry method-specific pairwise-cos columns; per the schema
    # guard's own policy ("write to a new path — never append mismatched rows"), each
    # method gets its own stable-schema file; floorssl keeps the legacy path/history.
    meth = rows[0].get("method")
    out = OUT if meth == "floorssl" else OUT.with_name(f"{OUT.stem}_{meth}{OUT.suffix}")
    fields = list(rows[0])
    # Appending a WIDER schema to an existing CSV silently shifts every column of the new
    # rows (caught 2026-08-07 the first time `--set` added `cond_stream`). Refuse instead:
    # a results file that reads fine and means something else is the worst failure mode here.
    if out.exists():
        with open(out, newline="") as f:
            have = next(csv.reader(f), [])
        assert have == fields, (
            f"schema mismatch on {out}\n  file: {have}\n  rows: {fields}\n"
            f"Migrate the file (add the missing columns with their legacy defaults) or write "
            f"to a new path — never append mismatched rows.")
    with open(out, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        if not out.stat().st_size:
            wr.writeheader()
        wr.writerows(rows)
    print("appended", OUT, flush=True)


if __name__ == "__main__":
    main()
