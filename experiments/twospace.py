"""Two-space estimators over the 8-view stores: capacity, the Theta spectrum, per-class organization vs view-sensitivity (results/twospace/)."""
import os
import re

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from sslgap.extract import FeatureStore
from sslgap.metrics import twospace as tw

def _views(store, run_id, man, space, V, idx=None):
    out = []
    for k in range(V):
        X = np.asarray(store.get(run_id, man, f"{space}.view{k}"), dtype=np.float64)
        out.append(X[idx] if idx is not None else X)
    return out

@hydra.main(version_base=None, config_path="configs", config_name="twospace")
def main(cfg: DictConfig):
    store = FeatureStore(cfg.store_root)
    run_dir = os.path.join(os.path.expanduser(cfg.store_root), cfg.run_id)
    mans = [m for m in sorted(os.listdir(run_dir)) if re.search(r"\.o\d+$", m)]
    if cfg.get("manifest"):
        mans = [m for m in mans if m == cfg.manifest]
    out_dir = os.path.join(cfg.results_root, "twospace")
    os.makedirs(out_dir, exist_ok=True)
    S, P, C = [], [], []

    for man in mans:
        meta = store.meta(cfg.run_id, man)
        V = int(meta["v"])
        bases = sorted({s.rsplit(".view", 1)[0] for s in store.spaces(cfg.run_id, man)})
        if cfg.get("max_tap_d"):
            wide = [b for b in bases
                    if store.get(cfg.run_id, man, f"{b}.view0").shape[1] > cfg.max_tap_d]
            if wide:
                print(f"[twospace] {man}: skipping {len(wide)} taps wider than "
                      f"{cfg.max_tap_d}: {wide}")
            bases = [b for b in bases if b not in wide]
        h_ref = (meta.get("ckpt_provenance") or {}).get("h_space")
        branch = h_ref.split(".h.")[0] if h_ref else bases[0].split(".h.")[0]
        if not h_ref or h_ref not in bases:
            h_ref = f"{branch}.h.cls" if f"{branch}.h.cls" in bases else f"{branch}.h.gap"
        z_taps = [b for b in bases if ".z." in b]
        h_taps = [h_ref] + ([b for b in bases if re.search(r"\.h\..*\.L\d+$", b)]
                            if cfg.h_layers else [])
        z_prim = cfg.get("z_primary") or next(
            (z for z in z_taps if z.endswith(".z.proj.out")), z_taps[0] if z_taps else None)
        idx = None
        N = int(meta["n"])
        if cfg.max_n and N > cfg.max_n:
            idx = np.sort(np.random.default_rng(cfg.seed).choice(N, cfg.max_n, replace=False))
        labels = store.labels(cfg.run_id, man)
        labels = labels[idx] if idx is not None else labels

        cache = {}
        def views_of(space):
            if space not in cache:
                cache.clear()
                cache[space] = _views(store, cfg.run_id, man, space, V, idx)
            return cache[space]

        centers, mom_of = {}, {}
        for sp in dict.fromkeys(h_taps + z_taps):
            vw = views_of(sp)
            mom = tw.cloud_moments(vw)
            ts = tw.theta_spectrum(mom["A"], mom["B"], cfg.floor_frac)
            centers[sp], mom_of[sp] = mom["m"], {"A": mom["A"]}
            print(f"[twospace] {man} {sp}: r={ts['r']} trΘ/r={ts['lam'].mean():.3f}")
            S.append({"manifest": man, "kind": "capacity", "space": sp,
                      "b_rank": ts["r"], "b_top": float(ts["b_eigs"][0]),
                      "theta_tr_over_r": float(ts["lam"].mean()),
                      "theta_op": float(ts["lam"][0]),
                      "theta_med": float(np.median(ts["lam"])),
                      "a_tr": float(np.trace(mom["A"])),
                      "b_tr": float(np.trace(mom["B"])),
                      "bx_rel_gap": float(np.linalg.norm(mom["Bx"] - mom["B"]) /
                                          np.linalg.norm(mom["B"]))})
            P += [{"manifest": man, "kind": "theta", "space": sp, "split": "",
                   "j": j, "value": float(l)} for j, l in enumerate(ts["lam"])]
            fs, fr, ctx = tw.center_fidelity(vw, cfg.holdout, cfg.floor_frac, cfg.seed)
            S.append({"manifest": man, "kind": "fidelity", "space": sp, **fs})
            P += [{"manifest": man, "kind": "fidelity", "space": sp, "split": "", **r}
                  for r in fr]
            gs = tw.gs_split(ctx["ts"], ctx["Sobs"], ctx["Vc"])
            S.append({"manifest": man, "kind": "gs_linear", "space": sp, **gs})
            if cfg.do_mlp and sp == h_ref:
                Sm, mdiag = tw.mlp_center_residuals(ctx, seed=cfg.seed)
                gm = tw.gs_split(ctx["ts"], Sm, ctx["Vc"])
                S.append({"manifest": man, "kind": "gs_mlp", "space": sp, **gm, **mdiag})

        for hs in h_taps:
            for zs in z_taps:
                acc, eigs = tw.accessibility(
                    centers[hs], centers[zs], Az=mom_of[zs]["A"], V=V,
                    holdout=cfg.holdout, floor_frac=cfg.floor_frac, seed=cfg.seed)
                S.append({"manifest": man, "kind": "accessibility",
                          "space": f"{hs}~{zs}", **acc})
                P += [{"manifest": man, "kind": "sigma_c", "space": f"{hs}~{zs}",
                       "split": split, "j": j, "value": float(v)}
                      for split, ev in eigs.items() for j, v in enumerate(ev)]
        if z_prim:
            og, rows = tw.organization(views_of(h_ref), centers[z_prim], labels,
                                       cfg.holdout, cfg.floor_frac, cfg.seed)
            S.append({"manifest": man, "kind": "organization",
                      "space": f"{h_ref}~{z_prim}", **og})
            C += [{"manifest": man, "space": f"{h_ref}~{z_prim}", **r} for r in rows]

    for frames, suff in ((S, ""), (P, ".spectra"), (C, ".classes")):
        if frames:
            df = pd.DataFrame(frames)
            df.insert(0, "run_id", cfg.run_id)
            df.to_csv(os.path.join(out_dir, f"{cfg.run_id}{suff}.csv"), index=False)
    print(f"[twospace] done: {cfg.run_id} -> {out_dir}")

if __name__ == "__main__":
    main()
