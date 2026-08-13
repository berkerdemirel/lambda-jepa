"""Selftest gate for sslgap/metrics/twospace.py (house rule: selftests gate runs).
Synthetic Gaussian world with known B, A, so every estimator and both theorem laws
have exact targets: (1) Â/B̂/B̂x/B̂pool recovery incl. the A/V term, (2) Θ recovery,
(3) the fidelity law λ/(1+λ)+λ/V_c predicted-vs-observed, (4) accessibility R² on an
exactly-linear bridge (up to the finite-view noise floor) and its monotone drop under
added target noise, (5) organization: in-span targets have small d̂ and the
decomposition identity holds, (6) G/S: linear closed form matches, PSD sanity.
Run: python experiments/twospace_selftest.py   (seconds, CPU)."""
import sys

import numpy as np

from sslgap.metrics import twospace as tw

rng = np.random.default_rng(0)
N, V, DH, DZ, CLS = 6000, 8, 24, 16, 10
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


# world: h-centers ~ N(0, B0), views add N(0, A0) noise; z = exact linear map of h
b0 = np.linspace(2.0, 0.5, DH)
B0 = np.diag(b0)
Qa = np.linalg.qr(rng.normal(size=(DH, DH)))[0]
A0 = Qa @ np.diag(np.linspace(1.5, 0.1, DH)) @ Qa.T
Ah = np.linalg.cholesky(A0)
y = rng.integers(0, CLS, N)
cls_means = rng.normal(size=(CLS, DH)) * 1.2
mh = cls_means[y] * 0.6 + rng.normal(size=(N, DH)) @ np.sqrt(np.diag(b0 * 0.4))
mh += rng.normal(size=(N, DH)) @ np.sqrt(np.diag(b0 * 0.1))
h_views = [mh + rng.normal(size=(N, DH)) @ Ah.T for _ in range(V)]

print("(1) cloud moments")
mom = tw.cloud_moments(h_views)
Btrue = np.cov(mh, rowvar=False)
rel = lambda X, Y: np.linalg.norm(X - Y) / np.linalg.norm(Y)
check("A_hat", rel(mom["A"], A0) < 0.05, f"rel={rel(mom['A'], A0):.4f}")
check("B_hat_debiased", rel(mom["B"], Btrue) < 0.05, f"rel={rel(mom['B'], Btrue):.4f}")
check("B_cross_group", rel(mom["Bx"], Btrue) < 0.05, f"rel={rel(mom['Bx'], Btrue):.4f}")
check("Bpool=B+A/V", rel(mom["Bpool"], Btrue + A0 / V) < 0.05,
      f"rel={rel(mom['Bpool'], Btrue + A0 / V):.4f}")

print("(2) theta")
ts = tw.theta_spectrum(mom["A"], mom["B"], floor_frac=1e-4)
Rt, _, _ = tw.whiten_map(Btrue, 1e-4)
lam_true = np.linalg.eigvalsh(Rt @ A0 @ Rt.T)[::-1]
check("theta_spectrum", rel(ts["lam"], lam_true) < 0.08,
      f"rel={rel(ts['lam'], lam_true):.4f} r={ts['r']}")

print("(3) fidelity law")
fs, fr, ctx = tw.center_fidelity(h_views, holdout=0.5, floor_frac=1e-4, seed=0)
check("law corr", fs["resid_corr"] > 0.95, f"corr={fs['resid_corr']:.4f}")
check("law ratio", abs(fs["resid_med_ratio"] - 1) < 0.15,
      f"med_ratio={fs['resid_med_ratio']:.4f}")
check("W=(I+Theta)^-1", fs["w_law_relerr"] < 0.15, f"relerr={fs['w_law_relerr']:.4f}")

print("(4) accessibility")
W0 = rng.normal(size=(DH, DZ))
mz_exact = mh @ W0
Az0 = np.eye(DZ) * 0.3
z_views = [mz_exact + rng.normal(size=(N, DZ)) * np.sqrt(0.3) for _ in range(V)]
mz_est = np.mean(z_views, axis=0)
acc, _ = tw.accessibility(mh, mz_est, Az=Az0, V=V, holdout=0.5, seed=0)
check("exact bridge R2", acc["r2_acc_holdout"] > 1 - acc["noise_floor"] - 0.03,
      f"R2={acc['r2_acc_holdout']:.4f} floor={acc['noise_floor']:.4f}")
mz_noisy = mz_exact + rng.normal(size=(N, DZ)) * 1.0
acc2, _ = tw.accessibility(mh, mz_noisy, holdout=0.5, seed=0)
check("noisy bridge drops", acc2["r2_acc_holdout"] < acc["r2_acc_holdout"] - 0.1,
      f"R2={acc2['r2_acc_holdout']:.4f}")
check("holdout<=insample", acc["r2_acc_holdout"] <= acc["r2_acc_insample"] + 0.01)

print("(5) organization")
og, rows = tw.organization(h_views, mz_est, y, holdout=0.5, floor_frac=1e-4, seed=0)
check("in-span d_h < trivial", og["d_h_med"] < 0.9, f"d_h_med={og['d_h_med']:.4f}")
check("decomp corr", og["decomp_corr"] > 0.9, f"corr={og['decomp_corr']:.4f}")
check("decomp ratio", abs(og["decomp_med_ratio"] - 1) < 0.2,
      f"med_ratio={og['decomp_med_ratio']:.4f}")
check("transfer bound", og["transfer_viol_frac"] < 0.2,
      f"viol={og['transfer_viol_frac']:.4f}")

print("(6) G/S split")
gs = tw.gs_split(ctx["ts"], ctx["Sobs"], ctx["Vc"])
check("S_lb<=theta_tr", gs["S_lb_tr"] <= gs["theta_tr"] + 1e-9)
check("linear~closed-form", abs(gs["S_lb_tr"] - gs["S_lb_tr_linclosed"])
      / max(gs["S_lb_tr_linclosed"], 1e-9) < 0.25,
      f"emp={gs['S_lb_tr']:.4f} closed={gs['S_lb_tr_linclosed']:.4f}")
check("G_ub_tr>=0", gs["G_ub_tr"] >= 0)

print(f"{'ALL GREEN' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
