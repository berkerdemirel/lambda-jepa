"""Converged, unaugmented linear read of stored IN-1k CLS features (Berker 2026-09-15: "we can do a lbfgs like fitting ... i am
curious how much we gain from augmentations from linear probe"). The bench (experiments/bench_probe.py, bench_linear_v1) is the
Lightly MAE recipe on the checkpoint: BatchNorm + Linear head, LARS, 90 epochs, RandomResizedCrop + HFlip on every step, max over
epochs. This driver reads the feature store instead — one centre view per image, `student.h.cls` of the extract's in1k.train.v1L /
in1k.val.v1L manifests — standardizes on the train split and fits ridge multinomial logistic regression to its optimum with
`sslgap.probes.linear.linear_lbfgs_v1` (the E36 reader) at a small ridge grid. bench − lbfgs = augmentation + head + schedule, jointly.
Usage: python experiments/lbfgs_linear.py <run_id> [<run_id> ...]   -> results/diag/lbfgs_linear.csv (appended) + printed rows."""
import csv, os, sys, time
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sslgap.probes.linear import linear_lbfgs_v1  # noqa: E402

LAMS = (1e-5, 1e-4, 1e-3)


def main():
    out = f"{ROOT}/results/diag/lbfgs_linear.csv"; new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["run_id", "space", "lam", "train_acc", "val_acc", "grad_norm", "n_train", "n_val", "d"])
        for rid in sys.argv[1:]:
            t0 = time.time(); base = f"{ROOT}/features/{rid}"
            Xtr = np.load(f"{base}/in1k.train.v1L/student.h.cls.npy").astype(np.float32); ytr = np.load(f"{base}/in1k.train.v1L/labels.npy")
            Xva = np.load(f"{base}/in1k.val.v1L/student.h.cls.npy").astype(np.float32); yva = np.load(f"{base}/in1k.val.v1L/labels.npy")
            mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
            Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
            print(f"[lbfgs] {rid}: train {Xtr.shape} val {Xva.shape} loaded in {time.time() - t0:.0f}s", flush=True)
            for lam in LAMS:
                t1 = time.time()
                r = linear_lbfgs_v1(Xtr, ytr, Xva, yva, num_classes=int(ytr.max()) + 1, lam=lam, max_iter=500, device="cuda")
                print(f"[lbfgs] {rid} lam={lam:g}: train {100 * r['train_acc']:.2f} val {100 * r['val_acc']:.2f} grad {r['grad_norm']:.1e} ({(time.time() - t1) / 60:.1f} min)", flush=True)
                wr.writerow([rid, "student.h.cls", lam, f"{100 * r['train_acc']:.3f}", f"{100 * r['val_acc']:.3f}", f"{r['grad_norm']:.2e}", len(ytr), len(yva), Xtr.shape[1]]); fh.flush()
                torch.cuda.empty_cache()
    print("LBFGS_LINEAR_DONE", flush=True)


if __name__ == "__main__":
    main()
