"""The paper's two loss-weight rules as arithmetic (Appendix "Setting the loss weights").

Both rules balance terms by their realized pull on the backbone, w * g, where g is the norm of the
gradient of the UNWEIGHTED term with respect to the backbone parameters on a fixed batch. The g's
are measured, not tuned: for an existing method by `experiments/pull.py` at its epoch-25
checkpoint; for lambda-JEPA by the in-training share logger (`share_log_every=1`), whose
`[share] ep1` line of a one-epoch pilot prints g per term.

  python experiments/loss_weights.py add G_SSL              # backbone term for an existing method
  python experiments/loss_weights.py own G_INV G_Z G_H      # the three lambda-JEPA weights

Worked values from the paper: `own 0.809 0.137 0.580` -> 31.07 / 225.50 / 1.671 (ViT-S);
`own 1.102 0.139 0.239` -> 22.81 / 222.26 / 4.054 (ViT-B).
"""
import sys

T, G_R = 0.06, 0.30
TOTAL = 57.0
SHARES = (0.441, 0.542, 0.017)

def beta_h(g_ssl):
    return T / (1 - T) * g_ssl / G_R

def own(g_inv, g_z, g_h):
    return tuple(TOTAL * s / g for s, g in zip(SHARES, (g_inv, g_z, g_h)))

if __name__ == "__main__":
    mode, args = sys.argv[1], [float(a) for a in sys.argv[2:]]
    if mode == "add":
        print(f"beta_h = {beta_h(*args):.4g}")
    elif mode == "own":
        w = own(*args)
        print(f"w_inv = {w[0]:.2f}  beta_z = {w[1]:.2f}  beta_h = {w[2]:.3f}")
    else:
        sys.exit(__doc__)
