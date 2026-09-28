"""Two-layer linear network (N_i = N_h = N_o = 8, hierarchical targets) trained by full-batch gradient descent with weight decay lambda_L2 and the encoder log-determinant term lambda_det log det(W_1); draws the block Gram matrix Q of the regularized network next to the theoretical Q at lambda_bal = lambda_det / lambda_L2 and the Q of a network initialized at that balance and trained without regularization (figures/)."""
import numpy as np
import seaborn as sns

from utils import get_random_regression_task, get_lambda_balanced, get_ntk, reshape_matrix, kernel_distance, cosine_similarity,BlindColours
import matplotlib.pyplot as plt

from networks.linear_network import LinearNetworkreg, LinearNetworkRepNorm

# ── Encoder-regularised variant of test_reg.py ───────────────────────────────
# test_reg.py regularises the DECODER: R = lambda_reg/2*(||W1||^2+||W2||^2)
#                                          + lambda_det * log det(W2)
# with fixed point  Lambda* = -(lambda_det/lambda_reg) * I.
#
# This script instead matches the paper's SACReg (Theorem: encoder-variational
# balance): the log-det term acts on the ENCODER Gram matrix W1 W1^T,
#   R_enc = lambda_reg/2*(||W1||^2+||W2||^2) - gamma/2 * log det(W1 W1^T)
# For a square W1, log det(W1 W1^T) = 2 log det(W1), so with the code's
# LinearNetworkreg convention (which adds +lambda_det_W1 * log det(W1) to the
# loss), lambda_det_W1 = -gamma.
#
# Redoing the imbalance dynamics with the det term on W1 instead of W2 flips
# the sign of that term's contribution to Lambda = W2^T W2 - W1 W1^T:
#   d(W1 W1^T)/dt |_det = -2*lambda_det_W1 * I   (vs. d(W2^T W2)/dt |_det = -2*lambda_det_W2*I before)
#   => dLambda/dt = -2*lambda_reg*Lambda + 2*lambda_det_W1 * I
#   => Lambda* = +(lambda_det_W1 / lambda_reg) * I     (sign flipped vs. decoder case)
# For lambda_det_W1 = -gamma < 0 (gamma>0, as required by the theorem), this
# gives Lambda* = -gamma/lambda_reg < 0 — a NEGATIVE balance, matching the
# paper's anti-collapse guarantee on the encoder.

bc = BlindColours(False)
blind_colours = bc.get_colours()
div_cmap = bc.get_div_cmap()
colour_steps = bc.get_colour_steps()
learning_rate = 0.001
training_steps = 30000
#Run model, keep track of w2, 1, log it into all of these lists
np.random.seed(1)
in_dim = 8
hidden_dim = 8
out_dim = 8

batch_size= 8
X, Y = get_random_regression_task(batch_size, in_dim, out_dim,Whiten=False)
X = np.eye(8) * np.sqrt(8)
Y = np.asarray([
            [1.,  1.,  1., -0.,  1., -0., -0., -0.],
            [1.,  1.,  1., -0., -1., -0., -0., -0.],
            [1.,  1., -1., -0., -0.,  1., -0., -0.],
            [1.,  1., -1.,  0, -0., -1.,  0., -0.],
            [1., -1., -0.,  1., -0., -0.,  1., -0.],
            [1., -1., -0.,  1., -0., -0., -1., -0.],
            [1., -1., -0., -1., -0., -0., -0.,  1.],
            [1., -1., -0., -1., -0., -0., -0.,  -1.]
        ])
Y =  Y/np.sqrt(8)

# Theoretical target
sigma_xx  = 1.0 / batch_size * (X @ X.T)
sigma_yx  = 1.0 / batch_size * (Y @ X.T)
W_theory  = sigma_yx @ np.linalg.inv(sigma_xx)

# --- Baseline: lambda-balanced init (aligned to data), NO regularisation ---
# This is the reference that should converge cleanly to W_theory.
np.random.seed(42)
_w1_bal, _w2_bal = get_lambda_balanced(0, in_dim, hidden_dim, out_dim, sigma_yx=sigma_yx)
_model_bal = LinearNetworkreg(in_dim, hidden_dim, out_dim,
                              lambda_reg=0, lambda_det_W1=0, lambda_det_W2=0,
                              init_w1=_w1_bal.copy(), init_w2=_w2_bal.copy())
_w1s_bal, _w2s_bal, _losses_bal = _model_bal.train(X, Y, training_steps, learning_rate)
ws_balanced    = np.expand_dims([w2 @ w1 for w2, w1 in zip(_w2s_bal, _w1s_bal)], axis=1)
losses_balanced = [1/(2*batch_size) * np.linalg.norm(w @ X - Y)**2 for w in ws_balanced]
print(f"Balanced baseline final loss: {losses_balanced[-1]:.6f}")

lmdas = [0, -0.2]       # lambda_det_W1 values (encoder log-det coefficient)
lmdas_reg = [0,0.1,]
colours = [blind_colours[i+1] for i in range(len(lmdas))]
initial_weight_pairs = {}
initial_weight_pairs = {}
output_empirical = {}
ws_empirical = {}
w1w1s_empirical = {}
w2w2s_empirical = {}
ntks_empirical = {}
losses_list_empirical = {}
kernel_distance_ntk = {}
kernel_distance_ntk_end = {}
kernel_distance_w1w1_end = {}
kernel_distance_w2w2_end = {}
lambda_empirical = {}
# Convergence threshold (example value, adjust as necessary)
convergence_threshold = 0.00099
convergence_steps = []


for lmda in lmdas:
    for lmda_reg in lmdas_reg:
        sigma = 0.5
        init_w1= np.random.normal(0., sigma, (hidden_dim, in_dim))
        init_w2 = np.random.normal(0., sigma, (out_dim, hidden_dim))

        # Encoder-regularised: log-det term now on W1, not W2.
        model = LinearNetworkreg(in_dim, hidden_dim, out_dim, lambda_reg= lmda_reg, lambda_det_W1 = lmda, lambda_det_W2  = 0, init_w1=init_w1.copy(),  init_w2=init_w2.copy())
        w1s, w2s, losses = model.train(X, Y, training_steps, learning_rate)

        ws_empirical[lmda_reg,lmda] = [w2 @ w1 for (w2, w1) in zip(w2s, w1s)]
        ws_empirical[lmda_reg,lmda] = np.expand_dims(ws_empirical[lmda_reg,lmda], axis=1)

        w1w1s_empirical[lmda_reg,lmda] = np.array([w1.T @ w1 for w1 in w1s])
        w2w2s_empirical[lmda_reg,lmda] = np.array([w2 @ w2.T for w2 in w2s])
        lambda_empirical[lmda_reg, lmda] = np.array([
            w2.T @ w2 - w1 @ w1.T
            for w2, w1 in zip(w2s, w1s)
        ])

        losses_list_empirical[lmda_reg,lmda] = [1 / (2 * batch_size) * np.linalg.norm(w @ X - Y) ** 2 for w in ws_empirical[lmda_reg,lmda]]
        ntks_empirical[lmda_reg,lmda] = np.array(
            [get_ntk(w1w1, w2w2, X, out_dim) for (w1w1, w2w2) in zip(w1w1s_empirical[lmda_reg,lmda], w2w2s_empirical[lmda_reg,lmda])])


        converged = False
        for step, loss in enumerate(losses):
            if loss < convergence_threshold:
                print(loss)
                convergence_steps.append(step)
                converged = True
                break
        if not converged:
            convergence_steps.append(training_steps)
        print(losses_list_empirical[lmda_reg,lmda][-1])

# ── Diagnostics: conservation-law check ──────────────────────────────────────
# For LinearNetworkreg with the log-det term on W1 (encoder), the balance
# Lambda = W2^TW2 - W1W1^T evolves as:
#   dLambda/dt = -2*lambda_reg * Lambda + 2*lambda_det_W1 * I
# Fixed-point (if lambda_reg > 0):  Lambda* = (lambda_det_W1 / lambda_reg) * I
# For lambda_reg=0: no fixed point — balance drifts by -2*lambda_det_W1 per unit time.
print("\n=== Conservation-law diagnostics (encoder log-det) ===")
for lmda in lmdas:
    for lmda_reg in lmdas_reg:
        key = (lmda_reg, lmda)
        lam_final = lambda_empirical[key][-1]
        eigs = np.linalg.eigvalsh(lam_final)
        fp = "no fixed pt" if lmda_reg == 0 else f"Λ*≈{lmda/lmda_reg:.1f}·I"
        print(f"  λ_det_W1={lmda}, λ_reg={lmda_reg}: "
              f"final Λ eigs ∈ [{eigs.min():.3f}, {eigs.max():.3f}]  ({fp})")
print()

# Lambda-balanced baselines — one per lmda value, no regularisation
# Maps regulariser lmda (=lambda_det_W1) → the lambda used for the balanced init
# (e.g. lmda_det_W1=-0.2, lmda_reg=0.1 → balanced init with lambda=-2, since
#  +λ_det_W1/λ_reg = -2 — a NEGATIVE balance, as required for the encoder
#  anti-collapse guarantee)
lmda_to_bal_init = {0: 0, -0.2: -2}

qqt_balanced_by_lmda = {}
np.random.seed(42)
for _lmda in lmdas:
    _bal_lmda = lmda_to_bal_init.get(_lmda, _lmda)
    _w1_b, _w2_b = get_lambda_balanced(_bal_lmda, in_dim, hidden_dim, out_dim, sigma_yx=sigma_yx)
    _model_b = LinearNetworkreg(in_dim, hidden_dim, out_dim,
                                lambda_reg=0, lambda_det_W1=0, lambda_det_W2=0,
                                init_w1=_w1_b.copy(), init_w2=_w2_b.copy())
    _w1s_b, _w2s_b, _ = _model_b.train(X, Y, training_steps, learning_rate)
    _w1_bf, _w2_bf = _w1s_b[-1], _w2s_b[-1]
    _ws_bf = _w2_bf @ _w1_bf
    qqt_balanced_by_lmda[_lmda] = np.vstack([
        np.hstack([_w1_bf.T @ _w1_bf, _ws_bf.T]),
        np.hstack([_ws_bf, _w2_bf @ _w2_bf.T])
    ])

# Analytical theory block Gram matrices — same closed form as test_reg.py
# (S1^2 = S_lam - lam/2, S2^2 = S_lam + lam/2 depend only on lam = Lambda's
# eigenvalue, regardless of which layer produced that balance):
U_th, S_th, Vt_th = np.linalg.svd(W_theory, full_matrices=True)

qqt_theory_by_lmda = {}
for _lmda in lmdas:
    lam = lmda_to_bal_init[_lmda]
    S_lam = np.sqrt(S_th**2 + lam**2 / 4)
    S1_sq = S_lam - lam / 2   # W1^T W1 eigenvalues
    S2_sq = S_lam + lam / 2   # W2 W2^T eigenvalues
    W1W1T_th = Vt_th.T @ np.diag(S1_sq) @ Vt_th
    W2W2T_th = U_th @ np.diag(S2_sq) @ U_th.T
    qqt_theory_by_lmda[_lmda] = np.vstack([
        np.hstack([W1W1T_th, W_theory.T]),
        np.hstack([W_theory, W2W2T_th])
    ])

# ── Diagnostics: does balanced baseline match theory? ─────────────────────────
print("=== Theory vs balanced baseline (should match if network converged) ===")
for _lmda in lmdas:
    lam = lmda_to_bal_init[_lmda]
    err = np.linalg.norm(qqt_balanced_by_lmda[_lmda] - qqt_theory_by_lmda[_lmda], 'fro')
    ref = np.linalg.norm(qqt_theory_by_lmda[_lmda], 'fro')
    print(f"  lmda_det_W1={_lmda} (λ_bal={lam}): ‖balanced − theory‖/‖theory‖ = {err/ref:.4f}")

print("\n=== Theory vs empirical reg (should mismatch for LinearNetworkreg) ===")
for lmda in lmdas:
    for lmda_reg in lmdas_reg:
        key = (lmda_reg, lmda)
        qqt_emp = np.vstack([
            np.hstack([w1w1s_empirical[key][-1], ws_empirical[key][-1][0].T]),
            np.hstack([ws_empirical[key][-1][0], w2w2s_empirical[key][-1]])
        ])
        err = np.linalg.norm(qqt_emp - qqt_theory_by_lmda[lmda], 'fro')
        ref = np.linalg.norm(qqt_theory_by_lmda[lmda], 'fro')
        print(f"  λ_det_W1={lmda}, λ_reg={lmda_reg}: ‖emp − theory‖/‖theory‖ = {err/ref:.4f}")
print()

n_rows = len(lmdas_reg)
n_cols = 3 * len(lmdas)
fig, axs = plt.subplots(n_rows, n_cols,
                        figsize=(3.5 * n_cols, 3.2 * n_rows),
                        sharex=True, sharey=True)

for i, label_reg in enumerate(lmdas_reg):
    for j, label_det in enumerate(lmdas):
        col_emp = 3 * j
        col_th  = 3 * j + 1
        col_bal = 3 * j + 2
        key = (label_reg, label_det)
        lam_label = lmda_to_bal_init[label_det]
        lam_info = f'$\\lambda_{{L2}}={label_reg},\\ \\lambda_{{det}}^{{W_1}}={label_det}$'

        qqt_emp = np.vstack([
            np.hstack([w1w1s_empirical[key][-1], ws_empirical[key][-1][0].T]),
            np.hstack([ws_empirical[key][-1][0], w2w2s_empirical[key][-1]])
        ])
        err = (np.linalg.norm(qqt_emp - qqt_theory_by_lmda[label_det], 'fro') /
               np.linalg.norm(qqt_theory_by_lmda[label_det], 'fro'))

        sns.heatmap(1./8.*qqt_emp, cmap=div_cmap, ax=axs[i, col_emp],
                    cbar=True, cbar_kws={"shrink": 0.6})
        axs[i, col_emp].set_xticks([]); axs[i, col_emp].set_yticks([])
        axs[i, col_emp].set_title(f'empirical\n{lam_info}', fontsize=8, pad=4)
        axs[i, col_emp].text(0.97, 0.03, f'$\\varepsilon$={err:.2f}',
                             transform=axs[i, col_emp].transAxes,
                             ha='right', va='bottom', fontsize=8,
                             color='white', fontweight='bold')

        sns.heatmap(1./8.*qqt_theory_by_lmda[label_det], cmap=div_cmap, ax=axs[i, col_th],
                    cbar=True, cbar_kws={"shrink": 0.6})
        axs[i, col_th].set_xticks([]); axs[i, col_th].set_yticks([])
        axs[i, col_th].set_title(f'theory  ($\\lambda_{{bal}}={lam_label}$)\n{lam_info}', fontsize=8, pad=4)

        sns.heatmap(1./8.*qqt_balanced_by_lmda[label_det], cmap=div_cmap,
                    ax=axs[i, col_bal],
                    cbar=True, cbar_kws={"shrink": 0.6})
        axs[i, col_bal].set_xticks([]); axs[i, col_bal].set_yticks([])
        axs[i, col_bal].set_title(f'balanced, no reg  ($\\lambda_{{bal}}={lam_label}$)\n{lam_info}', fontsize=8, pad=4)

if len(lmdas) > 1:
    sep_x = 3 / n_cols
    fig.add_artist(plt.Line2D([sep_x, sep_x], [0.02, 0.98],
                               transform=fig.transFigure,
                               color='gray', linewidth=1.5, linestyle='--'))

fig.suptitle('Block Gram matrix — encoder log-det reg  (empirical  |  theory  |  balanced init, no reg)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("./figures/figure-hier-qqt_encoder.png", bbox_inches='tight', dpi=150)
plt.show()

# ── Single-row figure: bottom-right block only (λ_L2=0.1, λ_det_W1=-0.2) ─────
label_reg = lmdas_reg[1]   # 0.1
label_det = lmdas[1]       # -0.2
key = (label_reg, label_det)
lam_label = lmda_to_bal_init[label_det]
lam_info = f'$\\lambda_{{L2}}={label_reg},\\ \\lambda_{{det}}^{{W_1}}={label_det}$'

qqt_emp = np.vstack([
    np.hstack([w1w1s_empirical[key][-1], ws_empirical[key][-1][0].T]),
    np.hstack([ws_empirical[key][-1][0], w2w2s_empirical[key][-1]])
])
err = (np.linalg.norm(qqt_emp - qqt_theory_by_lmda[label_det], 'fro') /
       np.linalg.norm(qqt_theory_by_lmda[label_det], 'fro'))

fig4, axs4 = plt.subplots(1, 3, figsize=(3.5 * 3, 3.2))

sns.heatmap(1./8.*qqt_emp, cmap=div_cmap, ax=axs4[0],
            cbar=True, cbar_kws={"shrink": 0.6})
axs4[0].set_xticks([]); axs4[0].set_yticks([])
axs4[0].set_title(f'empirical\n{lam_info}', fontsize=8, pad=4)
axs4[0].text(0.97, 0.03, f'$\\varepsilon$={err:.2f}',
             transform=axs4[0].transAxes,
             ha='right', va='bottom', fontsize=8,
             color='white', fontweight='bold')

sns.heatmap(1./8.*qqt_theory_by_lmda[label_det], cmap=div_cmap, ax=axs4[1],
            cbar=True, cbar_kws={"shrink": 0.6})
axs4[1].set_xticks([]); axs4[1].set_yticks([])
axs4[1].set_title(f'theory  ($\\lambda_{{bal}}={lam_label}$)\n{lam_info}', fontsize=8, pad=4)

sns.heatmap(1./8.*qqt_balanced_by_lmda[label_det], cmap=div_cmap, ax=axs4[2],
            cbar=True, cbar_kws={"shrink": 0.6})
axs4[2].set_xticks([]); axs4[2].set_yticks([])
axs4[2].set_title(f'balanced, no reg  ($\\lambda_{{bal}}={lam_label}$)\n{lam_info}', fontsize=8, pad=4)

fig4.suptitle('Block Gram matrix — encoder log-det reg  (empirical  |  theory  |  balanced init, no reg)', fontsize=11)
fig4.tight_layout(rect=[0, 0, 1, 0.92])
fig4.savefig("./figures/figure-hier-qqt_encoder_bottomright.png", bbox_inches='tight', dpi=150)
plt.show()
