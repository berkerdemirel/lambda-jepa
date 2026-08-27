---
theme: default
title: Two-Space Faithfulness in Projector-Based Self-Supervised Learning
info: |
  Slidev port of docs/slides/two_space_story.tex (beamer), built 2026-08-26.
  Same spine, same exhibits, same wording — figures are symlinked into public/,
  so regenerating an exhibit flows through on reload. Do not edit the paper .tex,
  cards, DECISIONS or the exhibit generators from here.
mdc: true
transition: none
drawings:
  persist: false
fonts:
  provider: none
layout: cover
---

# Two-Space Faithfulness in<br>Projector-Based Self-Supervised Learning

The objective acts on $Z=p(H)$; downstream use keeps $H$.

<div class="agenda">

1. the gap, and the bridge that relates the two spaces
2. a dual-space regularizer that controls what the bridge needs
3. how much augmentation variation $H$ should retain — and what it buys

</div>

---

# Each method's own promise: at $z$ and $h$

**§1** *Properties guaranteed in the projector space need not hold in $H$.* {.spine}

<img src="/fig_discrepancy.png" class="fig w-[78%]" />

---

# The bridge: how much of $Z$'s organization is linearly available from $H$?

**§1** *Linear accessibility provides a way to relate the two spaces, but requires sufficient rank and dimensionality in both.* {.spine}

<img src="/C3b_accessibility_profile.png" class="fig w-[94%]" />

Mean explained variance over the $k$ best-explained whitened $z$ directions vs. $k$ (on held out data); each curve runs to that method's *own* target rank at $z$, its endpoint dot is $R^2$. Violet dashed = control, green = $+$ the $h$-side regularization. {.reads}

---

# The objective: one regularizer, applied in both spaces

**§2** *We introduce a dual-space regularizer with a spectral barrier against collapse. Applied to both $H$ and $Z$, it directly controls the capacity required by the accessibility bridge.* {.spine}

$$
\begin{aligned}
\mathcal L\;&=\;\underbrace{\mathcal L_{\text{pair}}(z_1,\dots,z_V)}_{\text{organizes }z}
\;+\;\lambda_z\,\mathcal K(\widehat\mu_z,\widehat\Sigma_z)
\;+\;\lambda_h\,\mathcal K(\widehat\mu_h,\widehat\Sigma_h),\\[0.5em]
\mathcal K(\mu,\Sigma)\;&=\;\tfrac12\bigl(\|\mu\|^{2}
+\operatorname{tr}\Sigma-\log\det\Sigma-d\bigr).
\end{aligned}
$$

<div class="body">

A **center** is one image's $V$ augmented views, embedded and averaged. $\widehat\mu_s,\widehat\Sigma_s$ are the mean and covariance of those centers across a batch, in the backbone ($s=h$) or after the projector ($s=z$); $d$ is that space's dimension.

- **The barrier.** $-\log\det\Sigma$ diverges as any eigenvalue of $\Sigma$ approaches $0$; $\operatorname{tr}\Sigma$ penalizes the other side and $\|\mu\|^2$ pulls the mean to the origin.
- **Why centers.** $\operatorname{Cov}(\text{centers})=\operatorname{Cov}(\text{exact centers})+\tfrac1V\operatorname{Cov}(\text{views of one image})$ — averaging first shrinks the augmentation part by $1/V$, so the term protects across-image spread rather than augmentation spread.
- **Both ends.** The same term at $z$ and at $h$.

</div>

---

# It protects stable center capacity in the retained backbone

**§2** *…it controls the capacity required by the accessibility bridge.* {.spine}

<img src="/depth_capacity.png" class="fig w-full" />

---

# Added to existing methods, it moves each one the same way

**§3** *The same regularizer also changes how much augmentation variation is retained in $H$.* {.spine}

<img src="/fig_treatment_arrows.png" class="fig w-[83%]" />

One arrow per method, control $\to$ treated, in (retained view-sensitivity $\Theta_h$ at CLS, probe accuracy): linear top-1 left, kNN-200 right. {.reads}

---

# Where the gain comes from

**§3** *…neither too little nor too much is desirable.* {.spine}

<img src="/fig_org_vs_sensitivity.png" class="fig w-[76%]" />

One arrow per model, control $\to$ treated (left: mean over the 100 classes; right: our pair, one arrow per class). **y-axis** $=1-$MSE (test) of a linear probe on view-averaged features. **Right** $=$ what a *single* view costs relative to that average, along the class's own readout direction. {.reads}

---

# Competitive as a standalone objective (ImageNet-1k, one protocol)

**§3** *…while achieving competitive performance as a standalone SSL objective.* {.spine}

<div class="tbl">

| Method | Backbone | Ep. | Linear | kNN |
|:---|:---|---:|---:|---:|
| LeJEPA | ViT-S/16 | 100 | 64.1 | 47.1 |
| DINO | ViT-S/16 | 100 | 70.0 | 64.7 |
| iBOT | ViT-S/16 | 100 | 70.9 | 65.7 |
| **Ours** | ViT-S/16 | 100 | **66.3** | **54.5** |
| VISReg | ViT-B/16 | 100 | 70.3 | — |
| **Ours** | ViT-B/16 | 100 | **71.2** | **56.0** |
| **Ours** | ViT-L/16 | 100 | — | — |

</div>

---

# Summary

<div class="body sum">

**1.** *Self-supervised learning objectives are typically enforced after a projector, while downstream tasks use the backbone representation $H$. Properties guaranteed in the projector space therefore need not hold in $H$. Linear accessibility provides a way to relate the two spaces, but requires sufficient rank and dimensionality in both.*

**2.** *Motivated by feature-learning results on the rank and dimensionality of learned representations, we introduce a dual-space regularizer with a spectral barrier against collapse. Applied to both $H$ and $Z$, it directly controls the capacity required by the accessibility bridge.*

**3.** *The same regularizer also changes how much augmentation variation is retained in $H$. We show that this variation is meaningful relative to image-to-image variation, and that neither too little nor too much is desirable. Our method typically moves this ratio in a favorable direction while achieving competitive performance as a standalone SSL objective.*

</div>
