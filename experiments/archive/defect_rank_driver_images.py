"""Show the actual images behind the pursuit outlier directions (Berker's R2 criterion
2026-07-10: 'if those images qualitatively look bad we can consider keeping trimmed khat').
Top-5 driver images per toy cell, ranked by appearances across that cell's top-10 pursuit
directions (from results/diag/defect_rank_outliers.csv), tie-broken by extremity position.
Index chain guard: manifest imagenette.train.v1 has ref=i for HF train[i] (built from
range(len(ds))), and we assert labels.npy[idx] == ds[idx].label for every image shown.

  python experiments/defect_rank_driver_images.py
    -> results/figures/diag/defect_rank_driver_images.png
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import ImageOps

from defect_rank_toy import CELLS, FEAT, MAN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def top5_per_cell(df):
    """Rank images by (appearances across the cell's top-10 pursuit dirs, best extremity
    position, earliest dir); positions come from top10_idx which is ordered most-extreme first."""
    out = {}
    for cell, g in df[(df.kind == "pp") & (df["rank"] <= 10)].groupby("cell"):
        score = {}
        for _, row in g.sort_values("rank").iterrows():
            for pos, idx in enumerate(map(int, row.top10_idx.split(";"))):
                s = score.setdefault(idx, [0, pos, row["rank"]])
                s[0] -= 1                       # more appearances -> smaller sort key
                s[1] = min(s[1], pos)
        out[cell] = [i for i, _ in sorted(score.items(), key=lambda kv: kv[1])[:5]]
    return out


def main():
    df = pd.read_csv(os.path.join(ROOT, "results", "diag", "defect_rank_outliers.csv"))
    picks = top5_per_cell(df)
    ds = load_dataset("frgfm/imagenette", "160px", split="train")
    names = ds.features["label"].names
    labels = np.load(f"{FEAT}/{CELLS[0][1]}/{MAN}/labels.npy")

    cells = [c[0] for c in CELLS]
    fig, axes = plt.subplots(len(cells), 5, figsize=(9.5, 2.05 * len(cells)))
    for r, cell in enumerate(cells):
        for c, idx in enumerate(picks[cell]):
            ex = ds[idx]
            assert int(labels[idx]) == int(ex["label"]), f"index chain broken at {idx}"
            axes[r, c].imshow(ImageOps.fit(ex["image"].convert("RGB"), (160, 160)))
            axes[r, c].set_title(f"#{idx} {names[ex['label']]}", fontsize=7)
            axes[r, c].axis("off")
        axes[r, 0].text(-0.08, 0.5, cell, transform=axes[r, 0].transAxes, fontsize=8,
                        ha="right", va="center", rotation=90)
    fig.suptitle("Top-5 pursuit-outlier driver images per cell (appearances across top-10 dirs)",
                 fontsize=11)
    fig.tight_layout(rect=(0.02, 0, 1, 0.99))
    out = os.path.join(ROOT, "results", "figures", "diag", "defect_rank_driver_images.png")
    fig.savefig(out, dpi=140)
    print(f"[drivers] wrote {out}")
    for cell in cells:
        print(f"[drivers] {cell}: {[(i, names[int(labels[i])]) for i in picks[cell]]}")


if __name__ == "__main__":
    main()
