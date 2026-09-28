"""The K710-20 % pretraining list: union of the K400/600/700 train lists, duplicates and validation ids removed, 20 % per class (seed 0)."""
import numpy as np, pandas as pd
B = os.environ["LEVJEPA_DATA_ROOT"]
def norm(s): return " ".join(str(s).lower().replace("_", " ").split())
train = {v: pd.read_csv(f"{B}/k{v}/annotations/train.csv") for v in ("400", "600", "700_2020")}
val = {v: pd.read_csv(f"{B}/k{v}/annotations/val.csv") for v in ("400", "600", "700_2020")}
for v, d in train.items():
    d["label"] = d["label"].map(norm); d["source"] = f"k{v}"
    print(f"k{v} train: {len(d)} rows, {d.youtube_id.nunique()} ids, {d.label.nunique()} classes", flush=True)
union = pd.concat([train["700_2020"], train["600"], train["400"]]).drop_duplicates("youtube_id", keep="first")
val_ids = set(pd.concat(val.values()).youtube_id)
n0 = len(union); union = union[~union.youtube_id.isin(val_ids)]
print(f"union: {n0} unique ids -> {len(union)} after removing {n0 - len(union)} validation-overlap ids; "
      f"{union.label.nunique()} classes (normalized strings)", flush=True)
rng = np.random.default_rng(0)
parts = []
for lab, g in union.groupby("label", sort=True):
    k = int(round(0.2 * len(g)))
    parts.append(g.iloc[rng.choice(len(g), size=k, replace=False)] if k else g.iloc[:0])
sub = pd.concat(parts).sort_values(["label", "youtube_id"]).reset_index(drop=True)
print(f"20% draw: {len(sub)} videos over {sub.label.nunique()} classes; per-class min/median/max "
      f"{sub.label.value_counts().min()}/{int(sub.label.value_counts().median())}/{sub.label.value_counts().max()}; "
      f"by source {sub.source.value_counts().to_dict()}", flush=True)
cols = ["youtube_id", "time_start", "time_end", "label", "source"]
union[cols].to_csv(f"{B}/k710_union.csv", index=False)
sub[cols].to_csv(f"{B}/k710_20pct.csv", index=False)
with open(f"{B}/k710_20pct_members.txt", "w") as f:
    for r in sub.itertuples():
        f.write(f"{r.youtube_id}_{int(r.time_start):06d}_{int(r.time_end):06d}.mp4\n")
print("K710_SUBSAMPLE_DONE", flush=True)
