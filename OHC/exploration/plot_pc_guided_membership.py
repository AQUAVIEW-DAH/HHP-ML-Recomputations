"""Which features are in each group of pc_guided_selection.png.

Reads the saved selection table (no refitting) and draws, for each target, a
membership grid: one column per group, in the same order as the bars in
pc_guided_selection.png (worst at left), one row per feature that appears in
any group. A filled cell means the feature is in that group.

Output: OHC/output/redundancy_followup_20260916/pc_guided_selection_members.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/redundancy_followup_20260916")


def main() -> None:
    t = pd.read_csv(OUT / "pc_guided_selection.csv")
    fig, axes = plt.subplots(1, 2, figsize=(22, 13), constrained_layout=True)
    for ax, tn in zip(axes, ("tchp", "d26")):
        s = t[t.target == tn]
        recipe = s[s.kind == "reference"].features.iloc[0].split("|")
        groups = s[s.kind.isin(["pc_cluster", "one_per_pc"])].sort_values("mae", ascending=False)
        members = [g.split("|") for g in groups.features]
        used = [f for f in recipe if any(f in m for m in members)]
        for j, (m, kind) in enumerate(zip(members, groups.kind)):
            c = "#2563eb" if kind == "one_per_pc" else "#64748b"
            for f in m:
                ax.add_patch(plt.Rectangle((j - 0.42, used.index(f) - 0.42), 0.84, 0.84, color=c))
        ax.set_xlim(-0.5, len(groups) - 0.5); ax.set_ylim(len(used) - 0.5, -0.5)
        ax.set_yticks(range(len(used))); ax.set_yticklabels(used, fontsize=10)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g}\nMAE {m:.2f}\n(n={n})" for g, m, n in zip(groups.set, groups.mae, groups.n)],
                           rotation=90, fontsize=9.5)
        ax.xaxis.tick_top()
        ax.set_xticks([x + 0.5 for x in range(len(groups))], minor=True)
        ax.set_yticks([y + 0.5 for y in range(len(used))], minor=True)
        ax.grid(which="minor", color="#e2e8f0", lw=0.8); ax.tick_params(which="minor", length=0)
        unused = [f for f in recipe if f not in used]
        ax.set_xlabel(f"{tn.upper()}  (full recipe {len(recipe)} features, MAE "
                      f"{s[s.kind == 'reference'].mae.iloc[0]:.2f})\n"
                      f"never selected in any group ({len(unused)}): " +
                      "\n".join(", ".join(unused[i:i + 3]) for i in range(0, len(unused), 3)), fontsize=10)
    fig.suptitle("Features in each group of pc_guided_selection.png  "
                 "(grey = top 5 loaders of one component, blue = top loader of each of the first m components; "
                 "columns ordered as the bars, worst MAE at left)", fontsize=13)
    fig.savefig(OUT / "pc_guided_selection_members.png", dpi=150)
    print("wrote", OUT / "pc_guided_selection_members.png")


if __name__ == "__main__":
    main()
