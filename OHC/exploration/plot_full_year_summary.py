"""All full-year test results in one figure (train on one calendar year, test on the other).

Collects the full-year rows of the follow-up experiments (calendar / seasonal physics, RTOFS recent
history, pairwise combinations, Dr. Jacobs' feature set) and plots each one's change in error against the
full recipe of the same experiment, separately for the 2024 and the 2025 test year. Rows where both years
agree in sign are the trustworthy ones. Warm rows (26 C water in both Argo and RTOFS) throughout.

Output: OHC/output/full_year_summary_20261014/full_year_tests_all_experiments.png (+ csv)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.exploration.plot_meeting_followups_20261001 import (  # noqa: E402
    AXIS, BLUE, INK2, NEUTRAL, ORANGE, SURF, headline)

O = Path("/home/suramya/HHP-Prediction/OHC/output")
OUT = O / "full_year_summary_20261014"
Y24, Y25 = "full year: test 2024", "full year: test 2025"


def collect() -> pd.DataFrame:
    rows = []

    def add(group, label, t, base, cur, c24, c25, b24, b25):
        rows.append({"group": group, "label": label, "target": t, "base_2024": b24, "base_2025": b25,
                     "change_2024": c24 - b24, "change_2025": c25 - b25, "mae_2024": c24, "mae_2025": c25})

    R = pd.read_csv(O / "time_season_loyo_20261004/time_season_scores.csv")
    R = R[R["test design"] == "full-year tests"]
    for t in ("tchp", "d26"):
        b = R[(R.target == t) & (R.variant == "full recipe")].iloc[0]
        for v, lab in (("remove calendar features", "remove calendar features"),
                       ("remove seasonal physical features", "remove seasonal physical features"),
                       ("remove both", "remove both")):
            r = R[(R.target == t) & (R.variant == v)].iloc[0]
            add("1. time and season", lab, t, b, r, r[f"fold: {Y24}"], r[f"fold: {Y25}"], b[f"fold: {Y24}"], b[f"fold: {Y25}"])
    H = pd.read_csv(O / "rtofs_history_20261004/history_scores.csv")
    H = H[H.design == "full-year tests"]
    for t in ("tchp", "d26"):
        b = H[(H.target == t) & H.variant.str.startswith("baseline")].iloc[0]
        for v in ("+ all 12 history features", "+ changes only (3-day, 7-day; 6)", "+ TCHP history (4)",
                  "+ D26 history (4)", "+ SST history (4)"):
            r = H[(H.target == t) & (H.variant == v)].iloc[0]
            add("2. RTOFS recent history", v, t, b, r, r[f"fold: {Y24}"], r[f"fold: {Y25}"], b[f"fold: {Y24}"], b[f"fold: {Y25}"])
    P = pd.read_csv(O / "pairwise_decomposition_20261004/pairwise_family_scores.csv")
    P = P[P.design == "full-year tests"]
    for t in ("tchp", "d26"):
        b = P[(P.target == t) & P.variant.str.startswith("baseline")].iloc[0]
        for v in ("+ all pairwise (reference)", "+ location × physics", "+ physics × physics", "+ squares only",
                  "+ location × location"):
            r = P[(P.target == t) & (P.variant == v)].iloc[0]
            add("3. pairwise combinations", v.replace(" (reference)", ""), t, b, r, r[f"fold: {Y24}"],
                r[f"fold: {Y25}"], b[f"fold: {Y24}"], b[f"fold: {Y25}"])
    J = pd.read_csv(O / "jacobs_feature_set_20261013/jacobs_vs_full_scores.csv")
    J = J[(J.design == "full-year tests") & (J.population == "26 °C water in both (earlier results)")]
    for t in ("tchp", "d26"):
        b = J[(J.target == t) & J.features.str.startswith("full recipe")].iloc[0]
        for v, lab in (("Dr. Jacobs' set (", "Dr. Jacobs' set instead (16)"),
                       ("Dr. Jacobs' set + |lat|", "Dr. Jacobs' set + |lat| instead (17)")):
            r = J[(J.target == t) & J.features.str.startswith(v)].iloc[0]
            add("4. Dr. Jacobs' feature set*", lab, t, b, r, r[f"block: {Y24}"], r[f"block: {Y25}"],
                b[f"block: {Y24}"], b[f"block: {Y25}"])
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    D = collect()
    D.to_csv(OUT / "full_year_tests_all_experiments.csv", index=False)
    labels = list(dict.fromkeys(zip(D.group, D.label)))
    fig, axes = plt.subplots(1, 2, figsize=(17, 10), sharey=True)
    y = np.arange(len(labels))[::-1]
    for ax, t in zip(axes, ("tchp", "d26")):
        d = D[D.target == t].set_index(["group", "label"]).loc[labels]
        ax.axvspan(-0.02, 0.02, color=NEUTRAL, zorder=0)
        ax.axvline(0, color=AXIS, lw=1.2)
        for yy, (_, r) in zip(y, d.iterrows()):
            ax.plot([r.change_2024, r.change_2025], [yy, yy], color=AXIS, lw=1.5, zorder=1)
        ax.plot(d.change_2024, y + 0.0, "o", color=BLUE, ms=8, mec=SURF, label="test year 2024 (trained on 2025)", zorder=3)
        ax.plot(d.change_2025, y + 0.0, "D", color=ORANGE, ms=8, mec=SURF, label="test year 2025 (trained on 2024)", zorder=3)
        groups = list(dict.fromkeys(g for g, _ in labels))
        for g in groups[1:]:
            k = [i for i, (gg, _) in enumerate(labels) if gg == g][0]
            ax.axhline(y[k] + 0.5, color=AXIS, lw=0.8, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels([lab for _, lab in labels])
        ax.grid(axis="y", visible=False)
        b = d.iloc[0]
        ax.set_title(f"{t.upper()}   (full recipe: {b.base_2024:.2f} on 2024, {b.base_2025:.2f} on 2025)", loc="left")
        ax.set_xlabel(f"Change in error vs the full recipe ({'kJ/cm²' if t == 'tchp' else 'm'})\n← better | worse →")
    for ax in axes:
        lo = min(D.change_2024.min(), D.change_2025.min()) - 0.04
        hi = max(D.change_2024.max(), D.change_2025.max()) + 0.04
        ax.set_xlim(lo, hi)
    for g in dict.fromkeys(D.group):
        k = [i for i, (gg, _) in enumerate(labels) if gg == g][0]
        axes[0].text(-0.62, y[k] + 0.3, g, transform=axes[0].get_yaxis_transform(), fontsize=11.5,
                     fontweight="bold", color=INK2, ha="left", va="bottom")
    axes[1].legend(loc="center right", fontsize=10.5)
    headline(fig, "Full-year tests: recent history and pairwise combinations help in both years; the calendar does not",
             "Each row is one experiment, scored twice: tested on all of 2024 (trained on 2025) and on all of 2025 (trained "
             "on 2024), a week left out on each side of 1 Jan 2025. The grey line joins the two years;\nthe grey band "
             "is ±0.02, roughly run-to-run noise. Warm rows (26 °C water in both Argo and RTOFS).\n*Dr. Jacobs' set is "
             "compared with the full recipe trained on all data (his zero rule); the others with the full recipe trained on "
             "warm rows only.", top=0.85)
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.09, wspace=0.06)
    fig.savefig(OUT / "full_year_tests_all_experiments.png", dpi=150)
    plt.close(fig)
    print(D[["target", "label", "change_2024", "change_2025"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
