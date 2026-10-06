"""Does the season reach the model through RTOFS itself? Time features under full-year tests.

Idea being tested: RTOFS already carries the seasonal cycle in its own fields
(SST, mixed layer, boundary layer, TCHP itself), so the calendar features have
little left to add. Two checks:

  A. Seasonal share of each input: the fraction of its variance explained by
     month on top of location (10-degree boxes), i.e. R2(box x month) - R2(box).
     If RTOFS fields are as seasonal as the Argo truth while the error is not,
     the season arrives through the RTOFS values.
  B. Remove the calendar features, the seasonally varying physical features,
     or both, and score on two test designs: the locked forward blocks (~5
     months each) and leave-one-year-out (each test block a full year, a
     week left out each side of the boundary). If time mattered for real its
     effect would be consistent across full-year folds; if it flips sign
     between folds it is fitting what happened in a particular period.

Outputs: OHC/output/time_season_loyo_20261004/
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
    AQUA, AXIS, BLUE, INK2, MUTED, NEUTRAL, ORANGE, SURF, headline)
from OHC.exploration.rtofs_followup_common import (  # noqa: E402
    TARGET, TIME_COLS, evaluate, forward_folds, load_work, loyo_folds, median_X, recipe)

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/time_season_loyo_20261004")
SEASONAL_PHYS = ["model_temp_excess_26c", "model_temp_excess_x_abs_lat", "model_mixed_layer_thickness_m",
                 "model_mlt_x_abs_lat", "model_surface_boundary_layer_thickness_m", "d26_minus_mlt_m",
                 "d26_to_sblt_ratio", "model_sst_local_std_1deg", "model_sst_grad_mag_per_100km",
                 "model_sst_anom_from_1deg_mean"]
VARIANTS = {"full recipe": lambda c: c,
            "remove calendar features": lambda c: [x for x in c if x not in TIME_COLS],
            "remove seasonal physical features": lambda c: [x for x in c if x not in SEASONAL_PHYS],
            "remove both": lambda c: [x for x in c if x not in TIME_COLS and x not in SEASONAL_PHYS]}
VCOL = {"remove calendar features": BLUE, "remove seasonal physical features": ORANGE, "remove both": AQUA}
VMARK = {"remove calendar features": "o", "remove seasonal physical features": "s", "remove both": "D"}


def seasonal_share(work: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    box = (np.floor(work["lat"] / 10).astype(int).astype(str) + "_" + np.floor(work["lon"] / 10).astype(int).astype(str))
    rows = []
    for c in cols:
        v = pd.to_numeric(work[c], errors="coerce")
        ok = v.notna()
        x, b, m = v[ok], box[ok], work.loc[ok, "month_int"]
        sst = ((x - x.mean()) ** 2).sum()
        r2_box = 1 - ((x - x.groupby(b).transform("mean")) ** 2).sum() / sst
        r2_bm = 1 - ((x - x.groupby([b, m]).transform("mean")) ** 2).sum() / sst
        # chance level: the same calculation with months shuffled within each box (sparse box-month cells
        # explain some variance even with random labels)
        rng = np.random.default_rng(0)
        null = []
        for _ in range(5):
            mp = m.groupby(b).transform(lambda q: rng.permutation(q.to_numpy()))
            null.append(1 - ((x - x.groupby([b, mp]).transform("mean")) ** 2).sum() / sst - r2_box)
        rows.append({"variable": c, "r2_location": float(r2_box), "r2_location_month": float(r2_bm),
                     "seasonal_share_pct": float(100 * (r2_bm - r2_box)),
                     "chance_share_pct": float(100 * np.mean(null)),
                     "seasonal_share_above_chance_pct": float(100 * (r2_bm - r2_box - np.mean(null)))})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res, shares = [], []
    for tn in ("tchp", "d26"):
        t = TARGET[tn]
        work = load_work(tn)
        cols = recipe(tn, work)
        work["__argo"] = work[t.obs_col]
        work["__rtofs"] = work[t.model_col]
        work["__error"] = work[t.delta_col]
        s = seasonal_share(work, ["__argo", "__rtofs", "__error"] + [c for c in SEASONAL_PHYS if c in cols])
        s["target"] = tn
        shares.append(s)
        for fname, folds in (("forward blocks (~5 months)", forward_folds(work)), ("full-year tests", loyo_folds(work))):
            for vname, sel in VARIANTS.items():
                vc = sel(cols)
                r = evaluate(work, tn, folds, median_X(vc))
                res.append({"target": tn, "test design": fname, "variant": vname, "n_features": len(vc), **r})
                print(tn, fname, vname, round(r["mae"], 3), "+/-", round(r["seed_sd"], 3), flush=True)
    R = pd.DataFrame(res)
    base = R[R.variant == "full recipe"].set_index(["target", "test design"])
    for c in [c for c in R.columns if c == "mae" or c.startswith("fold: ")]:
        R[f"change {c}"] = R.apply(lambda r: r[c] - base.loc[(r.target, r["test design"]), c], axis=1)
    R.to_csv(OUT / "time_season_scores.csv", index=False)
    S = pd.concat(shares)
    S.to_csv(OUT / "seasonal_share.csv", index=False)
    plot_scores(R)
    plot_shares(S)


def plot_scores(R: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), sharey=True)
    for ax, tn in zip(axes, ("tchp", "d26")):
        r = R[(R.target == tn) & (R.variant != "full recipe")]
        rows = []
        for design in ("forward blocks (~5 months)", "full-year tests"):
            fold_cols = [c for c in r.columns if c.startswith("change fold: ") and r[r["test design"] == design][c].notna().any()]
            rows += [(design, c) for c in fold_cols] + [(design, "change mae")]
        y = np.arange(len(rows))[::-1]
        ax.axvline(0, color=AXIS, lw=1.2)
        for k, (vname, col) in enumerate(VCOL.items()):
            for yy, (design, c) in zip(y, rows):
                v = r[(r["test design"] == design) & (r.variant == vname)][c]
                if len(v) and np.isfinite(v.iloc[0]):
                    ax.plot(v.iloc[0], yy + (1 - k) * 0.22, VMARK[vname], color=col, ms=8, mec=SURF, mew=1.2,
                            label=vname if yy == y[0] else None)
        labels = [("ALL: " + d.split(" (")[0]) if c == "change mae" else c.replace("change fold: ", "  ")
                  for d, c in rows]
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        full = R[(R.target == tn) & (R.variant == "full recipe")].set_index("test design")
        for yy, (d, c) in zip(y, rows):
            if c == "change mae":
                ax.axhspan(yy - 0.45, yy + 0.45, color=NEUTRAL, zorder=0)
            base = full.loc[d, c.replace("change ", "")]
            ax.text(1.02, yy, f"{base:.2f}", transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=10.5, color=INK2, fontweight="bold" if c == "change mae" else None)
        ax.text(1.02, y[0] + 0.75, "full recipe\nMAE (= 0)", transform=ax.get_yaxis_transform(), ha="left",
                va="bottom", fontsize=9.5, color=MUTED)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{tn.upper()}", loc="left")
        ax.set_xlabel("Change in error (MAE) vs the full recipe on the same test block    ← better | worse →",
                      fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.55, 0.0))
    headline(fig, "The calendar adds little once RTOFS's seasonal fields are in, and hurts D26 in full-year tests",
             "Each row is one test block; shaded rows are overall scores. Full-year tests train on one year and test on "
             "the other. TCHP: removing the calendar or the seasonal\nphysical features (SST, mixed layer, boundary layer) "
             "alone changes little and the sign flips between periods; removing both costs most, so they carry the same\n"
             "seasonal information. D26: the seasonal physical features help in every test, while removing the calendar "
             "improves both full-year tests.", top=0.76)
    fig.subplots_adjust(left=0.15, right=0.93, bottom=0.16, wspace=0.2)
    fig.savefig(OUT / "time_vs_seasonal_physics_by_test_block.png", dpi=150)
    plt.close(fig)


def plot_shares(S: pd.DataFrame) -> None:
    name = {"__argo": "Argo value (truth)", "__rtofs": "RTOFS value", "__error": "error (Argo − RTOFS)"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    for ax, tn in zip(axes, ("tchp", "d26")):
        s = S[S.target == tn].copy()
        s["label"] = s.variable.map(lambda v: name.get(v, v.replace("model_", "").replace("surface_boundary_layer",
                                                                                         "boundary_layer")))
        s = s.iloc[::-1]
        colors = [BLUE if v in ("__argo", "__rtofs") else ORANGE if v == "__error" else AXIS for v in s.variable]
        ax.barh(np.arange(len(s)), s.seasonal_share_above_chance_pct, color=colors, edgecolor=SURF, lw=2,
                height=0.72)
        for i, v in enumerate(s.seasonal_share_above_chance_pct):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=10.5, color=INK2)
        ax.set_yticks(np.arange(len(s)))
        ax.set_yticklabels(s.label)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{tn.upper()}", loc="left")
        ax.set_xlabel("Seasonal share, above chance (%)")
    headline(fig, "The season reaches the model through RTOFS's own fields",
             "Share of each quantity's variation that the calendar month explains once location (10° box) is known, "
             "minus chance (months shuffled: 1-2%). Blue: Argo truth and RTOFS value;\norange: the error the model "
             "predicts; grey: RTOFS physical inputs. TCHP's error is less seasonal than TCHP itself. D26's error is "
             "more seasonal, but RTOFS's SST and mixed layer\n(3-6x more seasonal than either error) carry that "
             "season to the model, which is why the calendar has little left to add.", top=0.8)
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.11, wspace=0.62)
    fig.savefig(OUT / "seasonal_share_of_each_input.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if "--plot" in sys.argv:   # re-draw from the saved tables
        plot_scores(pd.read_csv(OUT / "time_season_scores.csv"))
        plot_shares(pd.read_csv(OUT / "seasonal_share.csv"))
    else:
        main()
