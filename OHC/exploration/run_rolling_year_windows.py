"""Full-year tests that do not start on 1 January (meeting of 2026-10-08).

The earlier full-year tests always split at 1 Jan 2025 (test 2024 / test 2025). Here the 12-month test
window starts in different months: Feb 2024, May 2024, Aug 2024, Nov 2024 and Jan 2025. Each model trains on
the remaining ~11 months (before and/or after the window), with 7 days left out on each side of the window.
This shows whether a result depends on where the test year begins.

Compared: the full recipe, the full recipe without calendar features, the full recipe plus RTOFS's
recent history at the float (12 features), and Dr. Jacobs' 16 features. Warm rows, one profile per cast,
3 seeds. Outputs: OHC/output/rolling_year_windows_20261015/
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.exploration.plot_meeting_followups_20261001 import (  # noqa: E402
    AQUA, AXIS, BLUE, INK2, MUTED, NEUTRAL, ORANGE, SURF, headline)
from OHC.exploration.rtofs_followup_common import TARGET, TIME_COLS, evaluate, median_X, recipe  # noqa: E402
from OHC.exploration.run_noise_feature_test import JAC, prepare  # noqa: E402

OUT = Path("/home/suramya/HHP-Prediction/OHC/output/rolling_year_windows_20261015")
HIST = Path("/home/suramya/HHP-Prediction/OHC/output/rtofs_history_20261004/history_features.parquet")
NOISE = Path("/home/suramya/HHP-Prediction/OHC/output/noise_feature_20261015/noise_summary.csv")
STARTS = ["2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01", "2025-01-01"]
BUFFER = 7
UNIT = {"tchp": "kJ/cm²", "d26": "m"}
VARIANTS = ["full recipe", "full recipe without calendar", "full recipe + RTOFS recent history", "Dr. Jacobs' set"]
COL = {"full recipe": BLUE, "full recipe without calendar": MUTED, "full recipe + RTOFS recent history": AQUA,
       "Dr. Jacobs' set": ORANGE}


def windows(work: pd.DataFrame) -> list[dict]:
    d = pd.to_datetime(pd.Series(sorted(work.ds.unique())), format="%Y%m%d")
    out = []
    for s in STARTS:
        a = pd.Timestamp(s)
        b = a + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        val = d[(d >= a) & (d <= b)]
        tr = d[(d < a - pd.Timedelta(days=BUFFER)) | (d > b + pd.Timedelta(days=BUFFER))]
        out.append({"name": f"{a:%b %Y} - {b:%b %Y}", "train_dates": tr.dt.strftime("%Y%m%d").tolist(),
                    "val_dates": val.dt.strftime("%Y%m%d").tolist()})
    return out


def task(args):
    tn, variant = args
    t = TARGET[tn]
    w = prepare(tn)
    H = pd.read_parquet(HIST)
    w = w.merge(H, on="cast_id", how="left")
    full = recipe(tn, w)
    cols = {"full recipe": full,
            "full recipe without calendar": [c for c in full if c not in TIME_COLS],
            "full recipe + RTOFS recent history": full + [c for c in H.columns if c != "cast_id"],
            "Dr. Jacobs' set": JAC}[variant]
    rows = []
    for f in windows(w):
        r = evaluate(w, tn, [f], median_X(cols))
        va = w.ds.isin(set(f["val_dates"])).to_numpy()
        raw = float(np.abs(w.loc[va, t.model_col] - w.loc[va, t.obs_col]).mean())
        rows.append({"target": tn, "variant": variant, "window": f["name"], "mae": r["mae"], "seed_sd": r["seed_sd"],
                     "raw_rtofs_mae": raw, "train_days": len(f["train_dates"]), "test_days": len(f["val_dates"]),
                     "test_profiles": int(va.sum())})
    print(tn, variant, "done", flush=True)
    return rows


def plots() -> None:
    R = pd.read_csv(OUT / "rolling_year_scores.csv")
    N = pd.read_csv(NOISE)
    band = {tn: 2 * float(N[(N.target == tn) & (N.design == "forward blocks")].val_mae_sd.mean()) for tn in ("tchp", "d26")}
    wins = list(dict.fromkeys(R.window))
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5), gridspec_kw={"height_ratios": [1.1, 1]})
    x = np.arange(len(wins))
    for j, tn in enumerate(("tchp", "d26")):
        r = R[R.target == tn]
        ax = axes[0, j]
        for v in VARIANTS:
            s = r[r.variant == v].set_index("window").loc[wins]
            ax.plot(x, s.mae, "-o", color=COL[v], ms=7, mec=SURF, lw=1.8, label=v)
        raw = r[r.variant == "full recipe"].set_index("window").loc[wins].raw_rtofs_mae
        ax.set_xticks(x)
        ax.set_xticklabels([f"{w}\n(raw RTOFS {v:.1f})" for w, v in zip(wins, raw)], fontsize=9.5)
        ax.set_ylabel(f"MAE on the 12-month window ({UNIT[tn]})")
        ax.set_title(tn.upper(), loc="left")
        ax.grid(axis="x", visible=False)

        ax = axes[1, j]
        base = r[r.variant == "full recipe"].set_index("window").loc[wins].mae
        ax.axhspan(-band[tn], band[tn], color=NEUTRAL, zorder=0)
        ax.axhline(0, color=AXIS, lw=1)
        wbar = 0.25
        for k, v in enumerate(VARIANTS[1:]):
            s = r[r.variant == v].set_index("window").loc[wins].mae - base
            ax.bar(x + (k - 1) * wbar, s.values, width=wbar - 0.02, color=COL[v], edgecolor=SURF, label=v)
        ax.set_xticks(x)
        ax.set_xticklabels([w.replace(" - ", "\nto ") for w in wins], fontsize=9.5)
        ax.set_ylabel(f"Change vs the full recipe ({UNIT[tn]})\n← better | worse →")
        ax.grid(axis="x", visible=False)
        ax.text(0.01, 0.97, f"grey band: chance level ±{band[tn]:.3f}", transform=ax.transAxes, va="top", fontsize=9.5,
                color=INK2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.905),
               fontsize=10.5)
    headline(fig, "In every 12-month window recent history helps most and Dr. Jacobs' set trails; the calendar's "
             "effect flips sign",
             "Each window is one year of test data; the model trains on the other ~11 months (7 days left out at each "
             "edge), 3 seeds. Raw RTOFS error for each window is in brackets.\nTop: error of each feature set. Bottom: "
             "change vs the full recipe; the grey band is the chance level from the noise-feature test.", top=0.85)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.09, hspace=0.35, wspace=0.18)
    fig.savefig(OUT / "1_rolling_12_month_windows.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(tn, v) for tn in ("tchp", "d26") for v in VARIANTS]
    with ProcessPoolExecutor(max_workers=2) as ex:
        rows = [r for rs in ex.map(task, tasks) for r in rs]
    pd.DataFrame(rows).to_csv(OUT / "rolling_year_scores.csv", index=False)
    plots()


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plots()
    else:
        main()
