"""Statistics above a critical value (Dr. Jacobs, 2026-08 comment).

Averaging over all observed values dilutes the differences between models;
tropical-cyclone work uses critical TCHP thresholds (rapid intensification is
favored above roughly 50-60 kJ/cm2). This script conditions every statistic on
the observed value exceeding a threshold:

1. Threshold sweep: MAE(rows with observed >= t) vs t for raw RTOFS, the
   previous best single global model, and the recommended MoE blend.
2. Named-box MAE bars restricted to the high regime (TCHP >= 60 kJ/cm2,
   D26 >= 100 m), the conditioned version of `*_moe_named_box_mae.png`.

Uses the saved out-of-fold predictions from the MoE showcase, so no
retraining is involved.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "figure.titlesize": 17})
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.build_hhp_density_scatter_diagnostics import NAMED_BOXES  # noqa: E402
from OHC.exploration.run_gom_attribution_analysis import SHORT, UNITS  # noqa: E402

RUN_DATE = "2026-08-12"
IN_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/moe_showcase_20260811")
OUT_DIR = Path(f"/home/suramya/HHP-Prediction/OHC/output/critical_threshold_{RUN_DATE.replace('-', '')}")
OBS_COL = {"tchp": "argo_tchp_kj_per_cm2", "d26": "argo_d26_m"}
CRITICAL = {"tchp": 60.0, "d26": 100.0}
SWEEP = {"tchp": np.arange(0, 130, 10.0), "d26": np.arange(0, 150, 10.0)}
SERIES = [("pred_obs__raw_rtofs", "raw RTOFS", "#dc2626"),
          ("pred_obs__global_best", "previous best (single global)", "#94a3b8"),
          ("pred_obs__moe_blend", "recommended MoE blend", "#2563eb")]
MIN_ROWS = 40
BOX_DEG = 20


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for tname in ("tchp", "d26"):
        df = pd.read_parquet(IN_DIR / f"moe_predictions_{tname}.parquet")
        ok = np.isfinite(df[[c for c, _, _ in SERIES]]).all(axis=1).to_numpy()
        df = df[ok].reset_index(drop=True)
        y = df[OBS_COL[tname]].to_numpy(float)

        # 1 — threshold sweep
        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        for col, label, color in SERIES:
            p = df[col].to_numpy(float)
            ts, maes = [], []
            for t in SWEEP[tname]:
                m = y >= t
                if m.sum() < MIN_ROWS:
                    break
                ts.append(t)
                maes.append(float(np.abs(p[m] - y[m]).mean()))
                rows.append({"target": tname, "model": label, "scope": "global",
                             "threshold": float(t), "rows": int(m.sum()), "mae": maes[-1],
                             "bias": float((p[m] - y[m]).mean())})
            ax.plot(ts, maes, marker="o", linewidth=2, color=color, label=label)
        ax.axvline(CRITICAL[tname], color="black", linestyle="--", linewidth=1.2,
                   label=f"critical value {CRITICAL[tname]:.0f} {UNITS[tname]}")
        ax.set_xlabel(f"threshold t: statistics over rows with observed {SHORT[tname]} ≥ t ({UNITS[tname]})")
        ax.set_ylabel(f"MAE above threshold ({UNITS[tname]})")
        ax.set_title(f"{SHORT[tname]}: error restricted to the high regime, as a function of the cutoff")
        ax.grid(True, alpha=0.15)
        ax.legend()
        fig.savefig(OUT_DIR / f"{tname}_threshold_sweep.png", dpi=180)
        plt.close(fig)

        # 2 — named-box bars above the critical value
        la = np.floor(df["lat"].to_numpy(float) / BOX_DEG) * BOX_DEG
        lo = np.floor(((df["lon"].to_numpy(float) + 180.0) % 360.0 - 180.0) / BOX_DEG) * BOX_DEG
        crit = CRITICAL[tname]
        high = y >= crit
        fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
        x = np.arange(len(NAMED_BOXES))
        width = 0.26
        for i, (col, label, color) in enumerate(SERIES):
            p = df[col].to_numpy(float)
            vals = []
            for b in NAMED_BOXES:
                sel = (la == b.lat0) & (lo == b.lon0) & high
                n = int(sel.sum())
                vals.append(float(np.abs(p[sel] - y[sel]).mean()) if n >= 25 else np.nan)
                rows.append({"target": tname, "model": label, "scope": f"box:{b.key}",
                             "threshold": crit, "rows": n,
                             "mae": vals[-1], "bias": float((p[sel] - y[sel]).mean()) if n >= 25 else np.nan})
            ax.bar(x + (i - 1) * width, vals, width, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([b.display for b in NAMED_BOXES], rotation=30, ha="right", fontsize=10)
        ax.set_ylabel(f"MAE over rows with observed ≥ {crit:.0f} {UNITS[tname]}")
        ax.set_title(f"{SHORT[tname]}: per-box error in the high regime only (observed ≥ {crit:.0f} {UNITS[tname]})")
        ax.grid(True, axis="y", alpha=0.15)
        ax.legend()
        fig.savefig(OUT_DIR / f"{tname}_critical_named_box_mae.png", dpi=180)
        plt.close(fig)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "critical_threshold_stats.csv", index=False)
    g = out[(out.scope == "global") & (out.threshold.isin([CRITICAL["tchp"], CRITICAL["d26"]]))]
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
