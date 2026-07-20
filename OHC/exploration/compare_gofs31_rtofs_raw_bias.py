"""SIDE EXPLORATION: does the raw-model error structure match across models?

First cross-dataset validation: compare the RAW model-minus-Argo error
statistics of GOFS 3.1 reanalysis (2015) against operational RTOFS
(2024-2025), globally and per named 20-degree box. If the same warm-pool
underestimation pattern appears in both, the bias our correction learns is a
family trait of HYCOM-based systems rather than an artifact of one model
vintage — supporting (though not proving) cross-model transfer of the
correction hypothesis. Expect GOFS residuals to be smaller overall because
NCODA assimilated the very Argo profiles we compare against.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OHC.build_hhp_density_scatter_diagnostics import NAMED_BOXES, _named_box_rows, _augment_regions_and_patches  # noqa: E402

GOFS_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_gofs31_collocated_2015.parquet")
RTOFS_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet")
OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/gofs31_cross_model")

TARGETS = {
    "tchp": ("argo_tchp_kj_per_cm2", "model_interp_tchp_kj_per_cm2", "kJ/cm²"),
    "d26": ("argo_d26_m", "model_interp_d26_m", "m"),
}


def _stats(df: pd.DataFrame, obs_col: str, model_col: str) -> dict:
    obs = pd.to_numeric(df[obs_col], errors="coerce").to_numpy(float)
    mod = pd.to_numeric(df[model_col], errors="coerce").to_numpy(float)
    ok = np.isfinite(obs) & np.isfinite(mod)
    if ok.sum() < 10:
        return {"rows": int(ok.sum()), "bias": np.nan, "mae": np.nan, "corr": np.nan}
    err = mod[ok] - obs[ok]
    return {
        "rows": int(ok.sum()),
        "bias": float(err.mean()),
        "mae": float(np.abs(err).mean()),
        "corr": float(pd.Series(obs[ok]).corr(pd.Series(mod[ok]))),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gofs = _augment_regions_and_patches(pd.read_parquet(GOFS_PATH))
    rtofs = _augment_regions_and_patches(pd.read_parquet(RTOFS_PATH))

    rows = []
    for tname, (obs_col, model_col, units) in TARGETS.items():
        for dataset, df in [("gofs31_2015", gofs), ("rtofs_2024_2025", rtofs)]:
            rows.append({"target": tname, "dataset": dataset, "scope": "global", "box": "global", **_stats(df, obs_col, model_col)})
            for box in NAMED_BOXES:
                rows.append({
                    "target": tname, "dataset": dataset, "scope": "named_box", "box": box.key,
                    **_stats(_named_box_rows(df, box), obs_col, model_col),
                })
    out = pd.DataFrame(rows)
    out_csv = OUT_DIR / "gofs31_vs_rtofs_raw_bias.csv"
    out.to_csv(out_csv, index=False)

    # Named-box raw-bias comparison figure, one panel per target.
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.2), constrained_layout=True)
    x = np.arange(len(NAMED_BOXES))
    width = 0.38
    for ax, (tname, (_, _, units)) in zip(axes, TARGETS.items()):
        sub = out[(out.target == tname) & (out.scope == "named_box")]
        for i, (dataset, color, label) in enumerate([
            ("rtofs_2024_2025", "#dc2626", "raw RTOFS (2024-2025)"),
            ("gofs31_2015", "#2563eb", "raw GOFS 3.1 reanalysis (2015)"),
        ]):
            vals = [float(sub[(sub.dataset == dataset) & (sub.box == b.key)]["bias"].iloc[0]) for b in NAMED_BOXES]
            ax.bar(x + (i - 0.5) * width, vals, width, color=color, label=label)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([b.display for b in NAMED_BOXES], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(f"raw bias, model - Argo ({units})")
        ax.set_title(tname.upper())
        ax.grid(True, axis="y", alpha=0.15, linewidth=0.4)
    axes[0].legend(fontsize=9)
    fig.suptitle(
        "Raw model bias vs Argo per named 20° box: operational RTOFS vs GOFS 3.1 reanalysis\n"
        "Caveat: GOFS assimilated these Argo profiles (NCODA), so its residuals are expected to be smaller",
        fontsize=13,
    )
    fig_path = OUT_DIR / "gofs31_vs_rtofs_named_box_bias.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(out[out.scope == "global"].to_string(index=False))
    print(f"\nWrote {out_csv}\nWrote {fig_path}")


if __name__ == "__main__":
    main()
