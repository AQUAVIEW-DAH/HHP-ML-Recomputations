"""Presentation figures for the diagnostics / redundancy / emergence batch.

Everything is rendered from cached predictions and result tables, so no model
is refitted. Figures:
  1  model_comparison_significance : MAE + RMSE, and bootstrap confidence
     intervals on the paired differences (is the ranking real?)
  2  {t}_detail_significance       : RF correction, its bootstrap instability,
     and the correction masked to where it is statistically significant
  3  {t}_transect                  : all four models along a latitude band,
     against binned observed errors with standard errors
  4  boundary_emergence_waterfall  : the ~8,000 boundary cases, linear scale,
     with the improvement of each feature group annotated
  5  correlation_vs_value          : the headline — a feature's correlation
     with the error against what it actually contributes
  6  {t}_detail_ladder             : re-render of the ladder (title fix)

Outputs: OHC/output/diagnostics_figures_20260910/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from OHC.seasonal_map_common import add_land_overlay  # noqa: E402

DIAG = Path("/home/suramya/HHP-Prediction/OHC/output/latlon_diagnostics_20260908")
LADDER = Path("/home/suramya/HHP-Prediction/OHC/output/smooth_ladder_20260909")
RED = Path("/home/suramya/HHP-Prediction/OHC/output/feature_redundancy_20260909")
PG = Path("/home/suramya/HHP-Prediction/OHC/output/prune_graft_20260910")
OUT = Path("/home/suramya/HHP-Prediction/OHC/output/diagnostics_figures_20260910")
UNIT = {"tchp": "kJ/cm²", "d26": "m"}
LABEL = {"rf": "random forest", "gpr": "Gaussian process", "svr_rbf": "SVR (RBF)",
         "xgb": "gradient boosting", "raw": "raw RTOFS"}
COLOR = {"rf": "#2563eb", "gpr": "#16a34a", "svr_rbf": "#a855f7", "xgb": "#f59e0b", "raw": "#dc2626"}


def fig1() -> None:
    m = pd.read_csv(DIAG / "mae_rmse_table.csv")
    b = pd.read_csv(DIAG / "bootstrap_ranking.csv")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for j, t in enumerate(("tchp", "d26")):
        ax = axes[0, j]
        s = m[(m.target == t) & (m.scope == "warm")].set_index("model")
        order = ["raw", "xgb", "svr_rbf", "gpr", "rf"]
        x = np.arange(len(order)); w = 0.38
        ax.bar(x - w/2, [s.loc[k, "mae"] for k in order], w, color="#94a3b8", label="MAE")
        ax.bar(x + w/2, [s.loc[k, "rmse"] for k in order], w, color="#334155", label="RMSE")
        for i, k in enumerate(order):
            ax.text(i - w/2, s.loc[k, "mae"] + 0.15, f"{s.loc[k,'mae']:.2f}", ha="center", fontsize=9)
            ax.text(i + w/2, s.loc[k, "rmse"] + 0.15, f"{s.loc[k,'rmse']:.2f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[k] for k in order], rotation=18, ha="right", fontsize=9)
        ax.set_ylabel(f"error ({UNIT[t]})"); ax.set_title(f"{t.upper()}: position-only models, both metrics")
        ax.grid(alpha=0.15, axis="y"); ax.legend(fontsize=9)

        ax = axes[1, j]
        s = b[b.target == t].iloc[::-1].reset_index(drop=True)
        for i, r in s.iterrows():
            c = "#16a34a" if r.significant else "#94a3b8"
            ax.plot([r.ci_lo, r.ci_hi], [i, i], color=c, lw=3, solid_capstyle="round")
            ax.plot(r.mean_diff, i, "o", color=c, ms=7)
            ax.text(r.ci_hi + 0.02, i, "significant" if r.significant else "not significant",
                    va="center", fontsize=8, color=c)
        ax.axvline(0, color="black", lw=1.0)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels([p.replace("_rbf", "").replace("gpr", "GP").replace("rf", "RF").replace("xgb", "XGB")
                            for p in s.pair], fontsize=9)
        ax.set_xlabel(f"difference in MAE ({UNIT[t]});  negative = first model better")
        ax.set_title(f"{t.upper()}: 1000 date-block bootstrap resamples")
        ax.grid(alpha=0.15, axis="x")
        ax.set_xlim(min(s.ci_lo) - 0.1, max(s.ci_hi) + 0.5)
    fig.suptitle("Which position-only model wins, and is the difference real?\n"
                 "no ranking flips between MAE and RMSE; every pairwise gap excludes zero",
                 fontsize=14)
    fig.savefig(OUT / "model_comparison_significance.png", dpi=160)
    plt.close(fig)
    print("fig1 done")


def fig2() -> None:
    for t in ("tchp", "d26"):
        z = np.load(DIAG / f"{t}_grid_predictions.npz")
        glat, glon, rf, std = z["glat"], z["glon"], z["rf"], z["rf_boot_std"]
        sig = np.abs(rf) > 2 * np.maximum(std, 1e-9)
        band = (glat > -25) & (glat < 25)
        share = 100 * sig[band].mean()
        vmax = float(np.nanpercentile(np.abs(rf), 99))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        fig, axes = plt.subplots(3, 1, figsize=(13, 13.5), constrained_layout=True)
        pm = axes[0].pcolormesh(glon, glat, rf, cmap="RdBu_r", norm=norm)
        axes[0].set_title(f"{t.upper()}: the random forest's correction from position alone", fontsize=12)
        fig.colorbar(pm, ax=axes[0], shrink=0.85).set_label(f"correction ({UNIT[t]})")
        pm2 = axes[1].pcolormesh(glon, glat, std, cmap="magma", vmin=0, vmax=np.percentile(std, 98))
        axes[1].set_title("how much that correction moves when the model is refitted on resampled dates", fontsize=12)
        fig.colorbar(pm2, ax=axes[1], shrink=0.85).set_label(UNIT[t])
        masked = np.where(sig, rf, np.nan)
        axes[2].set_facecolor("#e5e5e5")
        pm3 = axes[2].pcolormesh(glon, glat, np.ma.masked_invalid(masked), cmap="RdBu_r", norm=norm)
        axes[2].set_title(f"the same correction, keeping only what survives that test "
                          f"(grey = not distinguishable from zero; {share:.0f}% of the tropics survives)", fontsize=12)
        fig.colorbar(pm3, ax=axes[2], shrink=0.85).set_label(f"correction ({UNIT[t]})")
        for ax in axes:
            add_land_overlay(ax, facecolor="#c9c9c9", zorder=5)
            ax.set_xlim(-180, 180); ax.set_ylim(-60, 60)
        fig.suptitle(f"{t.upper()}: is the forest's fine detail real structure or sampling noise?", fontsize=14)
        fig.savefig(OUT / f"{t}_detail_significance.png", dpi=150)
        plt.close(fig)
    print("fig2 done")


def fig3() -> None:
    for t, lat_c in (("tchp", 8.0), ("d26", 8.0)):
        z = np.load(DIAG / f"{t}_grid_predictions.npz")
        glat, glon = z["glat"], z["glon"]
        row = int(np.argmin(np.abs(glat[:, 0] - lat_c)))
        lons = glon[row]
        df = pd.read_parquet(DIAG / f"{t}_oof_predictions.parquet")
        sel = (df.lat > lat_c - 2.5) & (df.lat < lat_c + 2.5)
        d = df[sel].copy()
        d["err"] = d.obs0 - d.mod0
        d["bin"] = (np.floor(d.lon / 10) * 10 + 5)
        g = d.groupby("bin")["err"].agg(["mean", "sem", "count"])
        g = g[g["count"] >= 25]
        fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
        ax.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], fmt="o", color="black",
                    ms=5, capsize=3, lw=1.2, label="observed error, 10° bins (95% interval)", zorder=5)
        for k in ("rf", "gpr", "xgb"):
            ax.plot(lons, z[k][row], lw=1.8, color=COLOR[k], label=LABEL[k], alpha=0.9)
        lz = np.load(LADDER / f"{t}_ladder_grids.npz")
        ax.plot(lons, lz["svr_gamma_x1_baseline"][row], lw=1.6, color="#c4b5fd",
                label="SVR, default bump width (as first run)", alpha=0.9)
        ax.plot(lons, lz["svr_gamma_x16"][row], lw=2.2, color="#7c3aed",
                label="SVR, bumps 16× narrower (best model)", alpha=0.95)
        moe = lz["MoE_blend_best_model_binned_OOF_correction"][row]
        ax.plot(lons, moe, lw=1.4, color="#0f172a", ls=":", label="MoE blend, full 35 features (reference)")
        ax.axhline(0, color="#94a3b8", lw=0.8)
        ax.set_xlim(-180, 180)
        ax.set_xlabel("Longitude"); ax.set_ylabel(f"correction ({UNIT[t]})")
        ax.set_title(f"{t.upper()}: what each model predicts along {lat_c:.0f}°N, against what the floats measured\n"
                     f"observations pooled over {lat_c-2.5:.0f}–{lat_c+2.5:.0f}°N; the tuned SVR recovers the forest's detail without its jitter", fontsize=13)
        ax.grid(alpha=0.15); ax.legend(fontsize=8.5, ncol=3, loc="upper left")
        fig.savefig(OUT / f"{t}_transect.png", dpi=160)
        plt.close(fig)
    print("fig3 done")


def fig4() -> None:
    b = pd.read_csv(PG / "boundary_emergence.csv")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for j, t in enumerate(("tchp", "d26")):
        s = b[(b.target == t) & (b.step != "raw RTOFS")].reset_index(drop=True)
        raw = b[(b.target == t) & (b.step == "raw RTOFS")].iloc[0]
        x = np.arange(len(s))
        for i, (col, lab, c) in enumerate((("mae_boundary", "boundary rows: one side has no 26 °C water", "#dc2626"),
                                           ("mae_warm", "warm rows: both sides have 26 °C water", "#2563eb"))):
            ax = axes[i, j]
            ax.plot(x, s[col], "o-", color=c, lw=2.2, ms=7)
            ax.axhline(raw[col], color="#94a3b8", ls="--", lw=1.2)
            ax.text(len(s) - 1, raw[col], f" raw RTOFS {raw[col]:.2f}", color="#64748b",
                    fontsize=9, va="bottom", ha="right")
            for k in range(1, len(s)):
                d = s[col].iloc[k] - s[col].iloc[k - 1]
                if abs(d) > 0.02 * s[col].iloc[0]:
                    ax.annotate(f"{d:+.2f}", (k, s[col].iloc[k]), textcoords="offset points",
                                xytext=(0, -18 if d < 0 else 10), ha="center", fontsize=9,
                                color="#166534" if d < 0 else "#b91c1c", weight="bold")
            ax.set_xticks(x)
            short = {"position": "position", "+ calendar": "+calendar",
                     "+ raw model value (0 where no 26C)": "+model value",
                     "+ SSH / MLT / SBLT": "+SSH, MLT,\nSBLT",
                     "+ temperature excess above 26C": "+temperature\nexcess",
                     "+ SST neighbourhood": "+SST\ncontext"}
            ax.set_xticklabels([short.get(w, w) for w in s.step], fontsize=9)
            ax.set_ylabel(f"MAE ({UNIT[t]})")
            ax.set_title(f"{t.upper()} — {lab}", fontsize=11.5)
            ax.grid(alpha=0.15)
            lo, hi = s[col].min(), max(s[col].max(), raw[col])
            ax.set_ylim(lo - 0.12 * (hi - lo) - 0.2, hi + 0.15 * (hi - lo) + 0.2)
    fig.suptitle("The ~8,000 boundary cases: temperature excess is an edge detector\n"
                 "features added cumulatively (left to right); it removes 7 m at the 26 °C edge while the warm rows do not improve",
                 fontsize=14)
    fig.savefig(OUT / "boundary_emergence_waterfall.png", dpi=160)
    plt.close(fig)
    print("fig4 done")


def fig5() -> None:
    sc = pd.read_csv(RED / "jacobs_feature_scores.csv")
    pg = pd.read_csv(PG / "prune_graft_results.csv")
    bnd = pd.read_csv(PG / "boundary_emergence.csv")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for ax, t in zip(axes, ("tchp", "d26")):
        core = float(pg[(pg.target == t) & (pg.name == "core only")].mae.iloc[0])
        g = pg[(pg.target == t) & (pg.kind == "graft-single")].copy()
        g["gain"] = core - g.mae
        j = sc[sc.target == t].set_index("feature")
        g["corr"] = [j.loc[n, "abs_corr_to_error"] if n in j.index else np.nan for n in g.name]
        g = g.dropna(subset=["corr"])
        ax.axhline(0, color="#94a3b8", lw=0.8)
        ax.scatter(g["corr"], g["gain"], s=45, c="#94a3b8", edgecolor="white", zorder=3)
        for _, r in g.iterrows():
            if r.gain > 0.05 or r["corr"] > 0.3:
                ax.annotate(r["name"], (r["corr"], r.gain), textcoords="offset points",
                            xytext=(6, 4), fontsize=7.5, color="#334155")
        te = j.loc["model_temp_excess_26c"] if "model_temp_excess_26c" in j.index else None
        if te is not None:
            row = bnd[(bnd.target == t)].reset_index(drop=True)
            i = row.index[row.step.str.contains("temperature excess")][0]
            bgain = row.mae_boundary.iloc[i - 1] - row.mae_boundary.iloc[i]
            wgain = row.mae_warm.iloc[i - 1] - row.mae_warm.iloc[i]
            ax.scatter([te.abs_corr_to_error], [wgain], s=200, marker="*", c="#dc2626", zorder=5)
            ax.annotate(f"model_temp_excess_26c\ncorrelation {te.abs_corr_to_error:.3f}; on warm rows it gains {wgain:+.2f}\n"
                        f"but on the boundary rows it gains {bgain:+.2f} {UNIT[t]}",
                        (te.abs_corr_to_error, wgain), textcoords="offset points", xytext=(20, 35),
                        fontsize=9, color="#dc2626", weight="bold",
                        arrowprops=dict(arrowstyle="->", color="#dc2626"))
        ax.set_xlabel("|Spearman correlation| with the forecast error")
        ax.set_ylabel(f"MAE gained when grafted onto the physical core ({UNIT[t]})")
        ax.set_title(f"{t.upper()}", fontsize=12)
        ax.grid(alpha=0.15)
    fig.suptitle("Correlation with the error does not predict a feature's value\n"
                 "the most useful features are not the most correlated, and the boundary-case detector has almost no correlation at all",
                 fontsize=14)
    fig.savefig(OUT / "correlation_vs_value.png", dpi=160)
    plt.close(fig)
    print("fig5 done")


def fig6() -> None:
    sc = pd.read_csv(LADDER / "smooth_ladder_scores.csv")
    ref = {"tchp": 12.441, "d26": 12.503}
    for t in ("tchp", "d26"):
        z = np.load(LADDER / f"{t}_ladder_grids.npz")
        keys = [k for k in z.files if k not in ("glat", "glon")]
        glat, glon = z["glat"], z["glon"]
        maes = {r.variant: r.mae_warm for _, r in sc[sc.target == t].iterrows()}
        def pretty(k):
            return (k.replace("_", " ").replace("deg", "°").replace("minus", "−"))
        allv = np.concatenate([z[k][np.isfinite(z[k])].ravel() for k in keys])
        vmax = float(np.nanpercentile(np.abs(allv), 99))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        ncol = 3; nrow = int(np.ceil(len(keys) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(20, 3.3 * nrow), constrained_layout=True)
        for ax, k in zip(axes.ravel(), keys):
            pm = ax.pcolormesh(glon, glat, np.ma.masked_invalid(z[k]), cmap="RdBu_r", norm=norm)
            add_land_overlay(ax, facecolor="#c9c9c9", zorder=5)
            ax.set_xlim(-180, 180); ax.set_ylim(-40, 40); ax.set_xticks([]); ax.set_yticks([])
            name = pretty(k)
            hit = [v for v in maes if v.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("°", "deg") == k]
            extra = f"  —  MAE {maes[hit[0]]:.2f}" if hit else ""
            ax.set_title(name + extra, fontsize=9.5)
        for ax in axes.ravel()[len(keys):]:
            ax.axis("off")
        fig.colorbar(pm, ax=axes.ravel().tolist(), shrink=0.55, pad=0.01).set_label(f"correction ({UNIT[t]})")
        fig.suptitle(f"{t.upper()}: how detailed a smooth model is, is a setting we choose\n"
                     f"random forest reference {ref[t]:.2f} {UNIT[t]}; the tuned SVR beats it",
                     fontsize=14)
        fig.savefig(OUT / f"{t}_detail_ladder.png", dpi=140)
        plt.close(fig)
    print("fig6 done")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("wrote", OUT)
