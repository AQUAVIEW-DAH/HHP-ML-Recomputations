"""Summarize monthly OHC collocations, make plots, and write a LaTeX report."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 8: "Aug", 9: "Sep", 10: "Oct",
}


def build_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, group in pairs.groupby("month"):
        g = group.dropna(subset=["obs_tchp_kj_cm2", "model_tchp_kj_cm2", "delta_tchp_kj_cm2"]).copy()
        if g.empty:
            continue
        rows.append({
            "month": month,
            "month_label": MONTH_LABELS.get(int(month), str(month)),
            "n_pairs": len(g),
            "n_platforms": g["platform"].nunique(),
            "n_dates": g["obs_date"].nunique(),
            "obs_mean_kj_cm2": g["obs_tchp_kj_cm2"].mean(),
            "model_mean_kj_cm2": g["model_tchp_kj_cm2"].mean(),
            "mean_delta_kj_cm2": g["delta_tchp_kj_cm2"].mean(),
            "median_delta_kj_cm2": g["delta_tchp_kj_cm2"].median(),
            "mae_kj_cm2": g["delta_tchp_kj_cm2"].abs().mean(),
            "rmse_kj_cm2": (g["delta_tchp_kj_cm2"] ** 2).mean() ** 0.5,
        })
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def build_seasonal_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    season_rows = []
    season_map = {
        "Jan-Mar": [1, 2, 3],
        "Aug-Oct": [8, 9, 10],
    }
    for name, months in season_map.items():
        g = pairs[pairs["month"].isin(months)].dropna(subset=["obs_tchp_kj_cm2", "model_tchp_kj_cm2", "delta_tchp_kj_cm2"]).copy()
        if g.empty:
            continue
        season_rows.append({
            "season": name,
            "months": name,
            "n_pairs": len(g),
            "obs_mean_kj_cm2": g["obs_tchp_kj_cm2"].mean(),
            "model_mean_kj_cm2": g["model_tchp_kj_cm2"].mean(),
            "mean_delta_kj_cm2": g["delta_tchp_kj_cm2"].mean(),
        })
    return pd.DataFrame(season_rows)


def plot_monthly_means(summary: pd.DataFrame, out_path: Path) -> None:
    x = range(len(summary))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar([i - width / 2 for i in x], summary["obs_mean_kj_cm2"], width=width, label="Argo GDAC", color="#0ea5e9")
    ax.bar([i + width / 2 for i in x], summary["model_mean_kj_cm2"], width=width, label="RTOFS", color="#f97316")
    ax.set_xticks(list(x), summary["month_label"])
    ax.set_ylabel("Mean TCHP (kJ/cm²)")
    ax.set_title("Monthly mean collocated OHC/TCHP: Argo vs RTOFS")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_monthly_difference_maps(pairs: pd.DataFrame, out_path: Path) -> None:
    months = sorted(pairs["month"].dropna().unique().tolist())
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axes = axes.ravel()
    v = pairs["delta_tchp_kj_cm2"].dropna()
    vmax = max(abs(v.min()), abs(v.max())) if not v.empty else 1.0
    for ax, month in zip(axes, months):
        g = pairs[pairs["month"] == month].dropna(subset=["delta_tchp_kj_cm2"])
        sc = ax.scatter(
            g["lon"], g["lat"], c=g["delta_tchp_kj_cm2"],
            cmap="coolwarm", vmin=-vmax, vmax=vmax, s=28, edgecolors="k", linewidths=0.2,
        )
        ax.set_title(MONTH_LABELS.get(int(month), str(month)))
        ax.set_xlim(-98, -80)
        ax.set_ylim(18, 31)
        ax.grid(alpha=0.15)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
    for ax in axes[len(months):]:
        ax.axis("off")
    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.95)
    cbar.set_label("Argo - RTOFS TCHP (kJ/cm²)")
    fig.suptitle("Monthly collocated OHC/TCHP differences in the Gulf of Mexico", fontsize=13)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def render_latex(summary: pd.DataFrame, seasonal: pd.DataFrame, selected_dates: pd.DataFrame, out_dir: Path, dates_per_month: int) -> Path:
    tex_path = out_dir / "monthly_ohc_report.tex"
    month_rows = "\n".join(
        f"{r.month_label} & {int(r.n_pairs)} & {int(r.n_platforms)} & {int(r.n_dates)} & "
        f"{r.obs_mean_kj_cm2:.2f} & {r.model_mean_kj_cm2:.2f} & {r.mean_delta_kj_cm2:.2f} \\\\"
        for r in summary.itertuples()
    )
    seasonal_rows = "\n".join(
        f"{r.season} & {int(r.n_pairs)} & {r.obs_mean_kj_cm2:.2f} & {r.model_mean_kj_cm2:.2f} & {r.mean_delta_kj_cm2:.2f} \\\\"
        for r in seasonal.itertuples()
    )
    date_rows = "\n".join(
        f"{MONTH_LABELS.get(int(r.month), str(r.month))} & {r.date} & {int(r.profile_count)} \\\\"
        for r in selected_dates.itertuples()
    )
    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{hyperref}}
\title{{Monthly Gulf of Mexico OHC/TCHP Comparison\\Argo GDAC vs Same-Day RTOFS}}
\date{{}}
\begin{{document}}
\maketitle

\section*{{Overview}}
This report documents a TEOS-10-backed collocation study carried out in the Gulf of Mexico
for 2024. For each month in January--March and August--October, we selected the top {dates_per_month}
Argo-coverage dates in the month, paired each Argo GDAC profile to the nearest same-day RTOFS
ocean column at the profile coordinates, and computed Tropical Cyclone Heat Potential (TCHP)
from both profiles using TEOS-10/GSW thermodynamics.

\section*{{Method}}
For each profile/column pair:
\begin{{itemize}}
\item Practical Salinity was converted to Absolute Salinity using \texttt{{gsw\_SA\_from\_SP}}.
\item Pressure/depth conversion used \texttt{{gsw\_z\_from\_p}} for Argo and \texttt{{gsw\_p\_from\_z}} for RTOFS.
\item Density used \texttt{{gsw\_rho\_t\_exact}} and heat capacity used \texttt{{gsw\_cp\_t\_exact}}.
\item The depth of the 26\,$^\circ$C isotherm ($D_{{26}}$) was found by linear interpolation.
\item TCHP was evaluated as $\int_0^{{D_{{26}}}} \rho(z) c_p(z) [T(z)-26]\,dz$ and reported in kJ/cm$^2$.
\end{{itemize}}

\section*{{Monthly Summary}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lrrrrrr}}
\toprule
Month & Pairs & Floats & Dates & Argo mean & RTOFS mean & Mean diff \\
 &  &  &  & (kJ/cm$^2$) & (kJ/cm$^2$) & (Argo-RTOFS) \\
\midrule
{month_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\section*{{Seasonal Means}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{l r r r r}}
\toprule
Window & Pairs & Argo mean & RTOFS mean & Mean diff \\
 &  & (kJ/cm$^2$) & (kJ/cm$^2$) & (kJ/cm$^2$) \\
\midrule
{seasonal_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\section*{{Selected Sample Dates}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{l l r}}
\toprule
Month & Date & Argo profiles \\
\midrule
{date_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\section*{{Figures}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\textwidth]{{monthly_means.png}}
\caption{{Monthly mean collocated TCHP for Argo GDAC and RTOFS.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{monthly_difference_maps.png}}
\caption{{Spatial distribution of collocated monthly differences, computed as Argo minus RTOFS TCHP.}}
\end{{figure}}

\section*{{Interpretation}}
These monthly means are based on collocated profile dates rather than full-month daily model fields.
That makes the comparison like-for-like at the same locations and times, which is the right framing
for a model-vs-observation OHC benchmark.

\end{{document}}
"""
    tex_path.write_text(tex)
    return tex_path


def ensure_tectonic(tools_dir: Path) -> Path:
    tools_dir.mkdir(parents=True, exist_ok=True)
    binary = tools_dir / "tectonic"
    if binary.exists():
        return binary
    url = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-gnu.tar.gz"
    tar_path = tools_dir / "tectonic.tar.gz"
    subprocess.run(["wget", "-q", "-O", str(tar_path), url], check=True)
    subprocess.run(["tar", "xzf", str(tar_path), "-C", str(tools_dir)], check=True)
    tar_path.unlink(missing_ok=True)
    return binary


def compile_pdf(tex_path: Path, tools_dir: Path) -> Path:
    tectonic = ensure_tectonic(tools_dir).resolve()
    subprocess.run([str(tectonic), str(tex_path.name)], cwd=tex_path.parent, check=True)
    return tex_path.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make monthly OHC plots and LaTeX report.")
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--selected-dates-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dates-per-month", type=int, default=5)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pairs_csv)
    selected_dates = pd.read_csv(args.selected_dates_csv)
    summary = build_summary(pairs)
    seasonal = build_seasonal_summary(pairs)

    summary.to_csv(out_dir / "monthly_summary.csv", index=False)
    seasonal.to_csv(out_dir / "seasonal_summary.csv", index=False)

    plot_monthly_means(summary, out_dir / "monthly_means.png")
    plot_monthly_difference_maps(pairs, out_dir / "monthly_difference_maps.png")

    tex_path = render_latex(summary, seasonal, selected_dates, out_dir, args.dates_per_month)
    pdf_path = compile_pdf(tex_path, out_dir / "tools")
    print(f"summary_csv: {out_dir / 'monthly_summary.csv'}")
    print(f"seasonal_csv: {out_dir / 'seasonal_summary.csv'}")
    print(f"tex: {tex_path}")
    print(f"pdf: {pdf_path}")


if __name__ == "__main__":
    main()
