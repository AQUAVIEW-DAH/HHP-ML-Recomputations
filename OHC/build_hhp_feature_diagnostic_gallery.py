"""Build a skim-friendly diagnostic gallery for HHP ML feature verification.

This script merges the base, global-physics, and profile-physics collocation
tables and renders a consistent plot bundle for the main feature families:

- overview diagnostics for the merged collocation table
- one diagnostic sheet per selected feature
- family overviews for steric-height and Brunt-Vaisala frequency features

The goal is not polished publication graphics. It is fast visual verification:
where the data exist, what the value ranges look like, and where finite rows
drop out across space and time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import add_land_overlay


BASE_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet")
GLOBAL_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_physics.parquet")
PROFILE_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_profile_physics.parquet")
OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/feature_diagnostics/hhp_feature_gallery_2024_2025")

SEASON_MONTHS = {
    "winter_jfm": {1, 2, 3},
    "summer_jas": {7, 8, 9},
}


@dataclass(frozen=True)
class FeatureSpec:
    column: str
    label: str
    units: str
    family: str
    cmap: str
    signed: bool = False
    log_scale: bool = False
    notes: str = ""


FEATURE_SPECS: list[FeatureSpec] = [
    FeatureSpec("model_surface_temp_c", "Surface Temperature", "deg C", "global_physics", "coolwarm", signed=True, notes="RTOFS daily diagnostic at collocated point."),
    FeatureSpec("model_ssh_m", "Sea-Surface Height", "m", "global_physics", "coolwarm", signed=True, notes="RTOFS diagnostic SSH, not a direct satellite observation in this pipeline."),
    FeatureSpec("model_mixed_layer_thickness_m", "Mixed-Layer Thickness", "m", "global_physics", "viridis", notes="RTOFS MLT diagnostic sampled at collocated point."),
    FeatureSpec("model_surface_boundary_layer_thickness_m", "Surface Boundary-Layer Thickness", "m", "global_physics", "viridis", notes="RTOFS SBLT diagnostic sampled at collocated point."),
    FeatureSpec("model_temp_excess_26c", "Surface Temp Excess Above 26C", "deg C", "global_physics", "coolwarm", signed=True, notes="T_s - 26C from model surface temperature."),
    FeatureSpec("d26_minus_mlt_m", "D26 Minus MLT", "m", "global_physics", "viridis", notes="Positive warm-layer separation diagnostic."),
    FeatureSpec("d26_minus_sblt_m", "D26 Minus SBLT", "m", "global_physics", "viridis", notes="Positive warm-layer separation diagnostic."),
    FeatureSpec("d26_to_mlt_ratio", "D26 / MLT", "ratio", "global_physics", "magma", notes="Relative depth scaling diagnostic."),
    FeatureSpec("d26_to_sblt_ratio", "D26 / SBLT", "ratio", "global_physics", "magma", notes="Relative depth scaling diagnostic."),
    FeatureSpec("warm_layer_thickness_positive_m", "max(D26 - MLT, 0)", "m", "global_physics", "viridis", notes="Warm-layer thickness clipped at zero."),
    FeatureSpec("model_steric_0_1000_m", "Steric Height 0/1000", "m", "steric_height", "viridis", notes="TEOS-10 dynamic height converted to steric height."),
    FeatureSpec("model_steric_0_2000_m", "Steric Height 0/2000", "m", "steric_height", "viridis", notes="TEOS-10 dynamic height converted to steric height."),
    FeatureSpec("model_steric_1000_ref2000_m", "Steric Height 1000 Ref 2000", "m", "steric_height", "plasma", notes="steric(0/2000) - steric(0/1000)."),
    FeatureSpec("model_n2_mean_upper200_s2", "Mean N^2 Upper 200 m", "s^-2", "brunt_vaisala", "cividis", log_scale=True, notes="TEOS-10 Brunt-Vaisala frequency summary."),
    FeatureSpec("model_n2_max_upper200_s2", "Max N^2 Upper 200 m", "s^-2", "brunt_vaisala", "cividis", log_scale=True, notes="TEOS-10 Brunt-Vaisala frequency summary."),
    FeatureSpec("model_n2_mean_to_d26_s2", "Mean N^2 to D26", "s^-2", "brunt_vaisala", "cividis", log_scale=True, notes="TEOS-10 Brunt-Vaisala frequency summary."),
    FeatureSpec("model_n2_max_to_d26_s2", "Max N^2 to D26", "s^-2", "brunt_vaisala", "cividis", log_scale=True, notes="TEOS-10 Brunt-Vaisala frequency summary."),
]


def _merge_feature_tables() -> pd.DataFrame:
    base = pd.read_parquet(BASE_PATH).reset_index(drop=True)
    global_df = pd.read_parquet(GLOBAL_PATH).reset_index(drop=True)
    profile_df = pd.read_parquet(PROFILE_PATH).reset_index(drop=True)

    key_cols = [
        "date",
        "year",
        "month",
        "lat",
        "lon",
        "nearest_rtofs_grid_distance_km",
        "argo_tchp_kj_per_cm2",
        "argo_d26_m",
        "model_interp_tchp_kj_per_cm2",
        "model_interp_d26_m",
        "delta_tchp_kj_per_cm2",
        "delta_d26_m",
    ]
    for name, df in [("global", global_df), ("profile", profile_df)]:
        if len(df) != len(base):
            raise RuntimeError(f"{name} table length {len(df)} does not match base length {len(base)}")
        for col in key_cols:
            if col not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(base[col]) and pd.api.types.is_numeric_dtype(df[col]):
                equal = np.allclose(np.asarray(base[col], dtype=float), np.asarray(df[col], dtype=float), equal_nan=True)
            else:
                equal = base[col].astype(str).equals(df[col].astype(str))
            if not equal:
                raise RuntimeError(f"{name} feature table is not aligned with base rows for column {col}")

    extra_global = [c for c in global_df.columns if c not in base.columns]
    extra_profile = [c for c in profile_df.columns if c not in base.columns and c not in extra_global]
    merged = pd.concat([base, global_df[extra_global], profile_df[extra_profile]], axis=1)
    merged["abs_lat"] = np.abs(merged["lat"].to_numpy(float))
    return merged


def _prepare_out_dirs() -> dict[str, Path]:
    paths = {
        "root": OUT_DIR,
        "feature_sheets": OUT_DIR / "feature_sheets",
        "family_overviews": OUT_DIR / "family_overviews",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _set_world_axes(ax) -> None:
    add_land_overlay(ax, zorder=5)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-80.0, 80.0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.15, linewidth=0.4)


def _finite_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _value_limits(values: np.ndarray, *, signed: bool, log_scale: bool) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    if log_scale:
        positive = finite[finite > 0.0]
        if positive.size == 0:
            return 1e-8, 1.0
        return float(np.quantile(positive, 0.05)), float(np.quantile(positive, 0.95))
    q05 = float(np.quantile(finite, 0.05))
    q95 = float(np.quantile(finite, 0.95))
    if signed:
        bound = max(abs(q05), abs(q95), 1e-12)
        return -bound, bound
    if q05 == q95:
        q95 = q05 + 1.0
    return q05, q95


def _make_norm(values: np.ndarray, spec: FeatureSpec):
    vmin, vmax = _value_limits(values, signed=spec.signed, log_scale=spec.log_scale)
    if spec.log_scale:
        return LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, vmin * 1.01))
    if spec.signed:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def _render_overview(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    ax = axes[0, 0]
    for year, color in [(2024, "#3b82f6"), (2025, "#ef4444")]:
        sub = df[df["year"] == year]
        ax.scatter(sub["lon"], sub["lat"], s=5, alpha=0.20, linewidths=0, c=color, label=str(year), zorder=3)
    _set_world_axes(ax)
    ax.set_title("All collocated rows by year")
    ax.legend(loc="lower left", frameon=True)

    ax = axes[0, 1]
    monthly = df.groupby(["year", "month"]).size().reset_index(name="rows")
    month_labels = []
    bar_x = np.arange(len(monthly))
    colors = ["#3b82f6" if y == 2024 else "#ef4444" for y in monthly["year"]]
    ax.bar(bar_x, monthly["rows"], color=colors)
    for _, row in monthly.iterrows():
        month_labels.append(f"{int(row['year'])}-{int(row['month']):02d}")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(month_labels, rotation=60, ha="right", fontsize=8)
    ax.set_title("Monthly collocation row counts")
    ax.set_ylabel("Rows")

    ax = axes[1, 0]
    dist = pd.to_numeric(df["nearest_rtofs_grid_distance_km"], errors="coerce")
    finite = dist[np.isfinite(dist)]
    ax.hist(finite, bins=50, color="#2563eb", alpha=0.85)
    q50 = float(np.quantile(finite, 0.50))
    q95 = float(np.quantile(finite, 0.95))
    ax.axvline(q50, color="black", linestyle="--", linewidth=1.0, label=f"p50={q50:.1f} km")
    ax.axvline(q95, color="#dc2626", linestyle="--", linewidth=1.0, label=f"p95={q95:.1f} km")
    ax.set_title("Nearest RTOFS grid distance")
    ax.set_xlabel("km")
    ax.set_ylabel("Rows")
    ax.legend()

    ax = axes[1, 1]
    feature_names = [spec.column for spec in FEATURE_SPECS]
    years = sorted(df["year"].dropna().astype(int).unique())
    mat = np.full((len(feature_names), len(years) * 12), np.nan, dtype=float)
    xticklabels = []
    total_counts = df.groupby(["year", "month"]).size()
    for yi, year in enumerate(years):
        for month in range(1, 13):
            xticklabels.append(f"{year % 100:02d}-{month:02d}")
            total = total_counts.get((year, month), 0)
            for fi, feature in enumerate(feature_names):
                finite_rows = df[(df["year"] == year) & (df["month"] == month)][feature]
                finite_n = pd.to_numeric(finite_rows, errors="coerce").notna().sum()
                mat[fi, yi * 12 + (month - 1)] = np.nan if total == 0 else finite_n / total
    img = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels([spec.label for spec in FEATURE_SPECS], fontsize=8)
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=60, ha="right", fontsize=7)
    ax.set_title("Finite-row fraction by feature and month")
    fig.colorbar(img, ax=ax, label="finite / total")

    fig.suptitle("HHP feature diagnostics overview", fontsize=16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _render_feature_sheet(df: pd.DataFrame, spec: FeatureSpec, out_path: Path) -> dict:
    vals = _finite_series(df, spec.column)
    finite_mask = np.isfinite(vals)
    finite_vals = vals[finite_mask].to_numpy(float)
    norm = _make_norm(finite_vals, spec)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)

    for ax, title, season_key in [
        (axes[0, 0], "Annual finite values", None),
        (axes[0, 1], "Winter (JFM) finite values", "winter_jfm"),
        (axes[0, 2], "Summer (JAS) finite values", "summer_jas"),
    ]:
        if season_key is None:
            sub = df[finite_mask]
        else:
            months = SEASON_MONTHS[season_key]
            sub = df[finite_mask & df["month"].isin(months)]
        if not sub.empty:
            artist = ax.scatter(
                sub["lon"],
                sub["lat"],
                c=pd.to_numeric(sub[spec.column], errors="coerce"),
                s=9,
                alpha=0.85,
                linewidths=0,
                cmap=spec.cmap,
                norm=norm,
                zorder=3,
            )
            cb = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(spec.units)
        _set_world_axes(ax)
        ax.set_title(f"{title}\nrows={len(sub):,}")

    ax = axes[1, 0]
    if finite_vals.size > 0:
        if spec.log_scale:
            positive = finite_vals[finite_vals > 0.0]
            ax.hist(positive, bins=40, color="#2563eb", alpha=0.85)
            ax.set_xscale("log")
        else:
            ax.hist(finite_vals, bins=40, color="#2563eb", alpha=0.85)
        ax.set_xlabel(spec.units)
        ax.set_ylabel("Rows")
    ax.set_title("Value distribution")
    if spec.notes:
        ax.text(0.02, 0.98, spec.notes, transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#cccccc"})

    ax = axes[1, 1]
    years = sorted(df["year"].dropna().astype(int).unique())
    heat = np.full((len(years), 12), np.nan, dtype=float)
    totals = df.groupby(["year", "month"]).size()
    for yi, year in enumerate(years):
        for month in range(1, 13):
            total = totals.get((year, month), 0)
            sub = df[(df["year"] == year) & (df["month"] == month)]
            finite_n = pd.to_numeric(sub[spec.column], errors="coerce").notna().sum()
            heat[yi, month - 1] = np.nan if total == 0 else finite_n / total
    img = ax.imshow(heat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_yticks(np.arange(len(years)))
    ax.set_yticklabels(years)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels([f"{m:02d}" for m in range(1, 13)])
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title("Finite-row fraction")
    for yi in range(len(years)):
        for mi in range(12):
            if np.isfinite(heat[yi, mi]):
                ax.text(mi, yi, f"{heat[yi, mi]:.2f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.02, label="finite / total")

    ax = axes[1, 2]
    missing = df[~finite_mask]
    ax.scatter(df["lon"], df["lat"], s=6, alpha=0.06, c="#6b7280", linewidths=0, zorder=2)
    if not missing.empty:
        ax.scatter(missing["lon"], missing["lat"], s=12, alpha=0.55, c="#dc2626", linewidths=0, zorder=3)
    _set_world_axes(ax)
    ax.set_title(f"Rows removed for this feature\nmissing={len(missing):,} / total={len(df):,}")

    fig.suptitle(f"{spec.label} diagnostics ({spec.column})", fontsize=16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {
        "feature": spec.column,
        "label": spec.label,
        "family": spec.family,
        "units": spec.units,
        "finite_rows": int(finite_mask.sum()),
        "missing_rows": int((~finite_mask).sum()),
        "plot": str(out_path),
    }


def _render_family_overview(df: pd.DataFrame, specs: list[FeatureSpec], out_path: Path, title: str) -> None:
    n = len(specs)
    ncols = 2 if n > 2 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.5 * ncols, 5.5 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, spec in zip(axes, specs):
        vals = _finite_series(df, spec.column)
        finite_mask = np.isfinite(vals)
        finite_vals = vals[finite_mask].to_numpy(float)
        if finite_vals.size > 0:
            norm = _make_norm(finite_vals, spec)
            artist = ax.scatter(
                df.loc[finite_mask, "lon"],
                df.loc[finite_mask, "lat"],
                c=finite_vals,
                s=10,
                alpha=0.85,
                linewidths=0,
                cmap=spec.cmap,
                norm=norm,
                zorder=3,
            )
            cb = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(spec.units)
        _set_world_axes(ax)
        ax.set_title(f"{spec.label}\nfinite rows={int(finite_mask.sum()):,}")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_index(manifest: dict, out_path: Path) -> None:
    lines = [
        "# HHP Feature Diagnostic Gallery",
        "",
        "This folder is organized for quick skim-review and Drive upload.",
        "",
        f"- Overview: `{manifest['overview_plot']}`",
        f"- Steric family overview: `{manifest['family_overviews']['steric_height']}`",
        f"- Brunt-Vaisala family overview: `{manifest['family_overviews']['brunt_vaisala']}`",
        "",
        "## Feature sheets",
        "",
    ]
    for item in manifest["feature_sheets"]:
        lines.extend([
            f"### {item['label']} (`{item['feature']}`)",
            f"- Family: `{item['family']}`",
            f"- Units: `{item['units']}`",
            f"- Finite rows: `{item['finite_rows']}`",
            f"- Missing rows: `{item['missing_rows']}`",
            f"- Plot: `{item['plot']}`",
            "",
        ])
    out_path.write_text("\n".join(lines))


def main() -> None:
    paths = _prepare_out_dirs()
    df = _merge_feature_tables()

    overview_path = paths["root"] / "overview_collocation_gallery.png"
    _render_overview(df, overview_path)

    manifest = {
        "base_path": str(BASE_PATH),
        "global_path": str(GLOBAL_PATH),
        "profile_path": str(PROFILE_PATH),
        "rows_total": int(len(df)),
        "overview_plot": str(overview_path),
        "family_overviews": {},
        "feature_sheets": [],
    }

    for spec in FEATURE_SPECS:
        out_path = paths["feature_sheets"] / f"{spec.family}__{spec.column}.png"
        manifest["feature_sheets"].append(_render_feature_sheet(df, spec, out_path))

    for family, title in [
        ("steric_height", "Steric-height feature family overview"),
        ("brunt_vaisala", "Brunt-Vaisala feature family overview"),
    ]:
        family_specs = [spec for spec in FEATURE_SPECS if spec.family == family]
        out_path = paths["family_overviews"] / f"{family}_overview.png"
        _render_family_overview(df, family_specs, out_path, title)
        manifest["family_overviews"][family] = str(out_path)

    manifest_path = paths["root"] / "gallery_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _write_index(manifest, paths["root"] / "README.md")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
