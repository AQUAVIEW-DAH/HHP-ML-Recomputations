"""Build density-scatter, patch, and error-relation diagnostics for HHP outputs.

This script is aimed at the current point-collocated HHP workflow:

- observed Argo TCHP/D26 values at collocated rows
- raw RTOFS sampled to those exact rows
- current best locked corrected predictions from the semi-ablation pass

It generates:

- global observed-vs-model density scatter plots
- 4 macro-region observed-vs-model density scatter plots
- 20-degree patch support maps
- top-patch observed-vs-model density scatter plots
- signed-error PDF and absolute-error CDF plots
- feature-binned MAE plots showing how raw vs corrected error varies by regime
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
import sys

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "figure.titlesize": 16})
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC.seasonal_map_common import add_land_overlay  # noqa: E402


OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/density_scatter_diagnostics_2024_2025")
LOCKED_TCHP_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/locked_physics_semi_ablation_predictions_tchp.parquet")
LOCKED_D26_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/locked_physics_semi_ablation_predictions_d26.parquet")
BASE_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet")
GLOBAL_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_physics.parquet")
PROFILE_PATH = Path("/home/suramya/HHP-Prediction/OHC/output/ml_collocation/data/argo_rtofs_collocated_2024_2025_profile_physics.parquet")

PATCH_SIZE_DEG = 20
PATCH_MIN_ROWS = 50
PATCH_TOP_K_PER_REGION = 3
PDF_BINS = 60


@dataclass(frozen=True)
class TargetConfig:
    name: str
    path: Path
    obs_col: str
    raw_col: str
    corrected_col: str
    corrected_label: str
    units: str
    short_label: str
    feature_cols: tuple[str, ...]


TARGETS = {
    "tchp": TargetConfig(
        name="tchp",
        path=LOCKED_TCHP_PATH,
        obs_col="argo_tchp_kj_per_cm2",
        raw_col="pred_obs__raw_rtofs",
        corrected_col="pred_obs__global_pruned",
        corrected_label="global_pruned",
        units="kJ/cm²",
        short_label="TCHP",
        feature_cols=(
            "model_ssh_m",
            "model_temp_excess_26c",
            "d26_minus_mlt_m",
            "model_steric_1000_ref2000_m",
            "model_n2_max_upper200_s2",
            "model_n2_mean_to_d26_s2",
        ),
    ),
    "d26": TargetConfig(
        name="d26",
        path=LOCKED_D26_PATH,
        obs_col="argo_d26_m",
        raw_col="pred_obs__raw_rtofs",
        corrected_col="pred_obs__drop_both_lat_interactions",
        corrected_label="drop_both_lat_interactions",
        units="m",
        short_label="D26",
        feature_cols=(
            "model_mixed_layer_thickness_m",
            "d26_minus_mlt_m",
            "d26_to_sblt_ratio",
            "model_steric_1000_ref2000_m",
            "model_n2_max_upper200_s2",
            "model_n2_mean_to_d26_s2",
        ),
    ),
}

REGION_ORDER = ["atlantic", "pacific", "indian", "high_latitude"]
REGION_DISPLAY = {
    "atlantic": "Atlantic",
    "pacific": "Pacific",
    "indian": "Indian",
    "high_latitude": "High Latitude",
}


@dataclass(frozen=True)
class NamedBox:
    key: str
    display: str
    basin: str
    lat0: int
    lon0: int

    @property
    def patch_label(self) -> str:
        return _patch_label(self.lat0, self.lon0)


# Fixed named 20° boxes for the mentor-requested regional breakdown.
# Rows are selected purely by box geometry (patch_lat0/patch_lon0), independent
# of the automatic macro-region assignment, so boxes that straddle the
# atlantic/indian/pacific longitude splits stay whole.
NAMED_BOXES = (
    NamedBox("gulf_of_mexico", "Gulf of Mexico & NW Caribbean", "Atlantic", 20, -100),
    NamedBox("atlantic_mdr", "Atlantic hurricane MDR / E Caribbean", "Atlantic", 0, -60),
    NamedBox("arabian_sea", "Arabian Sea", "Indian", 0, 60),
    NamedBox("bay_of_bengal", "Bay of Bengal", "Indian", 0, 80),
    NamedBox("philippine_sea", "Philippine Sea / W Pacific warm pool", "West Pacific", 0, 120),
    NamedBox("coral_sea", "Coral Sea", "South Pacific", -20, 140),
    NamedBox("central_eq_pacific", "Central equatorial N Pacific", "Central Pacific", 0, -180),
    NamedBox("sw_pacific_fiji", "SW Pacific (Fiji sector)", "South Pacific", -20, -180),
)


def _canonical_date_str(values: pd.Series) -> pd.Series:
    dt = pd.to_datetime(values, errors="coerce")
    out = dt.dt.strftime("%Y-%m-%d")
    return out


def _ensure_dirs() -> dict[str, Path]:
    paths = {
        "root": OUT_DIR,
        "global": OUT_DIR / "global_density",
        "presentation": OUT_DIR / "presentation_density",
        "regions": OUT_DIR / "region_density",
        "patches": OUT_DIR / "patch_density",
        "named_boxes": OUT_DIR / "named_box_density",
        "errors": OUT_DIR / "error_distributions",
        "features": OUT_DIR / "feature_relations",
        "tables": OUT_DIR / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


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
            raise RuntimeError(f"{name} feature table length mismatch with base")
        for col in key_cols:
            if col not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(base[col]) and pd.api.types.is_numeric_dtype(df[col]):
                equal = np.allclose(np.asarray(base[col], dtype=float), np.asarray(df[col], dtype=float), equal_nan=True)
            else:
                equal = base[col].astype(str).equals(df[col].astype(str))
            if not equal:
                raise RuntimeError(f"{name} feature table is not aligned with base rows for {col}")
    extra_global = [c for c in global_df.columns if c not in base.columns]
    extra_profile = [c for c in profile_df.columns if c not in base.columns and c not in extra_global]
    merged = pd.concat([base, global_df[extra_global], profile_df[extra_profile]], axis=1)
    merged["date"] = _canonical_date_str(merged["date"])
    return merged


def _macro_region(lat: float, lon: float) -> str:
    if not np.isfinite(lat) or not np.isfinite(lon):
        return "unknown"
    if abs(lat) > 45:
        return "high_latitude"
    if -100 <= lon < 20:
        return "atlantic"
    if 20 <= lon < 147:
        return "indian"
    return "pacific"


def _patch_floor_lon(lon: float) -> int:
    lon_wrapped = ((float(lon) + 180.0) % 360.0) - 180.0
    if lon_wrapped == 180.0:
        lon_wrapped = 179.999999
    return int(np.floor(lon_wrapped / PATCH_SIZE_DEG) * PATCH_SIZE_DEG)


def _patch_floor_lat(lat: float) -> int:
    lat_clip = min(max(float(lat), -89.999999), 89.999999)
    return int(np.floor(lat_clip / PATCH_SIZE_DEG) * PATCH_SIZE_DEG)


def _patch_label(lat0: int, lon0: int) -> str:
    return f"lat[{lat0},{lat0 + PATCH_SIZE_DEG}) lon[{lon0},{lon0 + PATCH_SIZE_DEG})"


def _augment_regions_and_patches(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["macro_region"] = [_macro_region(lat, lon) for lat, lon in zip(out["lat"], out["lon"])]
    out["patch_lat0"] = [_patch_floor_lat(lat) for lat in out["lat"]]
    out["patch_lon0"] = [_patch_floor_lon(lon) for lon in out["lon"]]
    out["patch_label"] = [_patch_label(lat0, lon0) for lat0, lon0 in zip(out["patch_lat0"], out["patch_lon0"])]
    return out


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    return sqrt(np.mean((x - y) ** 2))


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.allclose(np.std(x), 0.0):
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "rows": int(x.size),
        "rmse": float(_rmse(x, y)),
        "mae": float(np.mean(np.abs(x - y))),
        "bias": float(np.mean(y - x)),
        "corr": float(pd.Series(x).corr(pd.Series(y))) if x.size > 1 else float("nan"),
        "slope": _slope(x, y),
    }


def _nice_upper(v: float) -> float:
    if not np.isfinite(v) or v <= 0:
        return 1.0
    if v <= 20:
        step = 2
    elif v <= 100:
        step = 10
    elif v <= 250:
        step = 25
    else:
        step = 50
    return float(ceil(v / step) * step)


def _common_axis_limits(obs: np.ndarray, raw: np.ndarray, corrected: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    finite = np.concatenate([obs[np.isfinite(obs)], raw[np.isfinite(raw)], corrected[np.isfinite(corrected)]])
    lower = float(min(0.0, np.nanquantile(finite, 0.001)))
    upper = _nice_upper(float(np.nanquantile(finite, 0.995)))
    return (lower, upper), (lower, upper)


def _density_image(x: np.ndarray, y: np.ndarray, *, xlim: tuple[float, float], ylim: tuple[float, float], bins: int = PDF_BINS):
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim], density=False)
    total = hist.sum()
    pdf = hist / total if total > 0 else hist
    log_pdf = np.full_like(pdf, np.nan, dtype=float)
    mask = pdf > 0
    log_pdf[mask] = np.log10(pdf[mask])
    return log_pdf.T, x_edges, y_edges


def _density_color_limits(*imgs: np.ndarray) -> tuple[float, float]:
    occupied = []
    for img in imgs:
        vals = img[np.isfinite(img)]
        if vals.size:
            occupied.append(vals)
    if not occupied:
        return -1.0, 0.0
    merged = np.concatenate(occupied)
    vmin = float(np.nanmin(merged))
    vmax = float(np.nanmax(merged))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return -1.0, 0.0
    if np.isclose(vmin, vmax):
        pad = 0.25
        return vmin - pad, vmax + pad
    pad = 0.03 * (vmax - vmin)
    return vmin - pad, vmax + pad


def _plot_density_panel(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    bins: int = PDF_BINS,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    color_limits: tuple[float, float] | None = None,
):
    if xlim is None or ylim is None:
        xlim, ylim = _common_axis_limits(x, y, y)
    img, x_edges, y_edges = _density_image(x, y, xlim=xlim, ylim=ylim, bins=bins)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")
    if color_limits is None:
        color_limits = _density_color_limits(img)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        img,
        shading="auto",
        cmap=cmap,
        vmin=color_limits[0],
        vmax=color_limits[1],
    )
    ax.plot(xlim, xlim, linestyle="--", color="black", linewidth=1.0, alpha=0.9)
    # Conditional mean: mean model value in each observed-value bin (mentor
    # request) — makes nonlinearity of the model-vs-observed relation visible.
    finite_xy = np.isfinite(x) & np.isfinite(y)
    if finite_xy.sum() >= 50:
        bin_idx = np.clip(np.digitize(x[finite_xy], x_edges) - 1, 0, len(x_edges) - 2)
        sums = np.bincount(bin_idx, weights=y[finite_xy], minlength=len(x_edges) - 1)
        counts = np.bincount(bin_idx, minlength=len(x_edges) - 1)
        centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        good = counts >= 10
        if good.any():
            ax.plot(centers[good], sums[good] / counts[good], color="black", linewidth=2.2, alpha=0.95)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    m = _metrics(x, y)
    ax.set_title(
        f"{title}\nRMSE={m['rmse']:.2f}, MAE={m['mae']:.2f}, Bias={m['bias']:.2f}, "
        f"Corr={m['corr']:.2f}, Slope={m['slope']:.2f}, n={m['rows']}"
    )
    ax.grid(True, alpha=0.15, linewidth=0.4)
    return mesh, m, img


def _plot_global_density(target: TargetConfig, df: pd.DataFrame, out_path: Path, metrics_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    obs = df[target.obs_col].to_numpy(float)
    raw = df[target.raw_col].to_numpy(float)
    corr = df[target.corrected_col].to_numpy(float)
    xlim, ylim = _common_axis_limits(obs, raw, corr)
    raw_img, _, _ = _density_image(obs, raw, xlim=xlim, ylim=ylim)
    corr_img, _, _ = _density_image(obs, corr, xlim=xlim, ylim=ylim)
    color_limits = _density_color_limits(raw_img, corr_img)

    mesh, m_raw, _ = _plot_density_panel(
        axes[0], obs, raw,
        title=f"{target.short_label}: observed vs raw RTOFS",
        xlabel=f"Observed {target.short_label} ({target.units})",
        ylabel=f"Raw RTOFS {target.short_label} ({target.units})",
        xlim=xlim,
        ylim=ylim,
        color_limits=color_limits,
    )
    _, m_corr, _ = _plot_density_panel(
        axes[1], obs, corr,
        title=f"{target.short_label}: observed vs corrected ({target.corrected_label})",
        xlabel=f"Observed {target.short_label} ({target.units})",
        ylabel=f"Corrected {target.short_label} ({target.units})",
        xlim=xlim,
        ylim=ylim,
        color_limits=color_limits,
    )
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    cbar.set_label("log10(PDF)")
    fig.suptitle(f"Global density-scatter diagnostics for {target.short_label}", fontsize=16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    for model_name, m in [("raw_rtofs", m_raw), (target.corrected_label, m_corr)]:
        row = {"target": target.name, "scope": "global", "subset": "all", "model": model_name}
        row.update(m)
        metrics_rows.append(row)


def _plot_presentation_density(
    target: TargetConfig,
    df: pd.DataFrame,
    out_path: Path,
    *,
    y_col: str,
    label: str,
    metrics_rows: list[dict],
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 7.6), constrained_layout=True)
    obs = df[target.obs_col].to_numpy(float)
    pred = df[y_col].to_numpy(float)
    raw = df[target.raw_col].to_numpy(float)
    corr = df[target.corrected_col].to_numpy(float)
    xlim, ylim = _common_axis_limits(obs, raw, corr)
    raw_img, _, _ = _density_image(obs, raw, xlim=xlim, ylim=ylim)
    corr_img, _, _ = _density_image(obs, corr, xlim=xlim, ylim=ylim)
    color_limits = _density_color_limits(raw_img, corr_img)
    mesh, m, _ = _plot_density_panel(
        ax, obs, pred,
        title="",
        xlabel=f"Observed {target.short_label} ({target.units})",
        ylabel=f"{label} {target.short_label} ({target.units})",
        xlim=xlim,
        ylim=ylim,
        color_limits=color_limits,
    )
    ax.set_title(
        f"{target.short_label} observed vs. {label}\n"
        f"RMS: {m['rmse']:.2f}, MAE: {m['mae']:.2f}, Bias: {m['bias']:.2f}, "
        f"Corr: {m['corr']:.2f}, Slope: {m['slope']:.2f}"
    )
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label("log10(PDF)")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    row = {
        "target": target.name,
        "scope": "presentation_global",
        "subset": "all",
        "model": label,
    }
    row.update(m)
    metrics_rows.append(row)


def _plot_region_density(target: TargetConfig, df: pd.DataFrame, out_path: Path, metrics_rows: list[dict]) -> None:
    # Only render regions with data; high_latitude has no valid TCHP/D26 rows
    # by construction (the 26C isotherm target is undefined there).
    regions = [
        r for r in REGION_ORDER
        if np.isfinite(df.loc[df["macro_region"] == r, target.obs_col].to_numpy(float)).sum() > 0
    ]
    fig, axes = plt.subplots(len(regions), 2, figsize=(15, 5 * len(regions)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    mesh = None
    for i, region in enumerate(regions):
        sub = df[df["macro_region"] == region].copy()
        obs = sub[target.obs_col].to_numpy(float)
        raw = sub[target.raw_col].to_numpy(float)
        corr = sub[target.corrected_col].to_numpy(float)
        region_name = REGION_DISPLAY[region]
        xlim, ylim = _common_axis_limits(obs, raw, corr)
        raw_img, _, _ = _density_image(obs, raw, xlim=xlim, ylim=ylim)
        corr_img, _, _ = _density_image(obs, corr, xlim=xlim, ylim=ylim)
        color_limits = _density_color_limits(raw_img, corr_img)
        mesh, m_raw, _ = _plot_density_panel(
            axes[i, 0], obs, raw,
            title=f"{region_name}: observed vs raw",
            xlabel=f"Observed {target.short_label} ({target.units})",
            ylabel=f"Raw RTOFS {target.short_label} ({target.units})",
            xlim=xlim,
            ylim=ylim,
            color_limits=color_limits,
        )
        _, m_corr, _ = _plot_density_panel(
            axes[i, 1], obs, corr,
            title=f"{region_name}: observed vs corrected",
            xlabel=f"Observed {target.short_label} ({target.units})",
            ylabel=f"Corrected {target.short_label} ({target.units})",
            xlim=xlim,
            ylim=ylim,
            color_limits=color_limits,
        )
        for model_name, m in [("raw_rtofs", m_raw), (target.corrected_label, m_corr)]:
            row = {"target": target.name, "scope": "region", "subset": region, "model": model_name}
            row.update(m)
            metrics_rows.append(row)
        cbar = fig.colorbar(mesh, ax=[axes[i, 0], axes[i, 1]], shrink=0.9, pad=0.01)
        cbar.set_label("log10(PDF)")
    fig.suptitle(f"Macro-region density-scatter diagnostics for {target.short_label}", fontsize=16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _patch_summary(target: TargetConfig, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (region, lat0, lon0, label), g in df.groupby(["macro_region", "patch_lat0", "patch_lon0", "patch_label"]):
        obs = g[target.obs_col].to_numpy(float)
        raw = g[target.raw_col].to_numpy(float)
        corr = g[target.corrected_col].to_numpy(float)
        m_raw = _metrics(obs, raw)
        m_corr = _metrics(obs, corr)
        rows.append({
            "target": target.name,
            "macro_region": region,
            "patch_lat0": int(lat0),
            "patch_lon0": int(lon0),
            "patch_label": label,
            "rows": int(len(g)),
            "dates": int(g["date"].nunique()),
            "raw_rmse": m_raw["rmse"],
            "raw_mae": m_raw["mae"],
            "raw_bias": m_raw["bias"],
            "raw_corr": m_raw["corr"],
            "corrected_rmse": m_corr["rmse"],
            "corrected_mae": m_corr["mae"],
            "corrected_bias": m_corr["bias"],
            "corrected_corr": m_corr["corr"],
            "mae_improvement": m_raw["mae"] - m_corr["mae"],
            "rmse_improvement": m_raw["rmse"] - m_corr["rmse"],
        })
    out = pd.DataFrame(rows).sort_values(["macro_region", "rows", "mae_improvement"], ascending=[True, False, False])
    return out


def _plot_patch_support_map(target: TargetConfig, patch_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
    add_land_overlay(ax, zorder=2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-80, 80)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{target.short_label}: 20° patch support (rows per patch)")
    ax.grid(True, alpha=0.15, linewidth=0.4)

    counts = patch_df["rows"].to_numpy(float)
    norm = Normalize(vmin=float(np.nanmin(counts)), vmax=float(np.nanmax(counts)))
    cmap = plt.get_cmap("viridis")
    for _, row in patch_df.iterrows():
        rect = Rectangle(
            (row["patch_lon0"], row["patch_lat0"]),
            PATCH_SIZE_DEG,
            PATCH_SIZE_DEG,
            facecolor=cmap(norm(row["rows"])),
            edgecolor="white",
            linewidth=0.35,
            alpha=0.75,
            zorder=1,
        )
        ax.add_patch(rect)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("rows in patch")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_top_patch_density(target: TargetConfig, df: pd.DataFrame, patch_df: pd.DataFrame, out_path: Path, metrics_rows: list[dict]) -> None:
    selected = []
    for region in REGION_ORDER:
        sub = patch_df[(patch_df["macro_region"] == region) & (patch_df["rows"] >= PATCH_MIN_ROWS)].copy()
        selected.append(sub.head(PATCH_TOP_K_PER_REGION))
    selected_df = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    if selected_df.empty:
        return

    regions = [r for r in REGION_ORDER if not selected_df[selected_df["macro_region"] == r].empty]
    nrows = len(regions)
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    mesh = None
    for i, region in enumerate(regions):
        subsel = selected_df[selected_df["macro_region"] == region].copy()
        best = subsel.iloc[0]
        g = df[(df["macro_region"] == region) & (df["patch_label"] == best["patch_label"])].copy()
        obs = g[target.obs_col].to_numpy(float)
        raw = g[target.raw_col].to_numpy(float)
        corr = g[target.corrected_col].to_numpy(float)
        region_name = REGION_DISPLAY[region]
        patch_title = f"{best['patch_label']}, n={int(best['rows'])}, dates={int(best['dates'])}"
        xlim, ylim = _common_axis_limits(obs, raw, corr)
        raw_img, _, _ = _density_image(obs, raw, xlim=xlim, ylim=ylim)
        corr_img, _, _ = _density_image(obs, corr, xlim=xlim, ylim=ylim)
        color_limits = _density_color_limits(raw_img, corr_img)
        mesh, m_raw, _ = _plot_density_panel(
            axes[i, 0], obs, raw,
            title=f"{region_name} top patch: raw\n{patch_title}",
            xlabel=f"Observed {target.short_label} ({target.units})",
            ylabel=f"Raw RTOFS {target.short_label} ({target.units})",
            xlim=xlim,
            ylim=ylim,
            color_limits=color_limits,
        )
        _, m_corr, _ = _plot_density_panel(
            axes[i, 1], obs, corr,
            title=f"{region_name} top patch: corrected\n{patch_title}",
            xlabel=f"Observed {target.short_label} ({target.units})",
            ylabel=f"Corrected {target.short_label} ({target.units})",
            xlim=xlim,
            ylim=ylim,
            color_limits=color_limits,
        )
        for model_name, m in [("raw_rtofs", m_raw), (target.corrected_label, m_corr)]:
            row = {"target": target.name, "scope": "top_patch", "subset": best["patch_label"], "macro_region": region, "model": model_name}
            row.update(m)
            metrics_rows.append(row)
        cbar = fig.colorbar(mesh, ax=[axes[i, 0], axes[i, 1]], shrink=0.9, pad=0.01)
        cbar.set_label("log10(PDF)")
    fig.suptitle(f"Top 20° patch density-scatter diagnostics for {target.short_label}", fontsize=16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _named_box_rows(df: pd.DataFrame, box: NamedBox) -> pd.DataFrame:
    return df[(df["patch_lat0"] == box.lat0) & (df["patch_lon0"] == box.lon0)]


def _plot_named_box_map(target: TargetConfig, df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
    add_land_overlay(ax, zorder=2)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-80, 80)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{target.short_label}: named 20° regional boxes (rows per box)")
    ax.grid(True, alpha=0.15, linewidth=0.4)

    for box in NAMED_BOXES:
        n = len(_named_box_rows(df, box))
        rect = Rectangle(
            (box.lon0, box.lat0),
            PATCH_SIZE_DEG,
            PATCH_SIZE_DEG,
            facecolor="#2563eb",
            edgecolor="black",
            linewidth=1.2,
            alpha=0.35,
            zorder=1,
        )
        ax.add_patch(rect)
        ax.text(
            box.lon0 + PATCH_SIZE_DEG / 2,
            box.lat0 + PATCH_SIZE_DEG / 2,
            f"{box.display}\nn={n}",
            ha="center",
            va="center",
            fontsize=8,
            zorder=3,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
        )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_named_box_density(target: TargetConfig, df: pd.DataFrame, out_path: Path, metrics_rows: list[dict]) -> None:
    nrows = len(NAMED_BOXES)
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 4.9 * nrows), constrained_layout=True)
    mesh = None
    for i, box in enumerate(NAMED_BOXES):
        g = _named_box_rows(df, box)
        if len(g) < PATCH_MIN_ROWS:
            for j in range(2):
                axes[i, j].set_title(f"{box.display}: only {len(g)} rows (<{PATCH_MIN_ROWS})")
                axes[i, j].axis("off")
            continue
        obs = g[target.obs_col].to_numpy(float)
        raw = g[target.raw_col].to_numpy(float)
        corr = g[target.corrected_col].to_numpy(float)
        n_dates = int(g["date"].nunique())
        box_title = f"{box.patch_label}, n={len(g)}, dates={n_dates}"
        xlim, ylim = _common_axis_limits(obs, raw, corr)
        raw_img, _, _ = _density_image(obs, raw, xlim=xlim, ylim=ylim)
        corr_img, _, _ = _density_image(obs, corr, xlim=xlim, ylim=ylim)
        color_limits = _density_color_limits(raw_img, corr_img)
        mesh, m_raw, _ = _plot_density_panel(
            axes[i, 0], obs, raw,
            title=f"{box.display} ({box.basin}): raw\n{box_title}",
            xlabel=f"Observed {target.short_label} ({target.units})",
            ylabel=f"Raw RTOFS {target.short_label} ({target.units})",
            xlim=xlim,
            ylim=ylim,
            color_limits=color_limits,
        )
        _, m_corr, _ = _plot_density_panel(
            axes[i, 1], obs, corr,
            title=f"{box.display} ({box.basin}): corrected\n{box_title}",
            xlabel=f"Observed {target.short_label} ({target.units})",
            ylabel=f"Corrected {target.short_label} ({target.units})",
            xlim=xlim,
            ylim=ylim,
            color_limits=color_limits,
        )
        for model_name, m in [("raw_rtofs", m_raw), (target.corrected_label, m_corr)]:
            row = {
                "target": target.name,
                "scope": "named_box",
                "subset": box.key,
                "named_box_display": box.display,
                "basin": box.basin,
                "patch_label": box.patch_label,
                "model": model_name,
            }
            row.update(m)
            metrics_rows.append(row)
        # One colorbar per row: color limits are per-box, so a single
        # figure-level colorbar would only be valid for one row.
        cbar = fig.colorbar(mesh, ax=[axes[i, 0], axes[i, 1]], shrink=0.9, pad=0.01)
        cbar.set_label("log10(PDF)")
    fig.suptitle(f"Named 20° regional box density-scatter diagnostics for {target.short_label}", fontsize=16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_named_box_monthly_support(target: TargetConfig, df: pd.DataFrame, out_path: Path, out_csv: Path) -> None:
    """Heatmap of finite rows per named box per calendar month.

    Blank cells are months with no collocated support. With the current base
    table those gaps are dominated by missing reduced RTOFS daily fields, not
    by Argo coverage, so this panel doubles as a backfill progress check.
    """
    months = pd.period_range("2024-01", "2025-12", freq="M")
    month_labels = [str(m) for m in months]
    counts = np.zeros((len(NAMED_BOXES), len(months)), dtype=int)
    csv_rows = []
    for i, box in enumerate(NAMED_BOXES):
        g = _named_box_rows(df, box)
        per_month = pd.to_datetime(g["date"]).dt.to_period("M").value_counts()
        for j, month in enumerate(months):
            n = int(per_month.get(month, 0))
            counts[i, j] = n
            csv_rows.append({
                "target": target.name,
                "box": box.key,
                "box_display": box.display,
                "month": month_labels[j],
                "rows": n,
            })
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(16, 6.5), constrained_layout=True)
    masked = np.ma.masked_equal(counts, 0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    mesh = ax.pcolormesh(np.arange(len(months) + 1), np.arange(len(NAMED_BOXES) + 1), masked, cmap=cmap)
    ax.set_xticks(np.arange(len(months)) + 0.5)
    ax.set_xticklabels(month_labels, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(NAMED_BOXES)) + 0.5)
    ax.set_yticklabels([f"{b.display}\n{b.patch_label}" for b in NAMED_BOXES], fontsize=8)
    ax.invert_yaxis()
    for i in range(len(NAMED_BOXES)):
        for j in range(len(months)):
            if counts[i, j] > 0:
                ax.text(j + 0.5, i + 0.5, str(counts[i, j]), ha="center", va="center", fontsize=6.5, color="white")
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("valid rows in month")
    ax.set_title(
        f"{target.short_label}: monthly collocation-data support per named 20° box (valid observed rows)\n"
        "Counts reflect the full collocation table; the locked-protocol scatters evaluate a subset "
        "(the earliest blocked-forward block is train-only)"
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_error_distributions(target: TargetConfig, df: pd.DataFrame, out_path: Path) -> dict[str, str]:
    obs = df[target.obs_col].to_numpy(float)
    raw = df[target.raw_col].to_numpy(float)
    corr = df[target.corrected_col].to_numpy(float)
    raw_err = obs - raw
    corr_err = obs - corr
    raw_abs = np.abs(raw_err)
    corr_abs = np.abs(corr_err)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)

    q = float(np.nanquantile(np.abs(np.concatenate([raw_err, corr_err])), 0.995))
    bins = np.linspace(-q, q, 80)
    axes[0].hist(raw_err, bins=bins, density=True, histtype="step", linewidth=2.0, color="#dc2626", label="raw_rtofs")
    axes[0].hist(corr_err, bins=bins, density=True, histtype="step", linewidth=2.0, color="#2563eb", label=target.corrected_label)
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title(f"{target.short_label} signed error PDF")
    axes[0].set_xlabel(f"Observed - predicted ({target.units})")
    axes[0].set_ylabel("density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.15, linewidth=0.4)

    for arr, color, label in [
        (raw_abs, "#dc2626", "raw_rtofs"),
        (corr_abs, "#2563eb", target.corrected_label),
    ]:
        vals = np.sort(arr[np.isfinite(arr)])
        y = np.arange(1, vals.size + 1) / vals.size
        axes[1].plot(vals, y, color=color, linewidth=2.0, label=label)
    axes[1].set_title(f"{target.short_label} absolute error CDF")
    axes[1].set_xlabel(f"|Observed - predicted| ({target.units})")
    axes[1].set_ylabel("fraction <= threshold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.15, linewidth=0.4)

    fig.suptitle(f"Error-distribution diagnostics for {target.short_label}", fontsize=16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {"raw_error_std": float(np.nanstd(raw_err)), "corrected_error_std": float(np.nanstd(corr_err))}


def _feature_bin_summary(df: pd.DataFrame, feature_col: str, raw_abs_col: str, corr_abs_col: str, *, bins: int = 10) -> pd.DataFrame:
    sub = df[[feature_col, raw_abs_col, corr_abs_col]].copy()
    sub[feature_col] = pd.to_numeric(sub[feature_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return pd.DataFrame(columns=["bin_center", "raw_mae", "corrected_mae", "rows"])
    try:
        sub["bin"] = pd.qcut(sub[feature_col], q=min(bins, sub[feature_col].nunique()), duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=["bin_center", "raw_mae", "corrected_mae", "rows"])
    rows = []
    for _, g in sub.groupby("bin", observed=True):
        rows.append({
            "bin_center": float(np.nanmedian(g[feature_col].to_numpy(float))),
            "raw_mae": float(np.nanmean(g[raw_abs_col].to_numpy(float))),
            "corrected_mae": float(np.nanmean(g[corr_abs_col].to_numpy(float))),
            "rows": int(len(g)),
        })
    return pd.DataFrame(rows).sort_values("bin_center")


def _plot_feature_relations(target: TargetConfig, df: pd.DataFrame, out_path: Path, out_csv: Path) -> None:
    work = df.copy()
    work["raw_abs_error"] = np.abs(work[target.obs_col] - work[target.raw_col])
    work["corrected_abs_error"] = np.abs(work[target.obs_col] - work[target.corrected_col])

    ncols = 3
    nrows = int(ceil(len(target.feature_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5.3 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    summary_rows = []

    for ax, feature_col in zip(axes, target.feature_cols):
        bins_df = _feature_bin_summary(work, feature_col, "raw_abs_error", "corrected_abs_error")
        if bins_df.empty:
            ax.set_title(f"{feature_col}\ninsufficient valid rows")
            ax.axis("off")
            continue
        ax.plot(bins_df["bin_center"], bins_df["raw_mae"], marker="o", color="#dc2626", linewidth=2.0, label="raw_rtofs")
        ax.plot(bins_df["bin_center"], bins_df["corrected_mae"], marker="o", color="#2563eb", linewidth=2.0, label=target.corrected_label)
        ax.set_title(feature_col)
        ax.set_xlabel(feature_col)
        ax.set_ylabel(f"mean absolute error ({target.units})")
        ax.grid(True, alpha=0.15, linewidth=0.4)
        ymax = max(float(bins_df["raw_mae"].max()), float(bins_df["corrected_mae"].max()))
        for _, row in bins_df.iterrows():
            ax.text(row["bin_center"], ymax * 1.01, f"{int(row['rows'])}", rotation=90, ha="center", va="bottom", fontsize=7, alpha=0.7)
        summary_rows.append(bins_df.assign(target=target.name, feature=feature_col))

    for ax in axes[len(target.feature_cols):]:
        ax.axis("off")
    axes[0].legend(loc="best")
    fig.suptitle(
        f"{target.short_label}: feature-binned MAE (raw vs corrected)\n"
        "Numbers above points show rows per quantile bin",
        fontsize=16,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    if summary_rows:
        pd.concat(summary_rows, ignore_index=True).to_csv(out_csv, index=False)


def _load_target_predictions(target: TargetConfig, merged_features: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_parquet(target.path).copy()
    pred["date"] = _canonical_date_str(pred["date"])
    pred = _augment_regions_and_patches(pred)
    if target.name == "tchp":
        keys = ["date", "year", "month", "lat", "lon", "argo_tchp_kj_per_cm2", "model_interp_tchp_kj_per_cm2", "delta_tchp_kj_per_cm2"]
    else:
        keys = ["date", "year", "month", "lat", "lon", "argo_d26_m", "model_interp_d26_m", "delta_d26_m"]
    feature_subset = merged_features[keys + list(target.feature_cols)].copy().drop_duplicates()
    out = pred.merge(feature_subset, on=keys, how="left", validate="many_to_one")
    return out


def _named_box_spec_lines() -> str:
    return "\n".join(
        f"- `{box.key}` ({box.basin}): {box.display} — `{box.patch_label}`"
        for box in NAMED_BOXES
    )


def _write_spec_note(paths: dict[str, Path]) -> Path:
    note = f"""# HHP Density-Scatter Diagnostic Spec

This note documents the first-pass HHP diagnostics built in
`build_hhp_density_scatter_diagnostics.py`.

## Goal

Translate the current point-collocated HHP workflow into plots that are easy to
compare with the MLD-style 2-D PDF scatter figures requested by Dr. Jacobs,
while also adding region, patch, and feature-regime context.

## Inputs

- Locked semi-ablation OOF prediction table for TCHP:
  `{LOCKED_TCHP_PATH}`
- Locked semi-ablation OOF prediction table for D26:
  `{LOCKED_D26_PATH}`
- Merged base/global/profile collocation features:
  `{BASE_PATH}`, `{GLOBAL_PATH}`, `{PROFILE_PATH}`

## Corrected model used

- TCHP corrected model: `global_pruned`
- D26 corrected model: `drop_both_lat_interactions`

These are used because they are the current best locked-protocol candidates from
the semi-ablation pass.

## Plot families

1. Global density-scatter:
   observed vs raw RTOFS and observed vs corrected.
2. Presentation density-scatter:
   single-panel log10(PDF) figures in the MLD-style visual family, generated
   separately for raw and corrected predictions.
3. Macro-region density-scatter:
   same observed-vs-model diagnostic across 4 globe partitions.
4. 20-degree patch support + top-patch density-scatter:
   use 20° x 20° boxes and select the best-supported patch inside each region.
5. Named 20-degree regional boxes:
   fixed, named boxes in the main tropical-cyclone basins (map + per-box
   observed-vs-model density scatters + monthly support heatmap). Unlike
   family 4, the box set does not change with data support, so figures stay
   comparable across reruns. Blank months in the support heatmap are
   collocation gaps, currently dominated by missing reduced RTOFS daily
   fields (see the backfill note in `NEXT_SESSION_HANDOFF.md`).
6. Error distributions:
   signed-error PDF and absolute-error CDF.
7. Feature relations:
   binned MAE curves vs selected physics features.

## Macro-region definition

- `high_latitude`: `|lat| > 45`
- `atlantic`: `-100 <= lon < 20` and not high latitude
- `indian`: `20 <= lon < 147` and not high latitude
- `pacific`: remaining longitudes and not high latitude

## 20-degree patch definition

- latitude start: `floor(lat / 20) * 20`
- longitude start: `floor(lon / 20) * 20`
- patch label example:
  `lat[0,20) lon[140,160)`

## Named 20-degree boxes

Rows are selected purely by box geometry (independent of the macro-region
longitude splits):

{_named_box_spec_lines()}

## Density-scatter style

- 2-D histogram on observed/model axes
- color shows `log10(PDF)`
- empty bins shown in white
- 1:1 line overlaid
- subplot title includes:
  `RMSE`, `MAE`, `Bias`, `Corr`, `Slope`, `n`

## Feature-relation style

For selected features, rows are divided into quantile bins. Within each bin the
script plots:

- raw mean absolute error
- corrected mean absolute error
- number of rows in the bin

This was chosen over raw scatter because it is easier to interpret when asking
whether error systematically changes with a physics regime.

## Output root

`{paths['root']}`
"""
    out_path = paths["root"] / "hhp_density_scatter_plot_spec.md"
    out_path.write_text(note)
    return out_path


def main() -> None:
    paths = _ensure_dirs()
    merged_features = _merge_feature_tables()
    spec_path = _write_spec_note(paths)

    metrics_rows: list[dict] = []
    manifest = {"spec_note": str(spec_path), "targets": {}}

    for target in TARGETS.values():
        df = _load_target_predictions(target, merged_features)

        global_path = paths["global"] / f"{target.name}_global_density_scatter.png"
        raw_presentation_path = paths["presentation"] / f"{target.name}_raw_presentation_density_scatter.png"
        corrected_presentation_path = paths["presentation"] / f"{target.name}_corrected_presentation_density_scatter.png"
        region_path = paths["regions"] / f"{target.name}_macro_region_density_scatter.png"
        patch_csv = paths["tables"] / f"{target.name}_patch_metrics.csv"
        patch_map = paths["patches"] / f"{target.name}_patch_support_map.png"
        patch_density = paths["patches"] / f"{target.name}_top_patch_density_scatter.png"
        named_box_map = paths["named_boxes"] / f"{target.name}_named_box_map.png"
        named_box_density = paths["named_boxes"] / f"{target.name}_named_box_density_scatter.png"
        named_box_support = paths["named_boxes"] / f"{target.name}_named_box_monthly_support.png"
        named_box_support_csv = paths["tables"] / f"{target.name}_named_box_monthly_support.csv"
        error_path = paths["errors"] / f"{target.name}_error_pdf_cdf.png"
        feature_path = paths["features"] / f"{target.name}_feature_binned_mae.png"
        feature_csv = paths["tables"] / f"{target.name}_feature_binned_mae.csv"

        _plot_global_density(target, df, global_path, metrics_rows)
        _plot_presentation_density(
            target,
            df,
            raw_presentation_path,
            y_col=target.raw_col,
            label="raw RTOFS",
            metrics_rows=metrics_rows,
        )
        _plot_presentation_density(
            target,
            df,
            corrected_presentation_path,
            y_col=target.corrected_col,
            label=f"corrected ({target.corrected_label})",
            metrics_rows=metrics_rows,
        )
        _plot_region_density(target, df, region_path, metrics_rows)
        patch_df = _patch_summary(target, df)
        patch_df.to_csv(patch_csv, index=False)
        _plot_patch_support_map(target, patch_df, patch_map)
        _plot_top_patch_density(target, df, patch_df, patch_density, metrics_rows)
        _plot_named_box_map(target, df, named_box_map)
        _plot_named_box_density(target, df, named_box_density, metrics_rows)
        # Support heatmap counts the full collocation table (data support),
        # not the locked OOF subset the scatters evaluate.
        support_df = merged_features[pd.notna(merged_features[target.obs_col])].copy()
        support_df["date"] = _canonical_date_str(support_df["date"])
        support_df = _augment_regions_and_patches(support_df)
        _plot_named_box_monthly_support(target, support_df, named_box_support, named_box_support_csv)
        error_summary = _plot_error_distributions(target, df, error_path)
        _plot_feature_relations(target, df, feature_path, feature_csv)

        manifest["targets"][target.name] = {
            "rows": int(len(df)),
            "global_density": str(global_path),
            "presentation_raw_density": str(raw_presentation_path),
            "presentation_corrected_density": str(corrected_presentation_path),
            "region_density": str(region_path),
            "patch_support_map": str(patch_map),
            "top_patch_density": str(patch_density),
            "named_box_map": str(named_box_map),
            "named_box_density": str(named_box_density),
            "named_box_monthly_support": str(named_box_support),
            "named_box_monthly_support_csv": str(named_box_support_csv),
            "patch_metrics_csv": str(patch_csv),
            "error_pdf_cdf": str(error_path),
            "feature_binned_mae": str(feature_path),
            "feature_binned_mae_csv": str(feature_csv),
            "error_summary": error_summary,
        }

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = paths["tables"] / "density_scatter_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    manifest["metrics_csv"] = str(metrics_csv)

    manifest_path = paths["root"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
