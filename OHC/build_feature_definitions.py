"""Generate the feature reference: exact definition, computation and statistics.

Covers the 37 features across the two recommended recipes (34 for TCHP,
35 for D26). Definitions quote the code that computes them. Statistics are
computed on the population the model trains on: the rebuilt 2026-09-24
tables, rows where both Argo and RTOFS have 26 C water, one primary profile
per Argo cast.

Mode is only meaningful for discrete features; for continuous ones the
centre of the tallest of 100 histogram bins (between the 1st and 99th
percentiles) is reported instead and labelled as such.

Outputs: docs/feature_definitions.md, docs/feature_definitions.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import _prepare_features  # noqa: E402

OUT_MD = ROOT / "docs" / "feature_definitions.md"
OUT_CSV = ROOT / "docs" / "feature_definitions.csv"
IDW = ("sampled to the Argo position by inverse-distance-squared weighting over the 8 nearest "
       "RTOFS grid cells: F = sum(w_i F_i) / sum(w_i), w_i = 1/d_i^2, missing neighbours excluded")
STENCIL = ("from the daily RTOFS field at the nearest grid cell (y0, x0); square window "
           "|y - y0| <= h, |x - x0| <= h cells on the native 1/12-degree grid (longitude wraps); "
           "valid (finite) cells only, and NaN unless valid cells cover >= 25% of the window")

# feature: (group, units, definition, how it is computed, where in code)
DEFS = {
    "year": ("Calendar", "year", "Calendar year of the Argo profile's observation date.",
             "Taken from the profile date.", "build_rtofs_at_argo_points_multiyear.py"),
    "month_int": ("Calendar", "1-12", "Month m of the observation date.",
                  "`x['month'].astype(int)`.", "benchmark_rtofs_argo_tabular_models.py::_prepare_features"),
    "month_sin": ("Calendar", "-", "sin(2 pi m / 12).", "Cyclic encoding so December and January are adjacent.",
                  "_prepare_features"),
    "month_cos": ("Calendar", "-", "cos(2 pi m / 12).", "Cyclic encoding, paired with month_sin.", "_prepare_features"),
    "doy_sin": ("Calendar", "-", "sin(2 pi d / 366), d = day of year (1-366).",
                "Divides by 366 in every year, including non-leap years.", "_prepare_features"),
    "doy_cos": ("Calendar", "-", "cos(2 pi d / 366).", "Paired with doy_sin.", "_prepare_features"),
    "is_winter_jfm": ("Calendar", "0/1", "1 if m in {1, 2, 3}, else 0.",
                      "From `_season_from_month`: JFM -> winter_jfm.", "_prepare_features"),
    "is_summer_jas": ("Calendar", "0/1", "1 if m in {7, 8, 9}, else 0.",
                      "JAS -> summer_jas.", "_prepare_features"),
    "is_other": ("Calendar", "0/1", "1 if m in {4, 5, 6, 10, 11, 12}, else 0.",
                 "All remaining months.", "_prepare_features"),
    "lat": ("Location", "deg N", "Latitude of the Argo profile as reported by the float.",
            "From the Argo profile file.", "build_global_argo_2020_2024.py"),
    "lon": ("Location", "deg E", "Longitude of the Argo profile as reported by the float.",
            "From the Argo profile file.", "build_global_argo_2020_2024.py"),
    "abs_lat": ("Location", "deg", "|lat|, distance from the equator in degrees.",
                "`np.abs(x['lat'])`.", "_prepare_features"),
    "nearest_rtofs_grid_distance_km": (
        "Location / collocation", "km",
        "Great-circle distance from the Argo position to the nearest RTOFS native grid-cell centre.",
        "KD-tree on unit-sphere coordinates for the 8 nearest cells; chord c converted to "
        "arc 6371 * 2 asin(c/2); the smallest of the 8 is kept. Bounded by half the cell diagonal.",
        "build_rtofs_at_argo_points_multiyear.py:140"),
    "model_interp_tchp_kj_per_cm2": (
        "Raw model state", "kJ/cm^2",
        "RTOFS Tropical Cyclone Heat Potential at the profile position: heat above the 26 C isotherm.",
        "Daily field: TCHP = 1e-7 * sum over HYCOM layers above D26 of rho * cp * max(T - 26, 0) * dz_eff, "
        "with rho = gsw.rho_t_exact(SA, T, p) and cp = gsw.cp_t_exact(SA, T, p) (TEOS-10), "
        "layer thickness = HYCOM thickness (Pa) / 9806, dz_eff the full layer if it lies above D26 "
        "or the part above D26 if it straddles it. Source: rtofs_glo.t00z.f06.archv (00Z cycle, 6-h forecast). "
        "Then " + IDW + ". NaN where the column never reaches 26 C.",
        "build_rtofs_global_daily_tchp_fields.py (~L103-160); collocation in build_rtofs_at_argo_points_multiyear.py"),
    "model_interp_d26_m": (
        "Raw model state", "m", "RTOFS depth of the 26 C isotherm at the profile position.",
        "Daily field: first HYCOM layer (from the surface) with T <= 26 C, linearly interpolated between "
        "that layer's centre depth and the one above: D26 = z1 + (T1 - 26)/(T1 - T2) * (z2 - z1). "
        "Requires the top layer to be >= 26 C. Then " + IDW + ".",
        "build_rtofs_global_daily_tchp_fields.py (~L118-134)"),
    "model_ssh_m": ("Global physics", "m", "RTOFS sea-surface height.",
                    "Variable `ssh` in rtofs_glo_2ds_f006_diag.nc (00Z, 6-h forecast); " + IDW + ".",
                    "build_rtofs_global_physics_features_2024_2025.py"),
    "model_mixed_layer_thickness_m": ("Global physics", "m", "RTOFS mixed-layer thickness (HYCOM's own diagnostic).",
                                      "Variable `mixed_layer_thickness` in the 2D diagnostic file; " + IDW + ".",
                                      "build_rtofs_global_physics_features_2024_2025.py"),
    "model_surface_boundary_layer_thickness_m": (
        "Global physics", "m", "RTOFS surface boundary-layer thickness (HYCOM's own diagnostic).",
        "Variable `surface_boundary_layer_thickness` in the 2D diagnostic file; " + IDW + ".",
        "build_rtofs_global_physics_features_2024_2025.py"),
    "model_temp_excess_26c": ("Global physics", "deg C", "T_s - 26, where T_s is RTOFS surface temperature.",
                              "T_s = top-layer temperature from the daily TCHP field (`surface_temp_c`), " + IDW +
                              "; then `surf_t - 26.0`. Negative means the surface is below 26 C.",
                              "build_rtofs_global_physics_features_2024_2025.py:172"),
    "d26_minus_mlt_m": ("Global physics (derived)", "m", "model_interp_d26_m - model_mixed_layer_thickness_m.",
                        "Thickness of the warm layer below the mixed layer. Negative if D26 lies inside the mixed layer.",
                        "build_rtofs_global_physics_features_2024_2025.py"),
    "d26_to_sblt_ratio": ("Global physics (derived)", "-", "model_interp_d26_m / model_surface_boundary_layer_thickness_m.",
                          "How many boundary-layer depths the 26 C isotherm sits below the surface.",
                          "build_rtofs_global_physics_features_2024_2025.py:186"),
    "model_ssh_x_abs_lat": ("Global physics (interaction)", "m * deg", "model_ssh_m * |lat|.",
                            "Lets a single split act differently by latitude.", "build_rtofs_global_physics_features_2024_2025.py:193"),
    "model_mlt_x_abs_lat": ("Global physics (interaction)", "m * deg", "model_mixed_layer_thickness_m * |lat|.",
                            "As above.", "build_rtofs_global_physics_features_2024_2025.py:194"),
    "model_temp_excess_x_abs_lat": ("Global physics (interaction)", "deg C * deg", "model_temp_excess_26c * |lat|.",
                                    "As above.", "build_rtofs_global_physics_features_2024_2025.py:195"),
    "model_steric_1000_ref2000_m": (
        "Deep profile (D26 recipe only)", "m",
        "Steric height of the 1000-2000 dbar layer of the RTOFS water column.",
        "[dyn(0; p_ref=2000) - dyn(0; p_ref=1000)] / g with dyn = gsw.geo_strf_dyn_height at the surface, "
        "g = 9.81, from the RTOFS T/S profile interpolated to the Argo point (NaN-safe inverse-distance "
        "weighting). Requires the profile to reach 2000 dbar. Available on ~11% of rows (77 processed dates). "
        "The ~2% of negative values are all in the Mediterranean (33-44 N, 2-35 E): its deep water is so salty "
        "that at 1000-2000 dbar it is denser than the reference water despite being warm, so the specific-volume "
        "anomaly and hence the steric height are negative. Physical, not an error.",
        "build_rtofs_profile_physics_features_2024_2025.py (~L212-230)"),
    "model_n2_max_upper200_s2": (
        "Deep profile (D26 recipe only)", "s^-2", "Maximum buoyancy frequency squared in the upper 200 m of RTOFS.",
        "max of gsw.Nsquared(SA, CT, p, lat) over layer mid-points shallower than 200 m. ~11% coverage.",
        "build_rtofs_profile_physics_features_2024_2025.py:243"),
    "model_n2_mean_to_d26_s2": (
        "Deep profile (D26 recipe only)", "s^-2", "Mean buoyancy frequency squared between the surface and RTOFS D26.",
        "mean of gsw.Nsquared over mid-points shallower than model D26. ~11% coverage.",
        "build_rtofs_profile_physics_features_2024_2025.py:247"),
}
for f, lab in (("tchp", "TCHP"), ("d26", "D26"), ("sst", "surface temperature")):
    unit = {"tchp": "kJ/cm^2", "d26": "m", "sst": "deg C"}[f]
    DEFS[f"model_{f}_local_std_1deg"] = (
        "Neighbourhood stencil", unit, f"Standard deviation of RTOFS {lab} within about 1 degree of the profile.",
        f"Population std (ddof=0) of valid cells, h = 12 cells (~2 x 2 degree box); " + STENCIL + ".",
        "build_rtofs_neighborhood_features_2024_2025.py")
    DEFS[f"model_{f}_grad_mag_per_100km"] = (
        "Neighbourhood stencil", f"{unit} per 100 km", f"Magnitude of the horizontal gradient of RTOFS {lab}.",
        "100 * sqrt(((F[y0,x0+1] - F[y0,x0-1]) / (2 dx))^2 + ((F[y0+1,x0] - F[y0-1,x0]) / (2 dy))^2), "
        "centred differences at the nearest grid cell, dx and dy the local grid spacings in km "
        "(dx shrinks with cos lat). NaN if any of the four neighbours is missing.",
        "build_rtofs_neighborhood_features_2024_2025.py::_grad_mag_per_100km")
    DEFS[f"model_{f}_anom_from_1deg_mean"] = (
        "Neighbourhood stencil", unit, f"How much RTOFS {lab} at the profile's grid cell departs from its own 1-degree neighbourhood mean.",
        f"F[y0, x0] - mean of valid cells in the h = 12 window; " + STENCIL +
        ". Uses the nearest-cell value, not the interpolated one. Biased near the 26 C edge because "
        "sub-threshold cells are excluded rather than counted as zero.",
        "build_rtofs_neighborhood_features_2024_2025.py:165")
DEFS["model_tchp_local_std_2deg"] = (
    "Neighbourhood stencil", "kJ/cm^2", "Standard deviation of RTOFS TCHP within about 2 degrees of the profile.",
    "As the 1-degree version with h = 25 cells (~4 x 4 degree box); " + STENCIL + ".",
    "build_rtofs_neighborhood_features_2024_2025.py")

DISCRETE = {"year", "month_int", "is_winter_jfm", "is_summer_jas", "is_other", "month_sin", "month_cos"}


def fmt(v: float) -> str:
    if not np.isfinite(v):
        return "-"
    if abs(v) < 1e-12:
        return "0"
    a = abs(v)
    if a != 0 and (a < 1e-3 or a >= 1e5):
        return f"{v:.3g}"
    return f"{v:.4g}"


def main() -> None:
    tchp = abl.FEATURE_SETS_BY_TARGET["tchp"]["global_pruned_plus_neighborhood"]
    d26 = abl.FEATURE_SETS_BY_TARGET["d26"]["drop_both_lat_interactions_plus_neighborhood"]
    feats = list(dict.fromkeys(list(tchp) + list(d26)))
    missing = [f for f in feats if f not in DEFS]
    if missing:
        raise RuntimeError(f"no definition written for {missing}")

    df = _prepare_features(abl._merge_feature_tables())
    allrows = len(df)
    pop = df[df["argo_tchp_kj_per_cm2"].notna() & df["model_interp_tchp_kj_per_cm2"].notna()
             & df["is_primary_profile"].astype(bool)]
    rows = []
    for f in feats:
        g, u, d, how, where = DEFS[f]
        s = pd.to_numeric(pop[f], errors="coerce")
        v = s.dropna().to_numpy(float)
        if f in DISCRETE:
            vc = s.round(6).value_counts()
            mode, mode_note = float(vc.index[0]), f"{100 * vc.iloc[0] / len(v):.1f}% of rows"
        else:
            lo, hi = np.percentile(v, [1, 99])
            h, e = np.histogram(v[(v >= lo) & (v <= hi)], bins=100)
            k = int(np.argmax(h))
            mode, mode_note = float(0.5 * (e[k] + e[k + 1])), "modal bin (continuous)"
        rows.append({
            "feature": f, "group": g, "units": u,
            "in_tchp_recipe": f in tchp, "in_d26_recipe": f in d26,
            "definition": d, "computation": how, "code": where,
            "n_valid": int(len(v)), "pct_missing_imputed": 100 * (1 - len(v) / len(pop)),
            "coverage_all_rows_pct": 100 * pd.to_numeric(df[f], errors="coerce").notna().mean(),
            "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()),
            "p01": float(np.percentile(v, 1)), "p25": float(np.percentile(v, 25)),
            "median": float(np.median(v)), "p75": float(np.percentile(v, 75)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max()),
            "mode": mode, "mode_note": mode_note,
        })
    T = pd.DataFrame(rows)
    T.to_csv(OUT_CSV, index=False)

    L = ["# Feature reference: exact definitions, computation and statistics", "",
         f"Generated by `OHC/build_feature_definitions.py`. Covers the **{len(feats)}** features across the two "
         f"recommended recipes: **{len(tchp)}** for TCHP (`global_pruned_plus_neighborhood`) and **{len(d26)}** for D26 "
         "(`drop_both_lat_interactions_plus_neighborhood`). D26 drops the SSH x |lat| and temperature-excess x |lat| "
         "interactions and adds three deep-profile features.", "",
         "## How the statistics were computed", "",
         f"* **Population:** the rows the model trains and is scored on, from the rebuilt tables of 2026-09-24: both Argo and "
         f"RTOFS have 26 C water, one primary profile per Argo cast. **{len(pop):,}** rows (of {allrows:,} collocated).",
         "* **Missing values** are replaced by the training-fold median before fitting. `% imputed` is the share of this "
         "population where the feature is missing; `coverage (all rows)` is availability across all collocated rows, "
         "including cold water.",
         "* **Mode:** exact for discrete features. For continuous features the centre of the tallest of 100 histogram bins "
         "between the 1st and 99th percentiles is reported instead, since the literal mode of a continuous variable is "
         "an arbitrary repeated value.",
         "* **Range** is given twice: the 1st-99th percentile (the robust range) and the absolute min-max.", "",
         "## The targets (for context: these are not inputs)", "",
         "* **Argo TCHP and D26** (`hhp_core.compute_tchp_teos`, pressure axis): D26 is the first depth where temperature "
         "falls to 26 C, linearly interpolated between profile levels (depth from `gsw.z_from_p`). TCHP is the trapezoidal "
         "integral from the surface to D26 of rho * cp * max(T - 26, 0), with rho and cp from TEOS-10 at each level, "
         "converted to kJ/cm^2 (x 1e-7).",
         "* **What the model predicts** is the error delta = Argo - RTOFS for each quantity. The corrected forecast is "
         "RTOFS + predicted delta.", "",
         "## Summary table", "",
         "| feature | group | units | recipes | mean | median | 1st-99th pct | min - max | mode | % imputed |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rec = "/".join(x for x, ok in (("TCHP", r["in_tchp_recipe"]), ("D26", r["in_d26_recipe"])) if ok)
        md = fmt(r["mode"]) + ("" if r["mode_note"].startswith("modal") else f" ({r['mode_note']})")
        if r["mode_note"].startswith("modal"):
            md += " *"
        L.append(f"| `{r['feature']}` | {r['group']} | {r['units']} | {rec} | {fmt(r['mean'])} | {fmt(r['median'])} | "
                 f"{fmt(r['p01'])} to {fmt(r['p99'])} | {fmt(r['min'])} to {fmt(r['max'])} | {md} | "
                 f"{r['pct_missing_imputed']:.1f}% |")
    L += ["", "\\* modal histogram bin, for continuous features.", "", "## Definitions and computation, by group", ""]
    for grp in dict.fromkeys(r["group"] for r in rows):
        L += [f"### {grp}", ""]
        for r in rows:
            if r["group"] != grp:
                continue
            L += [f"#### `{r['feature']}`  ({r['units']})", "",
                  f"* **Definition:** {r['definition']}",
                  f"* **Computed:** {r['computation']}",
                  f"* **Code:** `{r['code']}`",
                  f"* **Statistics:** mean {fmt(r['mean'])}, std {fmt(r['std'])}, median {fmt(r['median'])}, "
                  f"quartiles {fmt(r['p25'])} to {fmt(r['p75'])}, 1st-99th pct {fmt(r['p01'])} to {fmt(r['p99'])}, "
                  f"min {fmt(r['min'])}, max {fmt(r['max'])}; mode {fmt(r['mode'])} ({r['mode_note']}); "
                  f"{r['n_valid']:,} valid, {r['pct_missing_imputed']:.1f}% imputed, "
                  f"{r['coverage_all_rows_pct']:.1f}% coverage across all rows.", ""]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(T[["feature", "mean", "median", "p01", "p99", "mode", "pct_missing_imputed"]].to_string(index=False))
    print("wrote", OUT_MD, "and", OUT_CSV)


if __name__ == "__main__":
    main()
