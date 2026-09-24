"""Generate the feature reference: exact definition, origin, computation and statistics.

Covers every feature in the two recommended recipes (34 for TCHP, 35 for D26)
and in the frozen recipes for the 2026 test (34 and 32). Definitions quote the
code that computes them, and each feature says where it comes from: used as
is from RTOFS or Argo, computed by us from RTOFS (with or without TEOS-10), or
derived by us. Upstream definitions are linked, and code references are
GitHub permalinks pinned to the current commit, with line numbers found by
searching the code so they cannot drift.

Statistics are computed on the population the model trains on: the rebuilt
2026-09-24 tables, rows where both Argo and RTOFS have 26 C water, one primary
profile per Argo cast. Mode is only meaningful for discrete features; for
continuous ones the centre of the tallest of 100 histogram bins (between the
1st and 99th percentiles) is reported instead and labelled as such.

Outputs: docs/feature_definitions.md, docs/feature_definitions.csv
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import OHC.run_locked_xgb_physics_semi_ablation as abl  # noqa: E402
from OHC.benchmark_rtofs_argo_tabular_models import _prepare_features  # noqa: E402
from OHC.exploration.run_frozen_recipe_dev import frozen_recipe  # noqa: E402

OUT_MD = ROOT / "docs" / "feature_definitions.md"
OUT_CSV = ROOT / "docs" / "feature_definitions.csv"
PT_CHECK = ROOT / "OHC" / "output" / "potential_temperature_check_20260924" / "result.json"
REPO = "https://github.com/AQUAVIEW-DAH/HHP-ML-Recomputations"
SHA = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

IDW = ("sampled to the Argo position by inverse-distance-squared weighting over the 8 nearest "
       "RTOFS grid cells: F = sum(w_i F_i) / sum(w_i), w_i = 1/d_i^2, missing neighbours excluded")
STENCIL = ("from the daily RTOFS field at the nearest grid cell (y0, x0); square window "
           "|y - y0| <= h, |x - x0| <= h cells on the native 1/12-degree grid (longitude wraps); "
           "valid (finite) cells only, and NaN unless valid cells cover >= 25% of the window")

# Upstream definitions (all checked to resolve on 2026-09-24, except that the Argo manual's DOI
# target, archimer.ifremer.fr, did not answer from this server; the Argo documentation page cites it).
RTOFS_PRODUCTS = ("NCEP RTOFS product list", "https://www.nco.ncep.noaa.gov/pmb/products/rtofs/")
RTOFS_DOCS = ("RTOFS-Global documentation", "https://github.com/NOAA-EMC/RTOFS_GLO/wiki")
RTOFS_DATA = ("RTOFS data on AWS (the bucket we download from)", "https://registry.opendata.aws/noaa-rtofs/")
HYCOM = ("HYCOM documentation", "https://www.hycom.org/hycom/documentation")
CF = ("CF standard-name table", "https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html")
ARGO = ("Argo user's manual", "https://doi.org/10.13155/29825")
ARGO_DOCS = ("Argo data documentation", "https://www.argodatamgt.org/Documentation")
TCHP_PAPER = ("Leipper & Volgenau 1972, the TCHP definition",
              "https://doi.org/10.1175/1520-0485%281972%29002%3C0218%3AHHPOTG%3E2.0.CO%3B2")
TEOS_MANUAL = ("TEOS-10 manual", "https://www.teos-10.org/pubs/TEOS-10_Manual.pdf")
GSW_PY = ("GSW-Python, the package we call", "https://teos-10.github.io/GSW-Python/")


def gsw_ref(name: str) -> tuple[str, str]:
    return f"TEOS-10 gsw_{name}", f"https://www.teos-10.org/pubs/gsw/html/gsw_{name}.html"


RTOFS = [RTOFS_PRODUCTS, RTOFS_DOCS, RTOFS_DATA]
ARGOS = [ARGO, ARGO_DOCS]
TEOS_TCHP = [gsw_ref(f) for f in ("p_from_z", "SA_from_SP", "rho_t_exact", "cp_t_exact")] + [TEOS_MANUAL, GSW_PY]
TEOS_PROFILE = [gsw_ref(f) for f in ("SA_from_SP", "CT_from_t")]

# Origin categories, in the order they are listed.
AS_IS_RTOFS = "RTOFS output, used as is"
AS_IS_ARGO = "Argo data, used as is"
TEOS = "computed by us from RTOFS, with TEOS-10"
COMPUTED = "computed by us from RTOFS, no TEOS-10"
FROM_RTOFS = "derived by us from RTOFS values"
FROM_ARGO = "derived by us from the Argo date or position"
GEOMETRY = "collocation geometry (Argo position + RTOFS grid)"
CATEGORIES = [AS_IS_RTOFS, AS_IS_ARGO, TEOS, COMPUTED, FROM_RTOFS, FROM_ARGO, GEOMETRY]

PT_NOTE = ("RTOFS temperature is potential temperature but is used as in-situ temperature here; "
           "see [Known approximation](#known-approximation-potential-vs-in-situ-temperature).")

BENCH = "OHC/benchmark_rtofs_argo_tabular_models.py"
COLLOC = "OHC/build_rtofs_at_argo_points_multiyear.py"
FIELDS = "OHC/build_rtofs_global_daily_tchp_fields.py"
PHYS = "OHC/build_rtofs_global_physics_features_2024_2025.py"
NBHD = "OHC/build_rtofs_neighborhood_features_2024_2025.py"
PROF = "OHC/build_rtofs_profile_physics_features_2024_2025.py"
ARGO_READER = "ml/sources/argo_gdac_source.py"

# feature: (group, units, definition, how it is computed)
DEFS = {
    "year": ("Calendar", "year", "Calendar year of the Argo profile's observation date.", "Taken from the profile date."),
    "month_int": ("Calendar", "1-12", "Month m of the observation date.", "`x['month'].astype(int)`."),
    "month_sin": ("Calendar", "-", "sin(2 pi m / 12).", "Cyclic encoding so December and January are adjacent."),
    "month_cos": ("Calendar", "-", "cos(2 pi m / 12).", "Cyclic encoding, paired with month_sin."),
    "doy_sin": ("Calendar", "-", "sin(2 pi d / 366), d = day of year (1-366).",
                "Divides by 366 in every year, including non-leap years."),
    "doy_cos": ("Calendar", "-", "cos(2 pi d / 366).", "Paired with doy_sin."),
    "is_winter_jfm": ("Calendar", "0/1", "1 if m in {1, 2, 3}, else 0.", "`_season_from_month`: JFM -> winter_jfm."),
    "is_summer_jas": ("Calendar", "0/1", "1 if m in {7, 8, 9}, else 0.", "JAS -> summer_jas."),
    "is_other": ("Calendar", "0/1", "1 if m in {4, 5, 6, 10, 11, 12}, else 0.", "All remaining months."),
    "lat": ("Location", "deg N", "Latitude of the Argo profile as reported by the float.", "From the Argo profile file."),
    "lon": ("Location", "deg E", "Longitude of the Argo profile as reported by the float.", "From the Argo profile file."),
    "abs_lat": ("Location", "deg", "|lat|, distance from the equator in degrees.", "`np.abs(x['lat'])`."),
    "lon_sin": ("Location", "-", "sin(lon), longitude in radians.",
                "Cyclic longitude: 180 W and 180 E become the same point, which removes the seam at the dateline. "
                "Frozen recipe only."),
    "lon_cos": ("Location", "-", "cos(lon), longitude in radians.", "Paired with lon_sin. Frozen recipe only."),
    "nearest_rtofs_grid_distance_km": (
        "Location / collocation", "km",
        "Great-circle distance from the Argo position to the nearest RTOFS native grid-cell centre.",
        "KD-tree on unit-sphere coordinates for the 8 nearest cells; chord c converted to "
        "arc 6371 * 2 asin(c/2); the smallest of the 8 is kept. Bounded by half the cell diagonal."),
    "model_interp_tchp_kj_per_cm2": (
        "Raw model state", "kJ/cm^2",
        "RTOFS Tropical Cyclone Heat Potential at the profile position: heat above the 26 C isotherm.",
        "Daily field: TCHP = 1e-7 * sum over HYCOM layers above D26 of rho * cp * max(T - 26, 0) * dz_eff, "
        "with p = gsw.p_from_z(-z, lat), SA = gsw.SA_from_SP(S, p, lon, lat), rho = gsw.rho_t_exact(SA, T, p) and "
        "cp = gsw.cp_t_exact(SA, T, p) (TEOS-10), layer thickness = HYCOM thickness (Pa) / 9806, dz_eff the full "
        "layer if it lies above D26 or the part above D26 if it straddles it. Source: rtofs_glo.t00z.f06.archv "
        "(00Z cycle, 6-h forecast). Then " + IDW + ". NaN where the column never reaches 26 C."),
    "model_interp_d26_m": (
        "Raw model state", "m", "RTOFS depth of the 26 C isotherm at the profile position.",
        "Daily field: first HYCOM layer (from the surface) with T <= 26 C, linearly interpolated between "
        "that layer's centre depth and the one above: D26 = z1 + (T1 - 26)/(T1 - T2) * (z2 - z1). "
        "Requires the top layer to be >= 26 C. Then " + IDW + "."),
    "model_ssh_m": ("Global physics", "m", "RTOFS sea-surface height.",
                    "Variable `ssh` in rtofs_glo_2ds_f006_diag.nc (00Z, 6-h forecast); " + IDW + "."),
    "model_mixed_layer_thickness_m": ("Global physics", "m", "RTOFS mixed-layer thickness (HYCOM's own diagnostic).",
                                      "Variable `mixed_layer_thickness` in the 2D diagnostic file; " + IDW + "."),
    "model_surface_boundary_layer_thickness_m": (
        "Global physics", "m", "RTOFS surface boundary-layer thickness (HYCOM's own diagnostic).",
        "Variable `surface_boundary_layer_thickness` in the 2D diagnostic file; " + IDW + "."),
    "model_temp_excess_26c": ("Global physics", "deg C", "T_s - 26, where T_s is RTOFS surface temperature.",
                              "T_s = top-layer temperature from the daily TCHP field (`surface_temp_c`), " + IDW +
                              "; then `surf_t - 26.0`. Negative means the surface is below 26 C."),
    "d26_minus_mlt_m": ("Global physics (derived)", "m", "model_interp_d26_m - model_mixed_layer_thickness_m.",
                        "Thickness of the warm layer below the mixed layer. Negative if D26 lies inside the mixed layer."),
    "d26_to_sblt_ratio": ("Global physics (derived)", "-", "model_interp_d26_m / model_surface_boundary_layer_thickness_m.",
                          "How many boundary-layer depths the 26 C isotherm sits below the surface."),
    "model_ssh_x_abs_lat": ("Global physics (interaction)", "m * deg", "model_ssh_m * |lat|.",
                            "Lets a single split act differently by latitude."),
    "model_mlt_x_abs_lat": ("Global physics (interaction)", "m * deg", "model_mixed_layer_thickness_m * |lat|.", "As above."),
    "model_temp_excess_x_abs_lat": ("Global physics (interaction)", "deg C * deg", "model_temp_excess_26c * |lat|.",
                                    "As above."),
    "model_steric_1000_ref2000_m": (
        "Deep profile (D26 recipe only)", "m",
        "Steric height of the 1000-2000 dbar layer of the RTOFS water column.",
        "[dyn(0; p_ref=2000) - dyn(0; p_ref=1000)] / g with dyn = gsw.geo_strf_dyn_height at the surface, "
        "g = 9.81, from the RTOFS T/S profile interpolated to the Argo point (NaN-safe inverse-distance "
        "weighting). Requires the profile to reach 2000 dbar. Available on ~11% of rows (77 processed dates). "
        "The ~2% of negative values are all in the Mediterranean (33-44 N, 2-35 E): its deep water is so salty "
        "that at 1000-2000 dbar it is denser than the reference water despite being warm, so the specific-volume "
        "anomaly and hence the steric height are negative. Physical, not an error."),
    "model_n2_max_upper200_s2": (
        "Deep profile (D26 recipe only)", "s^-2", "Maximum buoyancy frequency squared in the upper 200 m of RTOFS.",
        "max of gsw.Nsquared(SA, CT, p, lat) over layer mid-points shallower than 200 m. ~11% coverage."),
    "model_n2_mean_to_d26_s2": (
        "Deep profile (D26 recipe only)", "s^-2", "Mean buoyancy frequency squared between the surface and RTOFS D26.",
        "mean of gsw.Nsquared over mid-points shallower than model D26. ~11% coverage."),
}

# feature: (category, what exactly is taken from upstream, upstream links)
ORIGIN = {
    **{f: (FROM_ARGO, "From the Argo observation time `JULD`.", ARGOS)
       for f in ("year", "month_int", "month_sin", "month_cos", "doy_sin", "doy_cos",
                 "is_winter_jfm", "is_summer_jas", "is_other")},
    "lat": (AS_IS_ARGO, "Argo variable `LATITUDE` of the profile (GDAC profile file).", ARGOS),
    "lon": (AS_IS_ARGO, "Argo variable `LONGITUDE` of the profile (GDAC profile file).", ARGOS),
    "abs_lat": (FROM_ARGO, "From Argo `LATITUDE`.", ARGOS),
    "lon_sin": (FROM_ARGO, "From Argo `LONGITUDE`.", ARGOS),
    "lon_cos": (FROM_ARGO, "From Argo `LONGITUDE`.", ARGOS),
    "nearest_rtofs_grid_distance_km": (
        GEOMETRY, "Argo `LATITUDE`/`LONGITUDE` and the RTOFS grid-cell coordinates (`Latitude`, `Longitude` in "
        "rtofs_glo_2ds_f006_diag.nc).", ARGOS + RTOFS),
    "model_interp_tchp_kj_per_cm2": (
        TEOS, "RTOFS does not publish TCHP. We compute it from the RTOFS 3D fields `temp`, `salin` and `thknss` "
        "in rtofs_glo.t00z.f06.archv.a/.b; the formula is the Leipper & Volgenau (1972) definition, with density "
        "and heat capacity from TEOS-10. " + PT_NOTE, RTOFS + [HYCOM, TCHP_PAPER] + TEOS_TCHP),
    "model_interp_d26_m": (
        COMPUTED, "RTOFS does not publish D26. We compute it from the RTOFS 3D fields `temp` and `thknss`; no "
        "TEOS-10 is involved (depth = layer pressure thickness / 9806 Pa per m). " + PT_NOTE, RTOFS + [HYCOM]),
    "model_ssh_m": (
        AS_IS_RTOFS, "Variable `ssh` in rtofs_glo_2ds_f006_diag.nc; CF standard name `sea_surface_elevation`; "
        "HYCOM label `sea surf. height`.", RTOFS + [CF]),
    "model_mixed_layer_thickness_m": (
        AS_IS_RTOFS, "Variable `mixed_layer_thickness` in rtofs_glo_2ds_f006_diag.nc; CF standard name "
        "`ocean_mixed_layer_thickness`; HYCOM label `mix.layr.thickness`.", RTOFS + [HYCOM, CF]),
    "model_surface_boundary_layer_thickness_m": (
        AS_IS_RTOFS, "Variable `surface_boundary_layer_thickness` in rtofs_glo_2ds_f006_diag.nc; HYCOM label "
        "`bnd.layr.thickness` (the file gives no CF standard name).", RTOFS + [HYCOM]),
    "model_temp_excess_26c": (
        FROM_RTOFS, "RTOFS top-layer temperature (archive field `temp`, layer 1), used as is, minus 26 C.", RTOFS),
    "d26_minus_mlt_m": (FROM_RTOFS, "Our D26 minus RTOFS `mixed_layer_thickness`.", RTOFS),
    "d26_to_sblt_ratio": (FROM_RTOFS, "Our D26 divided by RTOFS `surface_boundary_layer_thickness`.", RTOFS),
    "model_ssh_x_abs_lat": (FROM_RTOFS, "RTOFS `ssh` times |Argo latitude|.", RTOFS),
    "model_mlt_x_abs_lat": (FROM_RTOFS, "RTOFS `mixed_layer_thickness` times |Argo latitude|.", RTOFS),
    "model_temp_excess_x_abs_lat": (FROM_RTOFS, "RTOFS top-layer temperature minus 26, times |Argo latitude|.", RTOFS),
    "model_steric_1000_ref2000_m": (
        TEOS, "RTOFS 3D temperature and salinity profile; TEOS-10 for absolute salinity, Conservative Temperature "
        "and dynamic height. " + PT_NOTE, RTOFS + TEOS_PROFILE + [gsw_ref("geo_strf_dyn_height"), GSW_PY]),
    "model_n2_max_upper200_s2": (
        TEOS, "RTOFS 3D temperature and salinity profile; TEOS-10 for absolute salinity, Conservative Temperature "
        "and N^2. " + PT_NOTE, RTOFS + TEOS_PROFILE + [gsw_ref("Nsquared"), GSW_PY]),
    "model_n2_mean_to_d26_s2": (
        TEOS, "As model_n2_max_upper200_s2. " + PT_NOTE, RTOFS + TEOS_PROFILE + [gsw_ref("Nsquared"), GSW_PY]),
}

# feature: [(file, text on the line to link)]
CODE = {
    "year": [("OHC/build_global_argo_2020_2024.py", '"year": int(yyyymmdd[:4])')],
    **{f: [(BENCH, f'x["{f}"] =')] for f in ("month_int", "month_sin", "month_cos", "doy_sin", "doy_cos", "abs_lat")},
    **{f: [(COLLOC, "def _season_from_month"), (BENCH, f'x["{f}"] =')]
       for f in ("is_winter_jfm", "is_summer_jas", "is_other")},
    "lat": [(ARGO_READER, 'lat=float(ds["LATITUDE"]')],
    "lon": [(ARGO_READER, 'lon=float(ds["LONGITUDE"]')],
    "lon_sin": [("OHC/exploration/run_frozen_recipe_dev.py", 'work["lon_sin"] =')],
    "lon_cos": [("OHC/exploration/run_frozen_recipe_dev.py", 'work["lon_cos"] =')],
    "nearest_rtofs_grid_distance_km": [(COLLOC, 'argo["nearest_rtofs_grid_distance_km"] =')],
    "model_interp_tchp_kj_per_cm2": [(FIELDS, "rho = gsw.rho_t_exact"), (FIELDS, "tchp = ohc / 1.0e7"),
                                     (COLLOC, "def _interpolate_neighbor_values")],
    "model_interp_d26_m": [(FIELDS, "d26[good] = z1[good]"), (COLLOC, "def _interpolate_neighbor_values")],
    "model_ssh_m": [(PHYS, 'ds["ssh"]')],
    "model_mixed_layer_thickness_m": [(PHYS, 'ds["mixed_layer_thickness"]')],
    "model_surface_boundary_layer_thickness_m": [(PHYS, 'ds["surface_boundary_layer_thickness"]')],
    "model_temp_excess_26c": [(FIELDS, "surf_t = np.where(valid_layer[0]"), (PHYS, "surf_t - REF_TEMP_C")],
    "d26_minus_mlt_m": [(PHYS, 'df["d26_minus_mlt_m"] =')],
    "d26_to_sblt_ratio": [(PHYS, 'df["d26_to_sblt_ratio"] =')],
    **{f: [(PHYS, f'df["{f}"] =')] for f in ("model_ssh_x_abs_lat", "model_mlt_x_abs_lat", "model_temp_excess_x_abs_lat")},
    "model_steric_1000_ref2000_m": [(PROF, "dyn_1000 = gsw.geo_strf_dyn_height"),
                                    (PROF, 'out["model_steric_1000_ref2000_m"] = float(')],
    "model_n2_max_upper200_s2": [(PROF, "n2, p_mid = gsw.Nsquared"), (PROF, 'out["model_n2_max_upper200_s2"] =')],
    "model_n2_mean_to_d26_s2": [(PROF, "n2, p_mid = gsw.Nsquared"), (PROF, 'out["model_n2_mean_to_d26_s2"] =')],
}

for f, lab in (("tchp", "TCHP"), ("d26", "D26"), ("sst", "surface temperature")):
    unit = {"tchp": "kJ/cm^2", "d26": "m", "sst": "deg C"}[f]
    field = ("the RTOFS top-layer temperature (archive field `temp`, layer 1), used as is" if f == "sst"
             else f"our daily RTOFS {lab} field (see model_interp_{'tchp_kj_per_cm2' if f == 'tchp' else 'd26_m'})")
    DEFS[f"model_{f}_local_std_1deg"] = (
        "Neighbourhood stencil", unit, f"Standard deviation of RTOFS {lab} within about 1 degree of the profile.",
        "Population std (ddof=0) of valid cells, h = 12 cells (~2 x 2 degree box); " + STENCIL + ".")
    DEFS[f"model_{f}_grad_mag_per_100km"] = (
        "Neighbourhood stencil", f"{unit} per 100 km", f"Magnitude of the horizontal gradient of RTOFS {lab}.",
        "100 * sqrt(((F[y0,x0+1] - F[y0,x0-1]) / (2 dx))^2 + ((F[y0+1,x0] - F[y0-1,x0]) / (2 dy))^2), "
        "centred differences at the nearest grid cell, dx and dy the local grid spacings in km "
        "(dx shrinks with cos lat). NaN if any of the four neighbours is missing.")
    DEFS[f"model_{f}_anom_from_1deg_mean"] = (
        "Neighbourhood stencil", unit,
        f"How much RTOFS {lab} at the profile's grid cell departs from its own 1-degree neighbourhood mean.",
        "F[y0, x0] - mean of valid cells in the h = 12 window; " + STENCIL +
        ". Uses the nearest-cell value, not the interpolated one. Biased near the 26 C edge because "
        "sub-threshold cells are excluded rather than counted as zero.")
    for kind, line in (("local_std_1deg", "vals.std()"), ("grad_mag_per_100km", "def _grad_mag_per_100km"),
                       ("anom_from_1deg_mean", "point_val - mean_1deg")):
        ORIGIN[f"model_{f}_{kind}"] = (FROM_RTOFS, f"Statistic of {field}.", RTOFS)
        CODE[f"model_{f}_{kind}"] = [(NBHD, "SCALES = {"), (NBHD, line)] if kind == "local_std_1deg" else [(NBHD, line)]
DEFS["model_tchp_local_std_2deg"] = (
    "Neighbourhood stencil", "kJ/cm^2", "Standard deviation of RTOFS TCHP within about 2 degrees of the profile.",
    "As the 1-degree version with h = 25 cells (~4 x 4 degree box); " + STENCIL + ".")
ORIGIN["model_tchp_local_std_2deg"] = ORIGIN["model_tchp_local_std_1deg"]
CODE["model_tchp_local_std_2deg"] = CODE["model_tchp_local_std_1deg"]

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


def link(label: str, url: str) -> str:
    return f"[{label}]({url})"


def code_link(path: str, text: str) -> str:
    """Permalink to the first line of `path` containing `text`, pinned to the current commit."""
    src = (ROOT / path).read_text().splitlines()
    hits = [i + 1 for i, line in enumerate(src) if text in line]
    if not hits:
        raise RuntimeError(f"{text!r} not found in {path}")
    committed = subprocess.run(["git", "show", f"{SHA}:{path}"], cwd=ROOT, capture_output=True, text=True)
    if committed.returncode != 0 or committed.stdout.splitlines() != src:
        raise RuntimeError(f"{path} differs from commit {SHA[:7]}; commit it first so the links show this code")
    return link(f"{Path(path).name}:{hits[0]}", f"{REPO}/blob/{SHA}/{path}#L{hits[0]}")


def main() -> None:
    tchp = abl.FEATURE_SETS_BY_TARGET["tchp"]["global_pruned_plus_neighborhood"]
    d26 = abl.FEATURE_SETS_BY_TARGET["d26"]["drop_both_lat_interactions_plus_neighborhood"]
    fz_tchp, fz_d26 = frozen_recipe("tchp"), frozen_recipe("d26")
    recipes = {"TCHP": tchp, "D26": d26, "frozen TCHP": fz_tchp, "frozen D26": fz_d26}
    feats = list(dict.fromkeys(list(tchp) + list(d26) + fz_tchp + fz_d26))
    missing = [f for f in feats if not (f in DEFS and f in ORIGIN and f in CODE)]
    if missing:
        raise RuntimeError(f"no definition, origin or code reference for {missing}")

    df = _prepare_features(abl._merge_feature_tables())
    df["lon_sin"] = np.sin(np.deg2rad(df["lon"]))
    df["lon_cos"] = np.cos(np.deg2rad(df["lon"]))
    allrows = len(df)
    pop = df[df["argo_tchp_kj_per_cm2"].notna() & df["model_interp_tchp_kj_per_cm2"].notna()
             & df["is_primary_profile"].astype(bool)]
    rows = []
    for f in feats:
        g, u, d, how = DEFS[f]
        cat, src, refs = ORIGIN[f]
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
            "feature": f, "group": g, "units": u, "origin": cat,
            "in_tchp_recipe": f in tchp, "in_d26_recipe": f in d26,
            "in_frozen_tchp": f in fz_tchp, "in_frozen_d26": f in fz_d26,
            "definition": d, "computation": how, "upstream": src,
            "upstream_links": " ; ".join(url for _, url in refs),
            "code": " ; ".join(f"{p}: {t}" for p, t in CODE[f]),
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

    L = ["# Feature reference: definitions, origin, computation and statistics", "",
         f"Generated by `OHC/build_feature_definitions.py`. Covers the **{len(feats)}** features used by any recipe: "
         f"the recommended recipes, **{len(tchp)}** for TCHP (`global_pruned_plus_neighborhood`) and **{len(d26)}** "
         "for D26 (`drop_both_lat_interactions_plus_neighborhood`), and the frozen recipes for the 2026 test, "
         f"**{len(fz_tchp)}** for TCHP and **{len(fz_d26)}** for D26.", "",
         "* **D26 vs TCHP:** the D26 recipe drops the SSH x |lat| and temperature-excess x |lat| interactions and "
         "adds the three deep-profile features (34 - 2 + 3 = 35).",
         "* **Frozen recipes:** remove `year` (no training fold can contain 2026), grid distance (no effect in any "
         "test) and, for D26, the three deep-profile features (cannot be built for 2026); add cyclic longitude "
         "`lon_sin`, `lon_cos`. See `OHC/exploration/run_frozen_recipe_dev.py`.",
         "* **The target is never an input.** Every feature listed here is an input. What the model predicts, the "
         "error delta = Argo - RTOFS, is held separately (see below).",
         f"* **Code links** are GitHub permalinks pinned to commit `{SHA[:7]}`.", "",
         "## Where the features come from", "",
         "| origin | TCHP | D26 | frozen TCHP | frozen D26 | features |", "|---|---|---|---|---|---|"]
    for cat in CATEGORIES:
        members = [r["feature"] for r in rows if r["origin"] == cat]
        if not members:
            continue
        counts = " | ".join(str(sum(m in rec for m in members)) for rec in recipes.values())
        L.append(f"| {cat} | {counts} | " + ", ".join(f"`{m}`" for m in members) + " |")
    L.append("| **total** | " + " | ".join(f"**{len(rec)}**" for rec in recipes.values()) + " | |")
    L += ["",
          "Only three inputs are RTOFS output used as is (SSH, mixed-layer thickness, boundary-layer thickness), and "
          "only two are Argo data used as is (latitude, longitude). RTOFS publishes neither TCHP nor D26: we compute "
          "both from its 3D fields. TEOS-10 is used for TCHP (density and heat capacity) and for the three "
          "deep-profile features (steric height, N^2), not for D26. The RTOFS surface temperature is also used as "
          "is, but only inside derived features (temperature excess and the three SST neighbourhood statistics).", "",
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
          "* **Argo TCHP and D26** (" + code_link("hhp_core.py", "def compute_tchp_teos") + ", pressure axis): D26 is the "
          "first depth where temperature falls to 26 C, linearly interpolated between profile levels (depth from "
          "`gsw.z_from_p`). TCHP is the trapezoidal integral from the surface to D26 of rho * cp * max(T - 26, 0), with "
          "rho = `gsw.rho_t_exact` and cp = `gsw.cp_t_exact` at each level, converted to kJ/cm^2 (x 1e-7). Argo `TEMP` is "
          "in-situ temperature, as these functions expect. Sources: " +
          ", ".join(link(*r) for r in [ARGO, ARGO_DOCS, TCHP_PAPER, gsw_ref("z_from_p"), gsw_ref("SA_from_SP"),
                                       gsw_ref("rho_t_exact"), gsw_ref("cp_t_exact")]) + ".",
          "* **What the model predicts** is the error delta = Argo - RTOFS for each quantity. The corrected forecast is "
          "RTOFS + predicted delta.", ""]

    pt = json.loads(PT_CHECK.read_text()) if PT_CHECK.exists() else None
    L += ["## Known approximation: potential vs in-situ temperature", "",
          "HYCOM, and so RTOFS, stores **potential** temperature (the RTOFS 3D z-level files label it "
          "`sea_water_potential_temperature`). Our RTOFS TCHP/D26 builder and profile-physics builder pass it where "
          "TEOS-10 expects **in-situ** temperature (`gsw.rho_t_exact`, `gsw.cp_t_exact`, `gsw.CT_from_t`), and compare "
          "it with 26 C as if it were in-situ. The Argo side uses in-situ temperature correctly."]
    if pt:
        g, dd, dh, st = (pt["insitu_minus_potential_temperature_c"], pt["d26_as_built_minus_correct_m"],
                         pt["tchp_as_built_minus_correct_kj_cm2"], pt["steric_1000_ref2000_as_built_minus_correct_m"])
        L += ["", f"Measured on RTOFS columns (`OHC/exploration/check_potential_vs_insitu_temperature.py`, "
              f"US-East regional file of 2024-10-07, {pt['warm_columns']:,} warm columns, {pt['deep_columns']:,} reaching "
              "2000 dbar), comparing the pipeline's computation with the correct one:", "",
              f"* In-situ temperature exceeds potential temperature by {g['60m']:.3f} C at 60 m, {g['100m']:.3f} C at "
              f"100 m and {g['200m']:.3f} C at 200 m.",
              f"* D26 comes out {abs(dd['mean']):.2f} m too shallow on average (99% of columns within "
              f"{dd['p99_abs']:.2f} m; mean D26 {dd['mean_d26_m']:.0f} m).",
              f"* TCHP comes out {abs(dh['mean']):.2f} kJ/cm^2 too low on average ({abs(dh['mean_pct']):.1f}%; 99% within "
              f"{dh['p99_abs']:.2f}; mean TCHP {dh['mean_tchp']:.0f}).",
              f"* Steric height 1000-2000 dbar comes out {abs(st['mean']) * 100:.1f} cm too low, an almost constant "
              f"offset (std {st['std'] * 100:.1f} cm), which a tree model cannot see.", "",
              "These are small against the model's error (MAE about 10.5 to 10.7) and are systematic, so the learned "
              "correction absorbs them. The 2026 holdout was built with the same code, so the test stays consistent. "
              "Fixing it would mean re-downloading the 3D archives and rebuilding every daily field."]
    L += ["", "## Summary table", "",
          "| feature | origin | group | units | recipes | frozen | mean | median | 1st-99th pct | min - max | mode | % imputed |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rec = "/".join(x for x, ok in (("TCHP", r["in_tchp_recipe"]), ("D26", r["in_d26_recipe"])) if ok) or "-"
        fz = "/".join(x for x, ok in (("TCHP", r["in_frozen_tchp"]), ("D26", r["in_frozen_d26"])) if ok) or "-"
        md = fmt(r["mode"]) + ("" if r["mode_note"].startswith("modal") else f" ({r['mode_note']})")
        if r["mode_note"].startswith("modal"):
            md += " *"
        L.append(f"| `{r['feature']}` | {r['origin']} | {r['group']} | {r['units']} | {rec} | {fz} | {fmt(r['mean'])} | "
                 f"{fmt(r['median'])} | {fmt(r['p01'])} to {fmt(r['p99'])} | {fmt(r['min'])} to {fmt(r['max'])} | {md} | "
                 f"{r['pct_missing_imputed']:.1f}% |")
    L += ["", "\\* modal histogram bin, for continuous features.", "", "## Definitions and computation, by group", ""]
    for grp in dict.fromkeys(r["group"] for r in rows):
        L += [f"### {grp}", ""]
        for r in rows:
            if r["group"] != grp:
                continue
            f = r["feature"]
            L += [f"#### `{f}`  ({r['units']})", "",
                  f"* **Definition:** {r['definition']}",
                  f"* **Origin:** {r['origin']}. {r['upstream']}",
                  "* **Original sources:** " + ", ".join(link(*ref) for ref in ORIGIN[f][2]),
                  f"* **Computed:** {r['computation']}",
                  "* **Code:** " + ", ".join(code_link(p, t) for p, t in CODE[f]),
                  f"* **Statistics:** mean {fmt(r['mean'])}, std {fmt(r['std'])}, median {fmt(r['median'])}, "
                  f"quartiles {fmt(r['p25'])} to {fmt(r['p75'])}, 1st-99th pct {fmt(r['p01'])} to {fmt(r['p99'])}, "
                  f"min {fmt(r['min'])}, max {fmt(r['max'])}; mode {fmt(r['mode'])} ({r['mode_note']}); "
                  f"{r['n_valid']:,} valid, {r['pct_missing_imputed']:.1f}% imputed, "
                  f"{r['coverage_all_rows_pct']:.1f}% coverage across all rows.", ""]
    OUT_MD.write_text("\n".join(L) + "\n")
    print(T[["feature", "origin", "mean", "median", "pct_missing_imputed"]].to_string(index=False))
    print("wrote", OUT_MD, "and", OUT_CSV)


if __name__ == "__main__":
    main()
