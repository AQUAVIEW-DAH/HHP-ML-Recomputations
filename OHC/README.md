# OHC

TEOS-10-backed Ocean Heat Content / Tropical Cyclone Heat Potential utilities.

This directory is meant to be the clean home for the standardized upper-ocean
heat-content calculation that we can use for both:

- Argo GDAC profiles
- RTOFS model profiles

## Scope

The TEOS-10 / GSW toolbox provides the thermodynamic building blocks:

- `gsw_SA_from_SP`
- `gsw_p_from_z`
- `gsw_z_from_p`
- `gsw_rho_t_exact`
- `gsw_cp_t_exact`

TCHP/OHC itself is then built on top of those ingredients as:

```text
TCHP = ∫[surface -> D26] rho(SA, t, p) * cp(SA, t, p) * (T(z) - 26°C) dz
```

where `D26` is the depth of the 26 C isotherm.

## Notes

- TEOS-10 supports Python through the `gsw` package.
- TEOS-10 does not provide a built-in `tchp` function; we compute the metric
  ourselves using TEOS-derived seawater properties.
- Output convention here is:
  - `J/m²`
  - `kJ/cm²`

## Files

- `teos_ohc.py`: core calculation code
- `single_profile_ohc_demo.py`: one hardcoded Argo-vs-RTOFS standalone example
- `build_monthly_ohc_pairs.py`: sampled monthly Argo-vs-RTOFS collocation builder
- `make_monthly_ohc_report.py`: plot generation + LaTeX PDF report

## Monthly Comparison Workflow

The monthly comparison currently uses a practical sampled-date strategy:

- region: Gulf of Mexico
- year: 2024
- months: January, February, March, August, September, October
- pairing: same-day Argo GDAC profile vs nearest RTOFS column at the profile coordinates
- metric: TEOS-10-backed TCHP/OHC above the 26 C isotherm

Because full daily RTOFS coverage is large, the monthly builder selects the
top-N dates in each month with the highest Argo profile counts, then computes a
like-for-like collocation comparison on those dates.
