# HHP Prediction

ML-corrected Hurricane Heat Potential (HHP/TCHP) replay prototype built on:

- NOAA RTOFS 3D ocean model fields
- Argo GDAC in-situ profiles
- TEOS-10 / GSW thermodynamics
- a lightweight tabular ML correction layer

This project is a sibling to [`MLD-ML-Recomputation`](../MLD-ML-Recomputation), but it targets tropical cyclone heat potential instead of mixed layer depth.

## What the app does

The current app is a **Milton 2024 replay sandbox**:

- choose a replay date
- click a point in the Gulf of Mexico
- see raw RTOFS TCHP
- see the ML-corrected TCHP
- inspect nearby Argo profiles used as provenance
- switch between raw field, final corrected field, and correction field

The profile panel now shows only:

- Argo temperature profile
- raw RTOFS temperature profile
- corrected scalar TCHP result

It does **not** display a fake corrected temperature-vs-depth curve.

## HHP definition

We use the standard tropical cyclone heat potential form:

```text
TCHP = ∫[0 → D26] ρ(z) · Cp(z) · (T(z) − 26°C) dz
```

Where:

- `D26` is the depth of the 26 C isotherm
- `ρ(z)` is in-situ density
- `Cp(z)` is isobaric heat capacity
- `T(z)` is the in-situ temperature profile

Output units are `kJ/cm²`.

## Thermodynamics

The active HHP computation path uses **TEOS-10** through the Python `gsw` package:

- Practical Salinity -> Absolute Salinity
- pressure/depth-aware thermodynamics
- depth-varying density
- depth-varying heat capacity

Implementation lives in [hhp_core.py](/home/suramya/HHP-Prediction/hhp_core.py).

For Argo profiles:

- input vertical axis is pressure (`dbar`)
- depth is derived with TEOS-10

For RTOFS profiles:

- input vertical axis is depth (`m`)
- pressure is derived with TEOS-10 from depth and latitude

## In-situ scope

For now, the in-situ source is **Argo GDAC only**.

That means:

- no WOD in the active training set
- no gliders in the active training set
- no mixed-source profile fusion yet

This keeps the training corpus consistent while the TEOS-10-backed HHP pipeline is being stabilized.

## Current training data

Current TEOS-10 / Argo-only replay datasets:

- `BERYL_2024`
- `HELENE_2024`
- `MILTON_2024`

Artifacts live under `artifacts/datasets/`.

The current default deployed model artifact is:

- [artifacts/models/argo_gdac_teos10_2024_allstorms_rf_delta_model.pkl](/home/suramya/HHP-Prediction/artifacts/models/argo_gdac_teos10_2024_allstorms_rf_delta_model.pkl)

## Current ML setup

Target:

```text
delta_tchp = obs_tchp - model_tchp
```

Prediction:

```text
final_tchp = raw_rtofs_tchp + predicted_delta
```

Simple tabular baselines are benchmarked:

- raw RTOFS
- ridge regression
- gradient-boosted trees
- random forest

Current best all-storm grouped-CV artifact:

- `rf_delta`

Benchmark report:

- [artifacts/reports/benchmarks/argo_gdac_teos10_2024_allstorms_benchmark.md](/home/suramya/HHP-Prediction/artifacts/reports/benchmarks/argo_gdac_teos10_2024_allstorms_benchmark.md)

## Key files

```text
hhp_core.py
  TEOS-10-backed D26 / TCHP computation

hhp_pipeline.py
  point-query orchestration and ML correction application

api.py
  FastAPI backend serving both API and built frontend

ml/sources/argo_gdac_source.py
  Argo GDAC discovery, download, QC, extraction

ml/processing/build_storm_pairs.py
  generic storm-event Argo/RTOFS pairing

ml/processing/build_milton_pairs.py
  Milton replay pair builder used by the sandbox app

ml/train/train_multistorm_gbr.py
  multi-event model benchmarking and artifact freezing

dashboard/
  React/Vite frontend
```

## Current limitations

- training still uses storm-bounded event datasets, not a full basin-season matchup archive
- only Argo is active in the training set
- ML corrects scalar TCHP, not the full vertical temperature profile
- corrected field is meaningful; corrected profile shape is intentionally not visualized as a fake curve

## Next direction

- expand from storm-bounded pairs to broader Atlantic hurricane-season Argo/RTOFS matchups
- standardize TEOS-derived profile products and diagnostics
- add clearer provenance around model version, TEOS method, and matchup coverage
