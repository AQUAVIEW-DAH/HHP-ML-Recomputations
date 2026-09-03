# Lat/lon-only error models — 2026-09-03

Dr. Jacobs' request (email, 2026-09): fit the TCHP and D26 differences with
features being just latitude and longitude, on zero-filled targets (TCHP=0 and
D26=0 where a side has no 26 °C water; mixed cases zero-fill the missing side),
so coverage spans all latitudes. Try a random forest (visualise the lat/lon
partition and its discontinuities), SVR with an RBF kernel, and Gaussian
process regression with separate lat/lon kernel scales.

Script `OHC/exploration/run_latlon_only_models.py`; outputs
`OHC/output/latlon_only_20260903/`; Drive `HHP-analysis-2026-09-03/latlon_only/`.
Locked blocked-forward protocol (3 folds, 1-date embargo), all 322,616 rows.

## Results

| lat/lon only, OOF MAE | TCHP all rows | TCHP warm | D26 all rows | D26 warm |
|---|---|---|---|---|
| raw RTOFS | 4.89 | 16.61 | 5.23 | 14.92 |
| random forest | **4.05** | **12.44** | 5.00 | 12.50 |
| SVR (RBF) | 4.63 | 13.49 | 5.13 | 12.63 |
| XGBoost (locked hyperparameters) | 5.35 | 13.48 | 5.74 | 12.62 |
| GP (anisotropic RBF) | 4.24 | 12.69 | **4.98** | **12.17** |

("warm" = both sides have 26 °C water, 60,666 rows — the scale of the locked
table, where the full 35-feature model sits at 11.19 / 10.55.)

0. **Learner-controlled comparison** (added after review): with the same locked
   XGBoost hyperparameters, lat/lon-only scores 13.48 / 12.62 vs 11.40 / 10.76
   with the 35 features — so the feature contribution, learner held fixed, is
   ~2.1 (TCHP) / ~1.9 (D26) MAE. Depth-4 boosting is the weakest pure-geography
   learner (worse than raw on the zero-filled scale: shallow trees smear
   corrections into the zero region); the forest and the anisotropic GP carve
   position best.
1. **Geography alone recovers ~75% of the correction.** Position is a static
   bias map worth ~4.2 (TCHP) / ~2.5 (D26) MAE; all physics features together
   add the last ~1.2–1.6. Sharpest form of the redundancy point: any feature
   that cannot beat the lat/lon floor is re-encoding geography.
2. **The error field is zonally elongated.** The GP fitted ~7.5° latitude vs
   33° (TCHP) / 62° (D26) longitude length scales — his anisotropy suggestion,
   confirmed quantitatively.
3. **Boxes vs smooth is a near-tie.** RF wins TCHP by 0.25; GP wins D26 by
   0.33 and has no discontinuities. RF subdomain jumps reach ~10–20 kJ/cm²
   in the warm pool (see `*_rf_discontinuity_map.png`) — an argument for
   smooth spatial representations if corrected fields are ever served as maps.
4. The 64-leaf partition-box figure shows the tree discovering the tropical
   band edges (±17°), the West Pacific warm pool, and the Gulf unaided.

Zero-fill convention note: this run adopts TCHP=0 / D26:=0 where the column
never reaches 26 °C, resolving the long-open none-vs-zero question in the
mentor's favour of zero.
