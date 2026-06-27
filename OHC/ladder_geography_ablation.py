"""Nested geography-ladder ablation to separate latitude-band physics from location lookup.

Reuses the EXACT data loader, preprocessor, model, and eval paths from
test_claude_methodology_recommendations.py; only the feature set varies.

Feature ladder (nested: F0 < F1a < F1b < F2 == GEO_STATE):
  F0  = state-only                         (no geography at all)
  F1a = state + abs_lat                     (+ raw latitude band)
  F1b = state + abs_lat + abs_lat-interactions   (+ latitude-band physics)
  F2  = F1b + lat + lon  == GEO_STATE       (+ absolute location)

Increments per (target, split):
  dABSLAT = MAE(F0) - MAE(F1b)   (gain from latitude-band physics)
  dLON    = MAE(F1b) - MAE(F2)   (gain from raw lat/lon == candidate location-lookup)

Key diagnostic:
  dLON(year_holdout) - dLON(tile_group_oof)
    = skill that the leaky year split rewards but disappears under spatial holdout
    = the location-lookup component that will NOT transfer to the float-desert.

Splits:
  year_holdout      : train 2024 / test 2025 (forward-time; 79% same-float leakage)
  tile_group_oof    : GroupKFold by 15-deg tile (spatial transfer; breaks fine lookup)
  platform_group_oof: GroupKFold by float id (identity-leakage guard; mixes years)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OHC import test_claude_methodology_recommendations as tcmr  # noqa: E402

OUT_DIR = Path("/home/suramya/HHP-Prediction/OHC/output/ml_benchmarks/claude_methodology_tests/ladder_ablation")

ABSLAT_INTERACTIONS = ["model_ssh_x_abs_lat", "model_mlt_x_abs_lat", "model_temp_excess_x_abs_lat"]
F0 = list(tcmr.STATE_ONLY_FEATURES)
F1A = F0 + ["abs_lat"]
F1B = F0 + ["abs_lat"] + ABSLAT_INTERACTIONS
F2 = F0 + ["abs_lat"] + ABSLAT_INTERACTIONS + ["lat", "lon"]
LADDER = {"F0_state_only": F0, "F1a_abslat": F1A, "F1b_abslat_physics": F1B, "F2_geo_state": F2}

# Sanity: F2 must equal GEO_STATE as a set (so F2 reproduces Codex's geo_state number).
assert set(F2) == set(tcmr.GEO_STATE_FEATURES), set(F2) ^ set(tcmr.GEO_STATE_FEATURES)
assert set(F0) == set(tcmr.STATE_ONLY_FEATURES)


def _year_holdout(df, target, feats):
    pred = tcmr._train_year_holdout_predictions(df, target, feats, "ladder")
    y = pred[target.obs_col].to_numpy(float)
    p = pred["prediction"].to_numpy(float)
    raw = pred[target.model_col].to_numpy(float)
    return dict(rows=int(len(pred)), mae=float(np.mean(np.abs(y - p))),
                bias=float(np.mean(p - y)), raw_mae=float(np.mean(np.abs(y - raw))))


def _grouped_oof(df, target, feats, group_col, n_splits=5):
    work = df[pd.notna(df[target.obs_col]) & pd.notna(df[target.model_col]) & pd.notna(df[target.delta_col])].copy()
    work = tcmr._prepare_features(work)
    groups = work[group_col].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    preds = np.full(len(work), np.nan, dtype=float)
    for tr_idx, va_idx in gkf.split(work, groups=groups):
        tr, va = work.iloc[tr_idx], work.iloc[va_idx]
        pre = tcmr._preprocessor(feats)
        x_tr = pre.fit_transform(tr[feats])
        x_va = pre.transform(va[feats])
        m = tcmr._make_model(target.name)
        m.fit(x_tr, tr[target.delta_col].to_numpy(float))
        preds[va_idx] = va[target.model_col].to_numpy(float) + m.predict(x_va)
    valid = np.isfinite(preds)
    s = work.iloc[valid]
    y = s[target.obs_col].to_numpy(float)
    p = preds[valid]
    raw = s[target.model_col].to_numpy(float)
    return dict(rows=int(valid.sum()), mae=float(np.mean(np.abs(y - p))),
                bias=float(np.mean(p - y)), raw_mae=float(np.mean(np.abs(y - raw))))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = tcmr._load_enriched_collocation()
    df["tile_group"] = [tcmr._tile_id(lat, lon) for lat, lon in zip(df["lat"], df["lon"])]
    df["platform_group"] = df["platform"].fillna("unknown").astype(str)
    df_known = df[pd.notna(df["platform"])].copy()

    rows = []
    for target in tcmr.TARGETS:
        for fname, feats in LADDER.items():
            rows.append({"target": target.name, "split": "year_holdout", "feature_set": fname, **_year_holdout(df, target, feats)})
            rows.append({"target": target.name, "split": "tile_group_oof", "feature_set": fname, **_grouped_oof(df, target, feats, "tile_group")})
            rows.append({"target": target.name, "split": "platform_group_oof", "feature_set": fname, **_grouped_oof(df_known, target, feats, "platform_group")})

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "ladder_ablation_metrics.csv", index=False)

    # Increment table + key diagnostic
    incr = []
    for target in tcmr.TARGETS:
        for split in ["year_holdout", "tile_group_oof", "platform_group_oof"]:
            sub = res[(res.target == target.name) & (res.split == split)].set_index("feature_set")
            mae = sub["mae"].to_dict()
            raw = float(sub["raw_mae"].iloc[0])
            incr.append({
                "target": target.name, "split": split, "raw_mae": round(raw, 3),
                "F0_state": round(mae["F0_state_only"], 3),
                "F1a_+abslat": round(mae["F1a_abslat"], 3),
                "F1b_+abslat_phys": round(mae["F1b_abslat_physics"], 3),
                "F2_+latlon(geo)": round(mae["F2_geo_state"], 3),
                "dABSLAT(F0-F1b)": round(mae["F0_state_only"] - mae["F1b_abslat_physics"], 3),
                "dLON(F1b-F2)": round(mae["F1b_abslat_physics"] - mae["F2_geo_state"], 3),
            })
    incr_df = pd.DataFrame(incr)
    incr_df.to_csv(OUT_DIR / "ladder_increments.csv", index=False)

    # Headline diagnostic: dLON(year) - dLON(tile)  =  location-lookup skill that won't transfer
    diag = []
    for target in tcmr.TARGETS:
        d = incr_df[incr_df.target == target.name].set_index("split")["dLON(F1b-F2)"]
        diag.append({
            "target": target.name,
            "dLON_year_holdout": float(d["year_holdout"]),
            "dLON_tile_group": float(d["tile_group_oof"]),
            "dLON_platform_group": float(d["platform_group_oof"]),
            "leaky_lookup (year - tile)": round(float(d["year_holdout"]) - float(d["tile_group_oof"]), 3),
        })
    diag_df = pd.DataFrame(diag)

    print("\n================ LADDER ABLATION: MAE by feature set ================")
    print(incr_df.to_string(index=False))
    print("\n================ KEY DIAGNOSTIC: dLON across splits =================")
    print("dLON = MAE(state+abslat+phys) - MAE(+lat/lon).  Larger = lat/lon helps more.")
    print("'leaky_lookup' = dLON(year) - dLON(tile): skill the leaky year split rewards but")
    print("                 that vanishes under spatial holdout (won't transfer to desert).")
    print(diag_df.to_string(index=False))

    # Reproduction check vs Codex's published numbers
    print("\n================ REPRODUCTION CHECK (should match Codex) ============")
    codex = {("tchp", "year_holdout", "F0_state"): 12.58, ("tchp", "year_holdout", "F2_+latlon(geo)"): 11.56,
             ("d26", "year_holdout", "F0_state"): 12.88, ("d26", "year_holdout", "F2_+latlon(geo)"): 11.36,
             ("tchp", "tile_group_oof", "F0_state"): 13.13, ("tchp", "tile_group_oof", "F2_+latlon(geo)"): 11.95,
             ("d26", "tile_group_oof", "F0_state"): 12.79, ("d26", "tile_group_oof", "F2_+latlon(geo)"): 10.91,
             ("tchp", "platform_group_oof", "F0_state"): 12.65, ("tchp", "platform_group_oof", "F2_+latlon(geo)"): 11.18,
             ("d26", "platform_group_oof", "F0_state"): 12.24, ("d26", "platform_group_oof", "F2_+latlon(geo)"): 10.30}
    for (tgt, split, col), exp in codex.items():
        got = float(incr_df[(incr_df.target == tgt) & (incr_df.split == split)][col].iloc[0])
        flag = "OK" if abs(got - exp) <= 0.06 else "DIFF"
        print(f"  [{flag}] {tgt:4s} {split:18s} {col:16s} mine={got:6.2f} codex={exp:6.2f}")

    (OUT_DIR / "ladder_diagnostic.json").write_text(json.dumps(diag, indent=2))
    print(f"\nWrote: {OUT_DIR}/ladder_ablation_metrics.csv, ladder_increments.csv, ladder_diagnostic.json")


if __name__ == "__main__":
    main()
