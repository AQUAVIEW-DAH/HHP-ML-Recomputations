# Mathematical documentation

Exact math behind each experiment/analysis in this repo, one file per family,
kept in the same commits as the code that implements it. GitHub renders the
`$...$` / `$$...$$` LaTeX natively, so these are readable directly in the web
UI. Update via the `document-math` skill (`.claude/skills/document-math/`).

| File | Scope | Status |
|---|---|---|
| [moe_recommended_model.md](moe_recommended_model.md) | **The recommended model**: mixture-of-experts blend — architecture, hyperparameters, every ML method evaluated, caveats | current |
| [gom_attribution.md](gom_attribution.md) | Gulf of Mexico SHAP attribution, backward elimination, distributions (`OHC/exploration/run_gom_attribution_analysis.py`) | current |
