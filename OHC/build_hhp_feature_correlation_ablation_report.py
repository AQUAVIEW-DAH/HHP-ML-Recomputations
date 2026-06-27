"""Build the feature-correlation / semi-ablation PDF report.

This generator produces a compact PDF that fits the current environment
without requiring a TeX engine. It intentionally keeps wide tables split by
target/section so the final output stays readable and avoids overlap.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path("/home/suramya/HHP-Prediction")
REPORT_DIR = ROOT / "OHC" / "output" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = REPORT_DIR / "hhp_feature_correlation_ablation_report.pdf"
TEX_PATH = REPORT_DIR / "hhp_feature_correlation_ablation_report.tex"


def fmt(value, nd: int = 3) -> str:
    try:
        return f"{float(value):.{nd}f}"
    except Exception:
        return str(value)


def load_tables() -> dict[str, pd.DataFrame]:
    base = ROOT / "OHC" / "output"
    return {
        "corr_tchp": pd.read_csv(base / "ml_benchmarks" / "added_physics_vs_delta_tchp_kj_per_cm2_pearson.csv"),
        "corr_d26": pd.read_csv(base / "ml_benchmarks" / "added_physics_vs_delta_d26_m_pearson.csv"),
        "redund": pd.read_csv(base / "ml_benchmarks" / "added_physics_high_redundancy_pairs.csv"),
        "year_ablation": pd.read_csv(base / "ml_benchmarks" / "year_holdout_physics_semi_ablation_summary.csv"),
        "locked": pd.read_csv(base / "ml_benchmarks" / "locked_physics_semi_ablation_oof_summary.csv"),
        "grouped": pd.read_csv(base / "ml_benchmarks" / "locked_physics_semi_ablation_grouped_all.csv"),
    }


GLOBAL_FEATURE_ROWS = [
    ("model_surface_temp_c", "surface temp T_s", "deg C", "No"),
    ("model_ssh_m", "sea-surface height eta", "m", "No"),
    ("model_mixed_layer_thickness_m", "mixed-layer thickness MLT", "m", "No"),
    ("model_surface_boundary_layer_thickness_m", "surface-boundary-layer thickness SBLT", "m", "No"),
    ("model_temp_excess_26c", "T_s - 26", "deg C", "No"),
    ("d26_minus_mlt_m", "D26 - MLT", "m", "Derived"),
    ("d26_minus_sblt_m", "D26 - SBLT", "m", "Derived"),
    ("d26_to_mlt_ratio", "D26 / MLT", "ratio", "Derived"),
    ("d26_to_sblt_ratio", "D26 / SBLT", "ratio", "Derived"),
    ("warm_layer_thickness_positive_m", "max(D26 - MLT, 0)", "m", "Derived"),
    ("model_ssh_x_abs_lat", "eta * |lat|", "m deg", "No"),
    ("model_mlt_x_abs_lat", "MLT * |lat|", "m deg", "No"),
    ("model_temp_excess_x_abs_lat", "(T_s - 26) * |lat|", "deg C deg", "No"),
]

PROFILE_FEATURE_ROWS = [
    ("model_steric_0_1000_m", "dyn_height(0/1000) / g", "m", "Yes: TEOS-10"),
    ("model_steric_0_2000_m", "dyn_height(0/2000) / g", "m", "Yes: TEOS-10"),
    ("model_steric_1000_ref2000_m", "steric(0/2000) - steric(0/1000)", "m", "Yes: TEOS-10"),
    ("model_n2_mean_upper200_s2", "mean(N^2; z <= 200 m)", "s^-2", "Yes: TEOS-10"),
    ("model_n2_max_upper200_s2", "max(N^2; z <= 200 m)", "s^-2", "Yes: TEOS-10"),
    ("model_n2_mean_to_d26_s2", "mean(N^2; z <= D26)", "s^-2", "Yes: TEOS-10"),
    ("model_n2_max_to_d26_s2", "max(N^2; z <= D26)", "s^-2", "Yes: TEOS-10"),
]

BASELINE_FEATURE_ROWS = [
    ("year", "calendar year of the collocated row", "time context"),
    ("month_int", "integer month 1-12", "seasonality"),
    ("lat", "latitude of the collocated Argo/RTOFS point", "geographic regime"),
    ("lon", "longitude of the collocated Argo/RTOFS point", "geographic regime"),
    ("abs_lat", "absolute latitude |lat|", "latitude-band physics"),
    ("nearest_rtofs_grid_distance_km", "distance from target point to nearest native RTOFS cell", "collocation quality"),
    ("month_sin", "sin(2 pi month / 12)", "cyclic seasonal encoding"),
    ("month_cos", "cos(2 pi month / 12)", "cyclic seasonal encoding"),
    ("doy_sin", "sin(2 pi doy / 366)", "cyclic annual encoding"),
    ("doy_cos", "cos(2 pi doy / 366)", "cyclic annual encoding"),
    ("is_winter_jfm", "1 for Jan-Feb-Mar rows", "season flag"),
    ("is_summer_jas", "1 for Jul-Aug-Sep rows", "season flag"),
    ("is_other", "1 outside JFM/JAS", "season flag"),
    ("model_interp_tchp_kj_per_cm2", "raw collocated RTOFS TCHP at the Argo point", "baseline model state"),
    ("model_interp_d26_m", "raw collocated RTOFS D26 at the Argo point", "baseline model state"),
]

FEATURE_SET_SUMMARY_ROWS = [
    ("base", "Baseline time, location, collocation-distance, and raw RTOFS TCHP/D26 features only."),
    ("base_plus_global_physics", "Base plus the full global RTOFS-side physics family from the 2-D diagnostics and daily field products."),
    ("global_pruned", "Base plus the pruned global physics subset: SSH, MLT, SBLT, temp-excess, D26-minus-MLT, D26-to-SBLT ratio, and three latitude interactions."),
    ("global_pruned_plus_profile_core", "global_pruned plus steric_1000_ref2000, N2_max_upper200, and N2_mean_to_D26."),
    ("drop_temp_lat_interaction", "global_pruned-style set with the temp-excess x |lat| interaction removed and profile-core terms added."),
    ("drop_ssh_lat_interaction", "global_pruned-style set with the SSH x |lat| interaction removed and profile-core terms added."),
    ("drop_both_lat_interactions", "D26-target set that removes both SSH x |lat| and temp-excess x |lat|, keeps MLT x |lat|, and retains profile-core terms."),
    ("surface_temp_swap", "Same role as drop_temp_lat_interaction but swaps temp-excess for raw surface temperature."),
]

AVAILABILITY_ROWS = [
    ("model_surface_temp_c", 41224),
    ("model_ssh_m", 41224),
    ("model_mixed_layer_thickness_m", 41224),
    ("model_surface_boundary_layer_thickness_m", 41224),
    ("model_temp_excess_26c", 41224),
    ("d26_minus_mlt_m", 12814),
    ("d26_minus_sblt_m", 12814),
    ("d26_to_mlt_ratio", 12814),
    ("d26_to_sblt_ratio", 12814),
    ("warm_layer_thickness_positive_m", 12814),
    ("model_ssh_x_abs_lat", 41224),
    ("model_mlt_x_abs_lat", 41224),
    ("model_temp_excess_x_abs_lat", 41224),
    ("model_steric_0_1000_m", 2954),
    ("model_steric_0_2000_m", 2785),
    ("model_steric_1000_ref2000_m", 2785),
    ("model_n2_mean_upper200_s2", 3068),
    ("model_n2_max_upper200_s2", 3068),
    ("model_n2_mean_to_d26_s2", 848),
    ("model_n2_max_to_d26_s2", 848),
]

APPENDIX_ROWS = [
    ("Code root", "/home/suramya/HHP-Prediction/OHC"),
    ("Collocation table", "output/ml_collocation/data/argo_rtofs_collocated_2024_2025.parquet"),
    ("Global-physics table", "output/ml_collocation/data/argo_rtofs_collocated_2024_2025_physics.parquet"),
    ("Profile-physics table", "output/ml_collocation/data/argo_rtofs_collocated_2024_2025_profile_physics.parquet"),
    ("Global feature builder", "build_rtofs_global_physics_features_2024_2025.py"),
    ("Profile feature builder", "build_rtofs_profile_physics_features_2024_2025.py"),
    ("Year ablation runner", "run_year_holdout_xgb_physics_semi_ablation.py"),
    ("Locked ablation runner", "run_locked_xgb_physics_semi_ablation.py"),
    ("Feature-feature corr matrix", "output/ml_benchmarks/added_physics_feature_corr_pearson.csv"),
    ("High-redundancy pairs", "output/ml_benchmarks/added_physics_high_redundancy_pairs.csv"),
    ("TCHP target corr", "output/ml_benchmarks/added_physics_vs_delta_tchp_kj_per_cm2_pearson.csv"),
    ("D26 target corr", "output/ml_benchmarks/added_physics_vs_delta_d26_m_pearson.csv"),
    ("Year ablation summary", "output/ml_benchmarks/year_holdout_physics_semi_ablation_summary.csv"),
    ("Locked OOF summary", "output/ml_benchmarks/locked_physics_semi_ablation_oof_summary.csv"),
    ("Locked grouped summary", "output/ml_benchmarks/locked_physics_semi_ablation_grouped_all.csv"),
]


def build_pdf() -> Path:
    tables = load_tables()

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22, spaceAfter=16))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], fontSize=16, leading=19, spaceBefore=12, spaceAfter=8))
    styles.add(ParagraphStyle(name="SubSection", parent=styles["Heading2"], fontSize=12.5, leading=15.5, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.7, leading=10.3))
    styles.add(ParagraphStyle(name="CodeInline", parent=styles["BodyText"], fontName="Courier", fontSize=8.3, leading=10))

    story = []

    def p(text: str, style: str = "BodyText") -> None:
        story.append(Paragraph(text, styles[style]))

    def s(height: float = 0.12) -> None:
        story.append(Spacer(1, height * inch))

    def make_table(data, widths, small: bool = False) -> None:
        font_size = 7.3 if small else 8.2
        table = Table(data, colWidths=widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F2")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), font_size),
                    ("LEADING", (0, 0), (-1, -1), font_size + 1.1),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#A6B4C3")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFD")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(table)

    def make_wrapped_table(data, widths, small: bool = False) -> None:
        font_size = 7.1 if small else 8.0
        cell_style = styles["Small"] if small else styles["BodyText"]
        wrapped = []
        for r, row in enumerate(data):
            out_row = []
            for cell in row:
                text = str(cell)
                if r == 0:
                    out_row.append(Paragraph(f"<b>{text}</b>", cell_style))
                else:
                    out_row.append(Paragraph(text.replace("|", "&#124;"), cell_style))
            wrapped.append(out_row)
        table = Table(wrapped, colWidths=widths, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F2")),
                    ("FONTSIZE", (0, 0), (-1, -1), font_size),
                    ("LEADING", (0, 0), (-1, -1), font_size + 1.5),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#A6B4C3")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFD")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(table)

    def add_page_num(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(doc.pagesize[0] - 0.55 * inch, 0.42 * inch, f"{canvas.getPageNumber()}")

    p("HHP ML Feature Correlation and Semi-Ablation Report", "TitleCenter")
    p("Prepared from the current code and output artifacts in <font name='Courier'>/home/suramya/HHP-Prediction/OHC</font>.")
    p("Date: 2026-06-15")
    s()

    p("Executive Summary", "Section")
    p(
        "This report documents the correlation diagnostics, redundancy analysis, and semi-ablation "
        "experiments for the RTOFS-to-Argo residual-learning workflow used in the HHP project. "
        "The strongest robust feature choices are target-specific: <font name='Courier'>global_pruned</font> "
        "for TCHP and <font name='Courier'>drop_both_lat_interactions</font> for D26 under the locked blocked-forward protocol."
    )
    p("All added features are computed on the RTOFS side only. Argo is used only for collocation coordinates, observed truth, and residual targets.")
    s()

    p("1. Scope and Artifact Locations", "Section")
    p("To keep the PDF readable, paths are shown relative to the code root <font name='Courier'>/home/suramya/HHP-Prediction/OHC</font>.")
    for label, rel_path in APPENDIX_ROWS[:8]:
        p(f"• <b>{label}:</b> <font name='Courier'>{rel_path}</font>", "Small")
    p("The merged collocation table contains 41,347 rows across 90 dates. The locked protocol evaluates 8,366 out-of-fold validation rows across 57 dates.")
    s()

    p("2. Feature Construction", "Section")
    p("2.1 Interpolation rule", "SubSection")
    p(
        "All RTOFS fields are sampled at the exact collocated Argo point using the same 8-neighbor native-grid interpolation rule. "
        "For non-exact points the implementation uses inverse-distance-squared weighting."
    )
    story.append(
        Preformatted(
            "x_hat = sum_i (w_i x_i) / sum_i w_i\n"
            "w_i   = 1 / max(d_i, 1e-6)^2",
            styles["CodeInline"],
        )
    )
    s(0.06)
    p("2.2 Which features are TEOS-10 and which are not", "SubSection")
    p(
        "The steric-height features and Brunt-Vaisala frequency summaries are computed with TEOS-10 "
        "via <font name='Courier'>gsw</font>. SSH, MLT, SBLT, and the latitude interaction terms are RTOFS diagnostics "
        "or direct algebraic combinations."
    )
    p(
        "Relative to the earlier tabular correction pipeline, the main additions in this report are: "
        "(i) model-side ocean-structure diagnostics sampled at the exact collocated point, "
        "(ii) steric-height summaries from the 3-D RTOFS profile, "
        "(iii) Brunt-Vaisala frequency summaries from the same profile, and "
        "(iv) semi-ablation experiments that explicitly prune correlated variants instead of only adding features."
    )
    s(0.06)
    p("2.3 Baseline features used across the ML pipeline", "SubSection")
    p(
        "Before any added physics features were introduced, the baseline ML pipeline already used time, geography, "
        "collocation quality, and the raw collocated RTOFS state. These baseline inputs appear in the original tabular "
        "benchmark, the year-holdout runs, the locked protocol, and the later semi-ablation experiments."
    )
    make_table([["Baseline feature", "Short meaning", "Why it is included"]] + BASELINE_FEATURE_ROWS, [1.7 * inch, 2.35 * inch, 2.0 * inch], small=True)
    PageBreak()
    p("2.4 Named feature-set recipes used in the experiments", "SubSection")
    p(
        "The experiment names in the ablation tables are shorthand for fixed feature recipes. "
        "This table gives a compact definition of the ones used in the current benchmarks."
    )
    make_wrapped_table([["Feature-set name", "Summary"]] + FEATURE_SET_SUMMARY_ROWS, [2.0 * inch, 4.1 * inch], small=True)
    s(0.08)
    p("2.4a Global/diagnostic added features", "SubSection")
    p(
        "These are the globally available RTOFS-side features sampled from the daily field products and "
        "the 2-D diagnostic grid, then optionally combined algebraically with model D26 or latitude."
    )
    make_table([["Global/diagnostic feature", "Definition", "Units", "TEOS-10?"]] + GLOBAL_FEATURE_ROWS, [2.15 * inch, 2.45 * inch, 0.7 * inch, 1.0 * inch], small=True)
    PageBreak()
    p("2.4b Profile added features", "SubSection")
    p(
        "These profile features require the 3-D RTOFS archive column and TEOS-10 thermodynamic calculations. "
        "They are available on a smaller subset of rows because deep valid columns are needed."
    )
    make_table([["Profile feature", "Definition", "Units", "TEOS-10?"]] + PROFILE_FEATURE_ROWS, [2.25 * inch, 2.5 * inch, 0.75 * inch, 1.0 * inch], small=True)
    PageBreak()

    p("2.5 TEOS-10 mathematics used in the added profile features", "SubSection")
    p("Pressure and thermodynamic conversions are built from the collocated RTOFS profile column:")
    story.append(
        Preformatted(
            "delta_z_k = delta_p_k / 9806\n"
            "S_A       = SA_from_SP(S_P, p, lon, lat)\n"
            "CT        = CT_from_t(S_A, t, p)",
            styles["CodeInline"],
        )
    )
    p("Dynamic height anomaly relative to reference pressure is converted to steric-height-style features in the current implementation as:")
    story.append(
        Preformatted(
            "DeltaPhi(0/p_ref) = geo_strf_dyn_height(S_A, CT, p, p_ref)\n"
            "H(0/p_ref)        = DeltaPhi(0/p_ref) / g\n"
            "g                 = 9.81 m s^-2   # current code constant",
            styles["CodeInline"],
        )
    )
    p("Brunt-Vaisala frequency is computed with TEOS-10 and then summarized over depth windows:")
    story.append(
        Preformatted(
            "N^2 = Nsquared(S_A, CT, p, lat)\n"
            "mean(N^2 ; z <= 200 m)\n"
            "max (N^2 ; z <= 200 m)\n"
            "mean(N^2 ; z <= D26)\n"
            "max (N^2 ; z <= D26)",
            styles["CodeInline"],
        )
    )
    s(0.06)
    p("2.6 Where SSH, MLT, and SBLT come from and why they were added", "SubSection")
    p(
        "The SSH, mixed-layer thickness, and surface-boundary-layer thickness features are not computed from Argo "
        "and are not TEOS-10 outputs. They are sampled from the cached global RTOFS diagnostic NetCDF file "
        "<font name='Courier'>rtofs_glo_2ds_f006_diag.nc</font> at the exact collocated location using the same "
        "8-neighbor interpolation rule. In code, these come from the variables "
        "<font name='Courier'>ssh</font>, <font name='Courier'>mixed_layer_thickness</font>, and "
        "<font name='Courier'>surface_boundary_layer_thickness</font> in "
        "<font name='Courier'>build_rtofs_global_physics_features_2024_2025.py</font>."
    )
    p(
        "They were added because the original residual-learning baseline only saw location, seasonality, and model "
        "TCHP/D26 values. The new diagnostics give the model extra information about surface state and upper-ocean structure "
        "without leaking Argo-derived physics into the inputs."
    )
    s(0.06)
    p("2.7 Feature availability", "SubSection")
    p(
        "The correlation calculations use a complete-case subset. Availability therefore matters: "
        "the deepest steric and D26-limited N^2 features only exist on a small subset of rows."
    )
    make_table([["Feature", "Finite rows"]] + [[name, f"{count:,}"] for name, count in AVAILABILITY_ROWS], [3.8 * inch, 1.2 * inch], small=True)
    PageBreak()

    p("3. Correlation and Redundancy Diagnostics", "Section")
    p("3.1 Pearson correlation coefficient used in the report", "SubSection")
    p("For two variables x and y over n complete-case rows, the Pearson coefficient is:")
    story.append(
        Preformatted(
            "r_xy = [ sum_i (x_i - x_bar)(y_i - y_bar) ] /\n"
            "       sqrt( sum_i (x_i - x_bar)^2 * sum_i (y_i - y_bar)^2 )",
            styles["CodeInline"],
        )
    )
    p(
        "In this report, x is one added RTOFS-side feature and y is a residual target: either "
        "<font name='Courier'>delta_tchp_kj_per_cm2</font> or <font name='Courier'>delta_d26_m</font>. "
        "The current merged complete-case subset for the added-feature correlation test contains 795 rows."
    )
    s(0.06)
    p("3.2 Top absolute Pearson correlations with the targets", "SubSection")
    make_table(
        [["Feature", "r with delta TCHP", "|r|"]]
        + [[row["feature"], fmt(row["pearson_r"]), fmt(row["abs_pearson_r"])] for _, row in tables["corr_tchp"].head(10).iterrows()],
        [3.55 * inch, 1.3 * inch, 0.65 * inch],
        small=True,
    )
    s(0.08)
    make_table(
        [["Feature", "r with delta D26", "|r|"]]
        + [[row["feature"], fmt(row["pearson_r"]), fmt(row["abs_pearson_r"])] for _, row in tables["corr_d26"].head(10).iterrows()],
        [3.55 * inch, 1.3 * inch, 0.65 * inch],
        small=True,
    )
    p(
        "How to read these tables: the column <font name='Courier'>r with delta TCHP</font> means Pearson correlation "
        "between one added RTOFS-side feature and the residual label "
        "<font name='Courier'>delta_tchp_kj_per_cm2 = Argo TCHP - RTOFS TCHP</font>. "
        "Likewise, <font name='Courier'>r with delta D26</font> means correlation with "
        "<font name='Courier'>delta_d26_m = Argo D26 - RTOFS D26</font>. "
        "A larger absolute value means the feature tracks the signed correction target more strongly on the complete-case subset."
    )
    p(
        "Immediate interpretation: for TCHP, the strongest linear relationships come from warm-layer geometry terms such as "
        "<font name='Courier'>d26_minus_sblt_m</font> and <font name='Courier'>d26_minus_mlt_m</font>. "
        "For D26, the strongest linear relationships come more from stratification and structure terms, especially "
        "<font name='Courier'>model_n2_max_upper200_s2</font>, mixed-layer thickness, and the steric-height family."
    )
    s(0.08)
    p("3.3 High-redundancy feature pairs", "SubSection")
    make_table(
        [["Feature A", "Feature B", "|r|"]]
        + [[row["feature_a"], row["feature_b"], fmt(row["abs_pearson_r"])] for _, row in tables["redund"].head(10).iterrows()],
        [2.7 * inch, 2.7 * inch, 0.65 * inch],
        small=True,
    )
    p("The strongest duplicates are temperature-excess vs. surface temperature, warm-layer-thickness vs. D26-minus-MLT, and the multiple steric-height variants.")
    p(
        "Interpretation: this table is not measuring skill against Argo directly. It is measuring redundancy between predictors. "
        "Very high |r| here means two features are carrying nearly the same information into the model, so keeping both can inflate "
        "feature importance instability without adding much new signal."
    )
    PageBreak()

    p("4. Semi-Ablation Results", "Section")
    p("4.1 Year-holdout semi-ablation (train 2024, test 2025)", "SubSection")
    p(
        "This pass is useful for ranking feature-set candidates, but it is more optimistic than the locked protocol "
        "because it tests on only 29 dates in 2025."
    )
    p(
        "In the year-holdout tables below, MAE and RMSE are errors against the Argo truth after applying the learned residual correction. "
        "<font name='Courier'>Gain vs raw</font> means the reduction in MAE relative to using raw collocated RTOFS alone."
    )
    year_tchp = tables["year_ablation"][tables["year_ablation"]["target"] == "tchp"].sort_values("mae").head(5)
    year_d26 = tables["year_ablation"][tables["year_ablation"]["target"] == "d26"].sort_values("mae").head(5)
    make_table(
        [["TCHP model", "MAE", "RMSE", "Gain vs raw"]]
        + [[row["model"], fmt(row["mae"]), fmt(row["rmse"]), fmt(row["mae_gain_vs_raw"])] for _, row in year_tchp.iterrows()],
        [3.4 * inch, 0.8 * inch, 0.8 * inch, 0.9 * inch],
        small=True,
    )
    s(0.08)
    make_table(
        [["D26 model", "MAE", "RMSE", "Gain vs raw"]]
        + [[row["model"], fmt(row["mae"]), fmt(row["rmse"]), fmt(row["mae_gain_vs_raw"])] for _, row in year_d26.iterrows()],
        [3.4 * inch, 0.8 * inch, 0.8 * inch, 0.9 * inch],
        small=True,
    )
    p(
        "Immediate interpretation: the year-holdout pass suggested a slightly different winner for each target. "
        "That was encouraging, but not yet enough to freeze the feature sets because the evaluation window was still comparatively narrow."
    )
    s(0.08)
    p("4.2 Locked blocked-forward semi-ablation", "SubSection")
    p(
        "This is the stronger decision table because it aggregates 57 validation dates and forces each fold to predict "
        "forward in time with a one-date embargo."
    )
    p(
        "Here, <font name='Courier'>Corr.</font> is the Pearson correlation between the corrected prediction and the Argo truth over the pooled out-of-fold rows. "
        "<font name='Courier'>Bias</font> is prediction minus truth, so a negative bias means the corrected model is still underestimating Argo on average."
    )
    locked_tchp = tables["locked"][tables["locked"]["target"] == "tchp"].copy()
    locked_d26 = tables["locked"][tables["locked"]["target"] == "d26"].copy()
    make_table(
        [["TCHP model", "MAE", "RMSE", "Bias", "Corr."]]
        + [[row["model"], fmt(row["mae"]), fmt(row["rmse"]), fmt(row["bias"]), fmt(row["corr"])] for _, row in locked_tchp.iterrows()],
        [3.15 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch, 0.7 * inch],
        small=True,
    )
    s(0.08)
    make_table(
        [["D26 model", "MAE", "RMSE", "Bias", "Corr."]]
        + [[row["model"], fmt(row["mae"]), fmt(row["rmse"]), fmt(row["bias"]), fmt(row["corr"])] for _, row in locked_d26.iterrows()],
        [3.15 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch, 0.7 * inch],
        small=True,
    )
    p(
        "Immediate interpretation: under the stricter locked protocol, D26 kept its semi-ablation winner "
        "(<font name='Courier'>drop_both_lat_interactions</font>), while TCHP reverted to the simpler and more robust "
        "<font name='Courier'>global_pruned</font> feature set. That reversal is precisely the kind of issue the locked protocol is meant to catch."
    )
    PageBreak()

    p("4.3 Seasonal locked results", "SubSection")
    grouped = tables["grouped"]
    season_t = grouped[
        (grouped["target"] == "tchp")
        & (grouped["grouping"] == "season")
        & (grouped["model"].isin(["raw_rtofs", "base", "global_pruned", "drop_temp_lat_interaction"]))
    ].copy()
    season_d = grouped[
        (grouped["target"] == "d26")
        & (grouped["grouping"] == "season")
        & (grouped["model"].isin(["raw_rtofs", "base", "global_pruned", "drop_both_lat_interactions", "global_pruned_plus_profile_core"]))
    ].copy()
    make_table(
        [["TCHP season", "Model", "MAE", "Bias", "Corr."]]
        + [[row["group"], row["model"], fmt(row["mae"]), fmt(row["bias"]), fmt(row["corr"])] for _, row in season_t.sort_values(["group", "mae"]).iterrows()],
        [1.0 * inch, 2.5 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch],
        small=True,
    )
    s(0.08)
    make_table(
        [["D26 season", "Model", "MAE", "Bias", "Corr."]]
        + [[row["group"], row["model"], fmt(row["mae"]), fmt(row["bias"]), fmt(row["corr"])] for _, row in season_d.sort_values(["group", "mae"]).iterrows()],
        [1.0 * inch, 2.5 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch],
        small=True,
    )
    p(
        "Immediate interpretation: these grouped tables check whether the global winners are only winning because of one season. "
        "For TCHP, <font name='Courier'>global_pruned</font> stays competitive across winter, summer, and other months. "
        "For D26, <font name='Courier'>drop_both_lat_interactions</font> is strongest overall, especially in summer and the non-JFM/JAS subset, "
        "while the profile-core variant remains close in winter."
    )
    s()

    p("5. Interpretation", "Section")
    p("• The added RTOFS-only physics features are useful, but only after pruning correlated variants.")
    p("• The strongest exact duplicates are the expected construction pairs: <font name='Courier'>model_surface_temp_c</font> vs. <font name='Courier'>model_temp_excess_26c</font>, and <font name='Courier'>warm_layer_thickness_positive_m</font> vs. <font name='Courier'>d26_minus_mlt_m</font>.")
    p("• Profile-physics features help more for D26 than for TCHP. That is physically plausible because D26 is more directly tied to upper-ocean structure and stratification.")
    p("• The year-holdout pass suggested <font name='Courier'>drop_temp_lat_interaction</font> for TCHP and <font name='Courier'>drop_both_lat_interactions</font> for D26. After the stricter locked validation, the D26 winner held, but TCHP reverted to the simpler <font name='Courier'>global_pruned</font> feature set.")
    p("• Current recommendation: <font name='Courier'>global_pruned</font> for TCHP and <font name='Courier'>drop_both_lat_interactions</font> for D26.")
    s()

    p("6. File Reference Appendix", "Section")
    p("The table below lists the main output and source locations relative to <font name='Courier'>/home/suramya/HHP-Prediction/OHC</font>.")
    make_table([["Artifact", "Relative path"]] + [[label, rel_path] for label, rel_path in APPENDIX_ROWS[1:]], [2.2 * inch, 4.0 * inch], small=True)

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=0.62 * inch,
        rightMargin=0.62 * inch,
        topMargin=0.68 * inch,
        bottomMargin=0.55 * inch,
    )
    doc.build(story, onFirstPage=add_page_num, onLaterPages=add_page_num)
    return PDF_PATH


def build_tex_stub() -> Path:
    content = r"""\documentclass[11pt]{report}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\title{HHP ML Feature Correlation and Semi-Ablation Report}
\author{Generated alongside the PDF report}
\date{2026-06-15}
\begin{document}
\maketitle
\chapter*{Note}
This LaTeX file is a lightweight companion source saved next to the PDF.
The fully formatted PDF report was generated in the current environment with
the Python report generator:
\begin{verbatim}
/home/suramya/HHP-Prediction/OHC/build_hhp_feature_correlation_ablation_report.py
\end{verbatim}
because a TeX engine is not installed here.

The canonical PDF for review is:
\begin{verbatim}
/home/suramya/HHP-Prediction/OHC/output/reports/hhp_feature_correlation_ablation_report.pdf
\end{verbatim}
\end{document}
"""
    TEX_PATH.write_text(content)
    return TEX_PATH


def main() -> None:
    pdf = build_pdf()
    tex = build_tex_stub()
    print(pdf)
    print(tex)


if __name__ == "__main__":
    main()
