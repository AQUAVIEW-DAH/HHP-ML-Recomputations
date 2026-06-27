"""Generate the AQUAVIEW HHP indexing-request PDF.

Produces docs/aquaview_hhp_indexing_request.pdf with two tiered tables of
datasets that AQUAVIEW should index for Hurricane Heat Potential (HHP)
work, including clickable upstream data and documentation URLs.
"""
from __future__ import annotations

from pathlib import Path
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "aquaview_hhp_indexing_request.pdf"

# Custom styles ---------------------------------------------------------------
styles = getSampleStyleSheet()

style_title = ParagraphStyle(
    "Title",
    parent=styles["Title"],
    fontSize=18,
    leading=22,
    spaceAfter=6,
    textColor=colors.HexColor("#0f172a"),
)
style_subtitle = ParagraphStyle(
    "Subtitle",
    parent=styles["Normal"],
    fontSize=11,
    leading=14,
    textColor=colors.HexColor("#475569"),
    spaceAfter=14,
)
style_h2 = ParagraphStyle(
    "H2",
    parent=styles["Heading2"],
    fontSize=14,
    leading=18,
    spaceBefore=14,
    spaceAfter=8,
    textColor=colors.HexColor("#0f172a"),
)
style_body = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=10,
    leading=13,
    textColor=colors.HexColor("#1e293b"),
    spaceAfter=8,
)
style_cell = ParagraphStyle(
    "Cell",
    parent=styles["Normal"],
    fontSize=8.5,
    leading=10.5,
    textColor=colors.HexColor("#1e293b"),
    alignment=TA_LEFT,
)
style_cell_bold = ParagraphStyle(
    "CellBold",
    parent=style_cell,
    fontName="Helvetica-Bold",
    textColor=colors.HexColor("#0f172a"),
)
style_link = ParagraphStyle(
    "CellLink",
    parent=style_cell,
    textColor=colors.HexColor("#1d4ed8"),
)
style_caption = ParagraphStyle(
    "Caption",
    parent=styles["Normal"],
    fontSize=8.5,
    leading=11,
    textColor=colors.HexColor("#64748b"),
    spaceAfter=4,
)


def link(url: str, label: str | None = None) -> str:
    """Return a hyperlinked HTML fragment for ReportLab paragraph use."""
    label = label or url
    return f'<link href="{url}" color="#1d4ed8"><u>{label}</u></link>'


def cell(text: str, style=style_cell) -> Paragraph:
    return Paragraph(text, style)


def link_cell(url: str, label: str | None = None) -> Paragraph:
    return Paragraph(link(url, label), style_link)


# Header ----------------------------------------------------------------------
def header_flowables() -> list:
    return [
        Paragraph("AQUAVIEW Indexing Request — HHP Prediction", style_title),
        Paragraph(
            "Datasets to add to the AQUAVIEW MCP catalog to support Hurricane Heat Potential "
            "(HHP / TCHP) prediction work.",
            style_subtitle,
        ),
        Paragraph(
            f"<b>Author:</b> Suramya Angdembay, Institute for Advanced Analytics and Society, USM. "
            f"&nbsp;&nbsp;<b>Date:</b> {date.today().isoformat()}.",
            style_caption,
        ),
        Paragraph(
            "<b>Status:</b> all gaps below were verified against the live AQUAVIEW MCP "
            "(catalog endpoint <font face='Courier' size='8'>https://mcp.aquaview.org/mcp</font>) "
            "using multi-strategy keyword and collection probes. Items already in AQUAVIEW are "
            "not re-listed here. The full corrected inventory of what is already indexed "
            "(notably the comprehensive AOML hurricane archive in collection NOAA_AOML_HDB) is "
            "available in the project's working notes.",
            style_caption,
        ),
        Spacer(1, 0.10 * inch),
    ]


# Tier tables ---------------------------------------------------------------
TIER1_HEADER = ["Dataset", "Why it's missing", "Why HHP cares", "Data access", "Documentation"]
TIER2_HEADER = ["Dataset", "Status in AQUAVIEW", "Why HHP cares", "Data access", "Documentation"]

TIER1_ROWS = [
    {
        "name": "CMEMS GLORYS-family TEMPERATURE at depth (T(z))",
        "why_missing": (
            "<font face='Courier' size='7'>salinity_model_g_&lt;year&gt;</font> exists but no "
            "<font face='Courier' size='7'>temperature_model_g_&lt;year&gt;</font> partner. "
            "Verified — q=\"temperature_model\" returns 0."
        ),
        "why_hhp": (
            "Without GLORYS T(z), we cannot compute GLORYS-derived TCHP as an independent comparator "
            "to RTOFS-derived TCHP. The salinity-only ingestion is presumably for Argo S calibration; "
            "we need the partner T fields."
        ),
        "data_url": "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description",
        "data_label": "Marine Copernicus product page (GLOBAL_MULTIYEAR_PHY_001_030)",
        "doc_url": "https://help.marine.copernicus.eu/en/articles/7949409-copernicus-marine-toolbox-introduction",
        "doc_label": "copernicus-marine toolbox docs",
    },
    {
        "name": "CMEMS GLORYS-family CURRENTS (u, v at depth)",
        "why_missing": (
            "Also missing as partner to <font face='Courier' size='7'>salinity_model_g_*</font> "
            "(same product family ships T, S, u, v together)."
        ),
        "why_hhp": (
            "Lower priority for HHP scalar output, but needed if we extend to MLD / eddy-correction "
            "work or to a Loop-Current intrusion mode."
        ),
        "data_url": "https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description",
        "data_label": "Marine Copernicus product page",
        "doc_url": "https://catalogue.marine.copernicus.eu/documents/PUM/CMEMS-GLO-PUM-001-030.pdf",
        "doc_label": "CMEMS PUM (Product User Manual)",
    },
    {
        "name": "Real-time 3D RTOFS (rtofs_glo_3dz_*)",
        "why_missing": (
            "AQUAVIEW UAF aggregator has 2 RTOFS items, both 2D surface forecasts. The 3D "
            "f006/f024 NetCDF files we use directly from S3 are not STAC items."
        ),
        "why_hhp": (
            "Direct AQUAVIEW discovery of the data we already pull would let any future user "
            "replicate the pipeline without bespoke S3 path knowledge."
        ),
        "data_url": "https://registry.opendata.aws/noaa-rtofs/",
        "data_label": "AWS Open Data Registry — NOAA Global RTOFS",
        "doc_url": "https://www.nco.ncep.noaa.gov/pmb/products/rtofs/",
        "doc_label": "NCO RTOFS product page",
    },
]

TIER2_ROWS = [
    {
        "name": "ECMWF ORAS5",
        "status": "Confirmed missing — q=\"ORAS5\" returns 0.",
        "why_hhp": (
            "OHC700 climate-scale comparator. Independent NEMOVAR 3DVar-FGAT reanalysis line; "
            "useful once we extend beyond Atlantic-basin focus."
        ),
        "data_url": "https://cds.climate.copernicus.eu/datasets/reanalysis-oras5",
        "data_label": "Climate Data Store — ORAS5 catalog",
        "doc_url": "https://www.ecmwf.int/en/elibrary/81235-ocean5-ecmwf-ocean-reanalysisanalysis-system-and-its-real-time-analysis-component",
        "doc_label": "ECMWF — OCEAN5 reanalysis system paper",
    },
    {
        "name": "Met Office EN4",
        "status": "Confirmed missing (Hadley / EN4 / ENACT keyword probes all 0 real hits).",
        "why_hhp": (
            "Gridded in-situ-OI T/S, the canonical \"Argo climatology\" — needed for the seasonal "
            "validation demo (winter / summer OHC means)."
        ),
        "data_url": "https://www.metoffice.gov.uk/hadobs/en4/download.html",
        "data_label": "UK Met Office — EN4 download page",
        "doc_url": "https://www.metoffice.gov.uk/hadobs/en4/documents.html",
        "doc_label": "EN4 documentation portal",
    },
    {
        "name": "ISAS (LOPS / Ifremer)",
        "status": "Confirmed missing.",
        "why_hhp": (
            "Independent European Argo-OI gridded T/S, ½° monthly with MLD diagnosed. "
            "Cross-product validation against EN4 and Roemmich-Gilson."
        ),
        "data_url": "https://www.seanoe.org/data/00412/52367/",
        "data_label": "SEANOE — ISAS dataset DOI",
        "doc_url": "https://www.coriolis.eu.org/Data-Products/Products/ISAS-Optimal-Analysis-of-temperature-and-salinity",
        "doc_label": "Coriolis — ISAS product page",
    },
    {
        "name": "Roemmich-Gilson Argo Climatology",
        "status": "Confirmed missing across 7 keyword strategies and full APDRC inventory.",
        "why_hhp": (
            "Reference Argo-only gridded T/S climatology (1° monthly). Pre-computed gridded "
            "fields solve the \"interpolate Argo to a uniform grid\" step Gregg specifically asked for."
        ),
        "data_url": "https://sio-argo.ucsd.edu/RG_Climatology.html",
        "data_label": "Scripps SIO — RG Argo Climatology",
        "doc_url": "https://doi.org/10.1016/j.pocean.2009.03.004",
        "doc_label": "Roemmich & Gilson 2009 — original paper (DOI)",
    },
    {
        "name": "NRL HYCOM 2019-present",
        "status": (
            "AQUAVIEW UAF has 31 HYCOM items but only the 2016-2018 GLBu0.08 expt 91.2 reanalysis."
        ),
        "why_hhp": (
            "Operational HYCOM updates regularly (now ESPC-D v02). Adding the recent stream gives "
            "an independent ocean model for cross-model validation."
        ),
        "data_url": "https://www.hycom.org/dataserver/espc-d-v02/global-analysis",
        "data_label": "HYCOM data server — ESPC-D v02 global analysis",
        "doc_url": "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020EA001199",
        "doc_label": "Barton et al. 2021 — Navy ESPC system paper",
    },
    {
        "name": "JMA TCHP for Western Pacific",
        "status": "Confirmed 0 hits.",
        "why_hhp": (
            "Out-of-basin sanity check. The only non-US operational TCHP equivalent currently published, "
            "covering the most active typhoon basin. Useful for demonstrating method transfer."
        ),
        "data_url": "https://ds.data.jma.go.jp/gmd/goos/data/database.html",
        "data_label": "JMA NEAR-GOOS database (RSMC Tokyo)",
        "doc_url": "https://www.data.jma.go.jp/gmd/kaiyou/english/tcp/tcp.html",
        "doc_label": "JMA Tropical Cyclone Heat Potential page",
    },
]


def build_table(headers: list[str], rows: list[dict], col_widths: list[float], header_row: list[str]) -> Table:
    body = [[Paragraph(f"<b>{h}</b>", style_cell_bold) for h in headers]]
    for r in rows:
        body.append([
            cell(f"<b>{r['name']}</b>", style_cell_bold),
            cell(r.get("why_missing") or r.get("status", "")),
            cell(r["why_hhp"]),
            link_cell(r["data_url"], r["data_label"]),
            link_cell(r["doc_url"], r["doc_label"]),
        ])
    t = Table(body, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING",    (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
        ("TOPPADDING",    (0, 1), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f1f5f9"), colors.white]),
        ("LINEABOVE", (0, 1), (-1, 1), 0.6, colors.HexColor("#94a3b8")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#94a3b8")),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#e2e8f0")),
    ]))
    return t


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=landscape(letter),
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
        title="AQUAVIEW Indexing Request — HHP Prediction",
        author="Suramya Angdembay (USM)",
    )

    flow: list = header_flowables()

    # Tier 1
    flow.append(Paragraph("Tier 1 — Real gaps with HHP impact", style_h2))
    flow.append(Paragraph(
        "Highest-priority missing items for the active HHP pipeline. Items rank by how blocked the "
        "pipeline is without them.",
        style_caption,
    ))
    col_widths_t1 = [1.55 * inch, 2.30 * inch, 2.55 * inch, 2.20 * inch, 2.00 * inch]
    flow.append(build_table(TIER1_HEADER, TIER1_ROWS, col_widths_t1, TIER1_HEADER))

    flow.append(Spacer(1, 0.20 * inch))

    # Tier 2
    flow.append(Paragraph("Tier 2 — Climate baselines and out-of-basin comparators", style_h2))
    flow.append(Paragraph(
        "Lower urgency for the active demo but high value once the project extends to seasonal "
        "validation, multi-storm benchmarking, or cross-model checks.",
        style_caption,
    ))
    col_widths_t2 = [1.55 * inch, 2.30 * inch, 2.55 * inch, 2.20 * inch, 2.00 * inch]
    flow.append(build_table(TIER2_HEADER, TIER2_ROWS, col_widths_t2, TIER2_HEADER))

    flow.append(Spacer(1, 0.18 * inch))

    flow.append(Paragraph("Distribution mechanism note", style_h2))
    flow.append(Paragraph(
        "<b>None of these datasets are natively distributed via ERDDAP.</b> AQUAVIEW already runs an "
        "ERDDAP scraper (most of NOAA_AOML_HDB was ingested that way), but the missing items above sit "
        "on different protocols:",
        style_body,
    ))
    proto_rows = [
        ["CMEMS GLORYS / GLO12",
         "copernicus-marine Python client (CMEMS account required)",
         "Lowest-effort: AOML already runs the CMEMS pipeline for salinity_model_g; adding T and u/v variables to the existing fetch is incremental work."],
        ["ECMWF ORAS5",
         "cdsapi Python client (CDS account required)",
         "New ingestion path. Monthly cadence, well-documented API."],
        ["Met Office EN4",
         "Plain HTTPS / FTP of NetCDF files",
         "Anonymous; simple requests-based fetcher."],
        ["ISAS (Ifremer)",
         "SEANOE DOI + Coriolis FTP",
         "Anonymous; per-version DOI updates."],
        ["Roemmich-Gilson Argo Climatology",
         "Plain HTTPS from sio-argo.ucsd.edu",
         "Anonymous; small NetCDF files; updated quarterly."],
        ["NRL HYCOM ESPC-D v02",
         "THREDDS catalog at tds.hycom.org",
         "Existing HYCOM-on-UAF pipeline could be extended."],
        ["NCEP RTOFS 3D",
         "AWS S3 bucket noaa-nws-rtofs-pds + NOMADS THREDDS",
         "Already-mirrored data; only the STAC indexing is missing."],
        ["JMA TCHP",
         "JMA NEAR-GOOS HTTP",
         "Lower-priority; coordinate with JMA on access if pursued."],
    ]
    flow.append(build_proto_table(proto_rows))

    flow.append(Spacer(1, 0.18 * inch))
    flow.append(Paragraph("Methodological notes for the AQUAVIEW catalog team", style_h2))
    flow.append(Paragraph(
        "Two systemic findings surfaced during the verification probes:",
        style_body,
    ))
    flow.append(Paragraph(
        "<b>1. q-search misses items whose titles do not carry common acronyms.</b> AOML AXBT data "
        "are filed as \"Expandable Bathythermograph\"; q=\"AXBT\" inside NOAA_AOML_HDB returns 0. "
        "Adding common acronyms to <font face='Courier' size='8'>aquaview:keywords</font> would "
        "resolve this.",
        style_body,
    ))
    flow.append(Paragraph(
        "<b>2. The aggregate endpoint's multi-word q parameter returns 0 even when search_datasets "
        "returns thousands of matches.</b> Aggregate q=\"ocean heat content\" returns 0; "
        "search_datasets q=\"ocean heat content\" returns 247,246. Single-word q on aggregate works "
        "fine. Recommend either fixing tokenization or documenting the restriction.",
        style_body,
    ))

    flow.append(Spacer(1, 0.18 * inch))
    flow.append(Paragraph(
        f"Generated {date.today().isoformat()} — Suramya Angdembay · "
        f"Institute for Advanced Analytics and Society, USM · "
        f"part of the {link('https://aquaview.org', 'AQUAVIEW')} ecosystem.",
        style_caption,
    ))

    doc.build(flow)
    print(f"Wrote {OUT}")


def build_proto_table(rows: list[list[str]]) -> Table:
    headers = [
        Paragraph("<b>Source</b>", style_cell_bold),
        Paragraph("<b>Distribution mechanism</b>", style_cell_bold),
        Paragraph("<b>Indexing complexity for AQUAVIEW</b>", style_cell_bold),
    ]
    body = [headers]
    for r in rows:
        body.append([cell(r[0], style_cell_bold), cell(r[1]), cell(r[2])])
    t = Table(body, colWidths=[2.0 * inch, 3.2 * inch, 5.4 * inch], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING",    (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
        ("TOPPADDING",    (0, 1), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f1f5f9"), colors.white]),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#e2e8f0")),
    ]))
    return t


if __name__ == "__main__":
    main()
