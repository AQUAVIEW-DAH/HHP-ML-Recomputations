"""Canonical repository paths for HHP-Prediction."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DATASETS_DIR = ARTIFACTS_DIR / "datasets"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
BENCHMARK_REPORTS_DIR = REPORTS_DIR / "benchmarks"
SOURCE_REPORTS_DIR = REPORTS_DIR / "source_audits"
TRAINING_REPORTS_DIR = REPORTS_DIR / "training"
LAYERS_DIR = ARTIFACTS_DIR / "layers"

# External caches (disk-backed, outside repo)
RTOFS_CACHE_DIR = Path("/data/suramya/rtofs_time_matched")
ARGO_CACHE_DIR = Path("/data/suramya/argo_cache_hhp")
IBTRACS_CACHE_DIR = Path("/data/suramya/ibtracs_cache")

for d in (DATASETS_DIR, MODELS_DIR, BENCHMARK_REPORTS_DIR, SOURCE_REPORTS_DIR, TRAINING_REPORTS_DIR, LAYERS_DIR):
    d.mkdir(parents=True, exist_ok=True)
