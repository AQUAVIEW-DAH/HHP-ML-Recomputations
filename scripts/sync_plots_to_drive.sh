#!/bin/bash
# Mirror the curated plot bundles to fixed Google Drive folders.
#
# Uses `rclone sync` (true mirror: updates files in place, deletes remote
# files that no longer exist locally) into stable folder names, so reruns
# update the same folders instead of creating dated copies. Version history
# of the tracked bundles lives in git, not in Drive.
#
# Drive root: the shared HHP-ML-features-plots folder.
set -euo pipefail

ROOT_ID="1vFB8vqNMuG9Hh_otFJim4gsug9cRyUsH"
OUT=/home/suramya/HHP-Prediction/OHC/output

sync_dir () {
    local src=$1
    local dest=$2
    echo "== ${src} -> HHP-plots/${dest}"
    rclone sync "${src}" "gdrive:HHP-plots/${dest}" \
        --drive-root-folder-id "${ROOT_ID}" \
        --exclude "*.parquet" \
        --stats-one-line
}

# NOTE: destinations must not be nested inside each other — `rclone sync`
# deletes remote files absent from its local source, so a nested folder
# would be wiped by its parent's sync.
sync_dir "${OUT}/density_scatter_diagnostics_2024_2025" "density-diagnostics"
sync_dir "${OUT}/presentation_spatial_ablation_2024_2025" "spatial-ablation"
sync_dir "${OUT}/ml_benchmarks/spatial_tiers" "spatial-ablation-tile20-holdout"
sync_dir "${OUT}/feature_diagnostics/hhp_feature_gallery_2024_2025_curated" "feature-gallery-curated"
sync_dir "${OUT}/box_stats_maps_2024_2025" "box-stats-maps"

echo "Done."
