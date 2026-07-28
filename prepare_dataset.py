"""Entry point for Step 1.3. Runs locally on M4, no GPU needed.

Assumes Steps 1.1 (centerlines) and 1.2 (DRR projections) are already
complete, and that data/splits/case_splits.json, normalization_stats.json,
and image_norm_stats.json already exist (produced during EDA/normalization
-- see notebooks/preprocessing.ipynb for the interactive derivation of
these stats).

Usage:
    python prepare_dataset.py
"""

from src.coronarycl.dataset import package_all

if __name__ == "__main__":
    package_all(
        centerline_dir="data/processed/centerlines",
        drr_dir="data/processed/DRR_Generation/Total",
        raw_dir="data/raw",
        splits_dir="data/splits",
        out_dir="data/processed/packaged",
    )
