"""One-off dataset download script (not core library code -- lives in
scripts/, not src/coronarycl/, matching the dermai-explainability
convention).

TODO: ImageCAS is distributed via a Google Drive / Baidu link from the
official repo (github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-
and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT), which
generally requires accepting a data-use agreement rather than a
scripted anonymous download. Fill in your access method below once
you have it (gdown with your authorized file ID, a signed URL, etc.).

Usage:
    python scripts/download_imagecas.py --out data/raw
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "Fill in your authorized ImageCAS download method here. "
        "See DATASET.md for access details."
    )


if __name__ == "__main__":
    main()
