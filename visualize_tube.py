"""Entry point for Qualitative visualization for clinical review Runs fully locally.

Usage:
    python visualize_tube.py --centerline path/to/centerline.npy --output tube.obj
"""

import argparse

import numpy as np

from src.coronarycl.visualize import save_obj, sweep_tube


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--centerline", required=True,
                        help="(N,4) (x,y,z,radius) .npy")
    parser.add_argument("--output", default="tube.obj")
    parser.add_argument("--n-circle-pts", type=int, default=16)
    args = parser.parse_args()

    centerline = np.load(args.centerline)
    verts, faces = sweep_tube(centerline, n_circle_pts=args.n_circle_pts)
    save_obj(verts, faces, args.output)
    print(f"Wrote {args.output}: {len(verts)} verts, {len(faces)} faces")


if __name__ == "__main__":
    main()
