"""
One-time converter: fake_image_detector.pkl  ->  portable .npz + .json

Why: the original .pkl is a plain pickle of raw NumPy arrays. A pickle written
with NumPy 2.x cannot be read by NumPy 1.x (and TensorFlow 2.15/2.16 require
NumPy 1.x). The .npz/.json pair has no such restriction.

Run it in an environment where the ORIGINAL pickle can be read (e.g. the
Colab/venv you trained in):

    python scripts/convert_image_model.py models/image/fake_image_detector.pkl

It writes fake_image_weights.npz and fake_image_meta.json next to the .pkl.
authen loads those two files first and only falls back to the .pkl if they
are missing.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python scripts/convert_image_model.py path/to/fake_image_detector.pkl")
    src = Path(sys.argv[1])
    with open(src, "rb") as f:
        bundle = pickle.load(f)

    out_dir = src.parent
    np.savez(out_dir / "fake_image_weights.npz",
             **{f"w{i:03d}": np.asarray(w) for i, w in enumerate(bundle["weights"])})
    meta = {
        "architecture_json": bundle["architecture_json"],
        "img_size": int(bundle["img_size"]),
        "class_map": {str(k): v for k, v in bundle["class_map"].items()},
        "preprocessing": bundle.get("preprocessing", "rescale_1_255"),
        "n_weights": len(bundle["weights"]),
    }
    (out_dir / "fake_image_meta.json").write_text(json.dumps(meta, indent=2))
    print("Wrote", out_dir / "fake_image_weights.npz", "and", out_dir / "fake_image_meta.json")


if __name__ == "__main__":
    main()
