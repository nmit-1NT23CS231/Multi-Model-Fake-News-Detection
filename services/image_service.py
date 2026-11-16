"""
Image model: AI-generated / fake image detector (small Keras CNN).

Loading strategy
----------------
1. Preferred: models/image/fake_image_meta.json + fake_image_weights.npz
   (portable; works with any NumPy version).
2. Fallback:  models/image/fake_image_detector.pkl  (the original bundle; only
   readable if your NumPy version is compatible with the one that wrote it).
See scripts/convert_image_model.py for how the .npz/.json pair is produced.

Pre-processing is identical to the original vscode_inference.py:
RGB -> resize to (img_size, img_size) -> divide by 255 -> add batch dimension.
"""
import base64
import io
import json
import os
import pickle
import threading
from pathlib import Path

# Keep TensorFlow's console output quiet. Must be set before TF is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "image"
META_PATH = MODEL_DIR / "fake_image_meta.json"
WEIGHTS_PATH = MODEL_DIR / "fake_image_weights.npz"
PKL_PATH = MODEL_DIR / "fake_image_detector.pkl"

# Distance from 0.5 at which a prediction counts as "High" confidence
# (0.30 -> probability <= 20% or >= 80%). Same default as the original script.
CONFIDENCE_THRESHOLD = 0.30

ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}
Image.MAX_IMAGE_PIXELS = 60_000_000  # reject decompression bombs


class ImageService:
    key = "image"

    def __init__(self):
        self.model = None
        self.img_size = 32
        self.class_map = {0: "real", 1: "fake"}
        self.error = None
        self._lock = threading.Lock()   # Keras predict() is not guaranteed thread-safe

    # ------------------------------------------------------------------ load
    def load(self):
        from tensorflow.keras.models import model_from_json  # imported lazily

        if META_PATH.exists() and WEIGHTS_PATH.exists():
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            with np.load(WEIGHTS_PATH) as z:
                weights = [z[k] for k in sorted(z.files)]
            architecture = meta["architecture_json"]
            self.img_size = int(meta["img_size"])
            self.class_map = {int(k): v for k, v in meta["class_map"].items()}
        elif PKL_PATH.exists():
            with open(PKL_PATH, "rb") as f:
                bundle = pickle.load(f)
            weights = bundle["weights"]
            architecture = bundle["architecture_json"]
            self.img_size = int(bundle["img_size"])
            self.class_map = {int(k): v for k, v in bundle["class_map"].items()}
        else:
            raise FileNotFoundError(f"No image model files found in {MODEL_DIR}")

        model = model_from_json(architecture)
        model.set_weights(weights)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.model = model
        # Warm-up so the first real request isn't slow.
        self.model.predict(np.zeros((1, self.img_size, self.img_size, 3), np.float32), verbose=0)

    @property
    def ready(self):
        return self.model is not None

    @property
    def summary(self):
        return f"Convolutional network \u00b7 reads images at {self.img_size}\u00d7{self.img_size} pixels"

    # --------------------------------------------------------------- predict
    def predict(self, data: bytes) -> dict:
        try:
            img = Image.open(io.BytesIO(data))
            fmt = img.format
            img.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise ValueError("That file could not be read as an image. "
                             "Upload a JPG, PNG, BMP or WebP file.")
        if fmt not in ALLOWED_FORMATS:
            raise ValueError(f"{fmt or 'This'} images are not supported. "
                             "Upload a JPG, PNG, BMP or WebP file.")

        width, height = img.size
        small = img.convert("RGB").resize((self.img_size, self.img_size))
        x = np.expand_dims(np.asarray(small, dtype=np.float32) / 255.0, axis=0)

        with self._lock:
            prob_fake = float(self.model.predict(x, verbose=0)[0][0])

        label = self.class_map[1] if prob_fake >= 0.5 else self.class_map[0]
        certainty = abs(prob_fake - 0.5)
        confidence = "High" if certainty >= CONFIDENCE_THRESHOLD else "Low"

        buf = io.BytesIO()
        small.save(buf, format="PNG")
        model_view = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "classification": label.upper(),                    # FAKE | REAL
            "ai_probability_percent": round(prob_fake * 100, 2),
            "confidence": confidence,
            "width": width,
            "height": height,
            "format": fmt,
            "model_input_size": self.img_size,
            "model_view": model_view,                           # what the CNN actually sees
        }
