#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JWEL AMS ENCODER
Top 10 embeddings by det_score per identity
Dashboard-safe version:
- UTF-8 safe logging
- No emoji in logs
- Safe subprocess/stdout behavior
- Better dataset checks
"""

import os
import sys
import cv2
import pickle
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
from sklearn.preprocessing import normalize
import logging
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# FORCE UTF-8 FOR DASHBOARD / SUBPROCESS SAFETY
# =========================================================
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset"
ENCODINGS_FILE = BASE_DIR / "encodings.pickle"
LOG_FILE = BASE_DIR / "encode_log.txt"

# =========================================================
# LOGGING SETUP
# =========================================================
if LOG_FILE.exists():
    try:
        LOG_FILE.unlink()
    except Exception:
        pass

logger = logging.getLogger("encode_faces")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info("encode_faces.py STARTED")
print("encode_faces.py running...")
print("JWEL AMS ENCODER")
print("Using buffalo_l GPU model")

# =========================================================
# CHECK DATASET
# =========================================================
logger.info(f"Dataset path: {DATASET_PATH}")
logger.info(f"Dataset exists: {DATASET_PATH.exists()}")

if not DATASET_PATH.exists():
    logger.error(f"Dataset folder not found: {DATASET_PATH}")
    raise SystemExit(1)

person_dirs = [d for d in DATASET_PATH.iterdir() if d.is_dir()]
logger.info(f"Found {len(person_dirs)} person folders")

if not person_dirs:
    logger.error("No person folders found in dataset")
    raise SystemExit(1)

# =========================================================
# FACE MODEL
# =========================================================
logger.info("Loading FaceAnalysis buffalo_l ...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 480), det_thresh=0.65)

logger.info("FaceAnalysis ready")

# =========================================================
# STORAGE
# =========================================================
identity_map = {}
rejected_map = {}

# =========================================================
# SCAN DATASET
# =========================================================
for person_dir in person_dirs:
    safe_name = person_dir.name.strip()
    logger.info(f"Processing person: {safe_name}")

    if safe_name not in identity_map:
        identity_map[safe_name] = []
        rejected_map[safe_name] = []

    img_paths = sorted(list(person_dir.glob("*.jpg")))
    logger.info(f"Found {len(img_paths)} JPG images in {safe_name}")

    if not img_paths:
        logger.warning(f"No JPG images found for {safe_name}")
        continue

    for img_path in img_paths:
        try:
            img_str = str(img_path.relative_to(BASE_DIR))
        except Exception:
            img_str = str(img_path)

        image = cv2.imread(str(img_path))

        if image is None or image.size == 0:
            logger.warning(f"SKIP bad image: {img_str}")
            rejected_map[safe_name].append((img_str, "bad_image"))
            continue

        try:
            faces = app.get(image)
        except Exception as e:
            logger.warning(f"SKIP inference error on {img_str}: {e}")
            rejected_map[safe_name].append((img_str, f"inference_error_{str(e)}"))
            continue

        if len(faces) != 1:
            logger.warning(f"SKIP {len(faces)} faces in {img_str}")
            rejected_map[safe_name].append((img_str, f"{len(faces)}_faces"))
            continue

        face = faces[0]

        if face.det_score < 0.70:
            logger.warning(f"SKIP low det_score={face.det_score:.2f} {img_str}")
            rejected_map[safe_name].append((img_str, f"low_det_score_{face.det_score:.2f}"))
            continue

        emb = face.normed_embedding
        identity_map[safe_name].append((img_str, emb, float(face.det_score)))
        logger.info(f"ACCEPT {img_str} det_score={face.det_score:.2f}")

# =========================================================
# BUILD PROTOTYPES
# =========================================================
def top_embeddings_by_det_score(img_emb_list, max_prototypes=10):
    sorted_samples = sorted(img_emb_list, key=lambda x: x[2], reverse=True)
    kept = sorted_samples[:max_prototypes]
    embs = [emb for (_, emb, _) in kept]
    return np.array(embs)

final_names = []
final_encodings = []

logger.info("Building prototypes ...")

for name, img_emb_list in identity_map.items():
    if not img_emb_list:
        logger.warning(f"No good samples for {name}")
        continue

    prototypes = top_embeddings_by_det_score(img_emb_list, max_prototypes=10)
    num_protos = len(prototypes)

    logger.info(f"{name}: kept {num_protos} prototypes (top by det_score)")

    final_encodings.extend(prototypes)
    final_names.extend([name] * num_protos)

# =========================================================
# SAVE ENCODINGS
# =========================================================
if final_encodings:
    logger.info("Normalizing embeddings ...")
    final_encodings = np.array(final_encodings)
    final_encodings = normalize(final_encodings, norm="l2", axis=1)

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(
            {
                "encodings": final_encodings,
                "names": final_names
            },
            f
        )

    logger.info(f"SAVED {len(final_encodings)} encodings for {len(set(final_names))} identities")
    print(f"Built {len(final_encodings)} encodings for {len(set(final_names))} identities")
    print(f"Saved: {ENCODINGS_FILE}")
else:
    logger.warning("NO VALID ENCODINGS GENERATED - check dataset/images")
    print("No valid encodings generated")
    raise SystemExit(2)

logger.info("encode_faces.py COMPLETED")
print("Done. Ready for system.py")