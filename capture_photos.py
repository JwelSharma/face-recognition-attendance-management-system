"""
🎥 540x720 NATURAL FACE CAPTURE - JWEL AMS (Aligned with system.py)
✅ Single‑person check ✅ High‑quality crops ✅ buffalo_l encoder ready
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import shutil
import sys
from pathlib import Path

import cv2
from insightface.app import FaceAnalysis

from utils import load_stream_url



print("🎥 540x720 NATURAL FACE CAPTURE - JWEL AMS")
print("📸 's'=SAVE → 'q'=DONE → 'quit'=EXIT")


cli_person_name = sys.argv[1].strip() if len(sys.argv) > 1 else ""



# Env (same style as system.py)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|stimeout;5000000"
)


# InsightFace buffalo_l (same as system.py)
print("\n🔥 LOADING buffalo_l GPU model...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 480), det_thresh=0.65)  # slightly stricter


dataset_path = Path("dataset")
dataset_path.mkdir(parents=True, exist_ok=True)


while True:
    if cli_person_name:
        person_name = cli_person_name
        cli_person_name = ""
    else:
        person_name = input("\n👤 Person name (quit=exit): ").strip()
    if person_name.lower() in ["quit", "exit", "q"]:
        break

    if not person_name:
        print("❌ Enter a valid name!")
        continue

    safe_name = person_name.replace(" ", "_").replace("-", "_")
    person_path = dataset_path / safe_name

    if person_path.exists():
        shutil.rmtree(person_path)
    person_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 FRESH dataset/{safe_name}/")

    # SAME CAMERA SETUP AS system.py
    urls = load_stream_url()

    cap = None
    for url in urls:
        print(f"🔄 Trying {url}...")
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 25)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))

        # 10‑frame warmup (same as system.py)
        for i in range(10):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"✅ CONNECTED! {url} → {frame.shape}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                break
        else:
            cap.release()
            continue
        break

    if cap is None or not cap.isOpened():
        print("❌ All streams failed - restart IP Webcam")
        sys.exit(1)

    window_name = f"📸 {safe_name.upper()} 540x720"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 540, 720)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    photo_count = 0
    total_saved = 0
    print(f"\n🎯 Goal: 60+ photos | S=SAVE | Q=DONE (single‑face only)")


    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        h, w = display.shape[:2]

        # Detect faces (same buffalo_l pipeline as system.py)
        faces = app.get(frame)
        face_ok = False

        if len(faces) == 1:
            face = faces[0]
            if face.det_score >= 0.75:  # stricter than runtime
                x1, y1, x2, y2 = [int(c) for c in face.bbox]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display,
                    f"{person_name} {face.det_score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                face_ok = True
        elif len(faces) == 0:
            cv2.putText(
                display,
                "⚠️ NO FACE",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                display,
                "⚠️ MULTIPLE FACES",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Status bar
        cv2.rectangle(display, (10, h - 60), (w - 10, h - 10), (0, 255, 0), 2)
        cv2.putText(
            display,
            f"👤 {person_name} ({total_saved}/60+)",
            (15, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display,
            "'S'=SAVE | 'Q'=DONE | ESC=ABORT",
            (15, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            if not face_ok:
                print("❌ SKIP: No / bad face detected")
                continue

            # Save aligned face crop (not full frame)
            face = faces[0]
            x1, y1, x2, y2 = [int(c) for c in face.bbox]
            margin = 10
            y1_margin = max(0, y1 - margin)
            y2_margin = min(h, y2 + margin)
            x1_margin = max(0, x1 - margin)
            x2_margin = min(w, x2 + margin)

            crop = frame[y1_margin:y2_margin, x1_margin:x2_margin]
            if min(crop.shape[:2]) < 50:
                print("❌ SKIP: too small crop")
                continue

            filename = person_path / f"{safe_name}_{photo_count:03d}.jpg"
            cv2.imwrite(str(filename), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            photo_count += 1
            total_saved = photo_count
            print(f"✅ {person_name} #{photo_count} (aligned face crop)")

        elif key == ord("q"):
            print(f"🎉 {person_name}: {photo_count} aligned 540x720 face crops!")
            break

        elif key == 27:  # ESC
            print(f"⏹️ Aborted: {photo_count} photos")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"📁 dataset/{safe_name}/{photo_count} SAVED!")


print("\n🏆 540x720 NATURAL CAPTURE READY!")