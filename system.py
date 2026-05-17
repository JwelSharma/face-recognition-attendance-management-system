#!/usr/bin/env python3
"""
🚀 AMS GPU-ONLY PRODUCTION v4.9
✅ CUDA
✅ IDENTICAL CSV + GOOGLE SHEETS (8 COLUMNS)
✅ OAUTH2 SHEETS
✅ STATUS JSON
✅ CAMERA AUTO-RECONNECT
✅ DAILY AUTO-ROLLOVER
✅ ROW_ID DUPLICATE PROTECTION
✅ SESSION_DURATION_MIN AUTO-UPDATE
✅ SAFE BATCH SHEETS + RETRY
✅ SETTINGS MATCH ADMIN_UI
✅ CLEAN EXIT
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|stimeout;100000|buffer_size;50000|probesize;50000|"
    "analyzeduration;100000|flush_packets;1"
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "error"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false;qt5ct.warning=false"

import cv2
import csv
import json
import time
import atexit
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort

from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import normalize
from insightface.app import FaceAnalysis
warnings.filterwarnings("ignore")

from utils import load_stream_url
urls = load_stream_url()

ROOT = Path(__file__).resolve().parent
SETTINGS_FILE = ROOT / "settings.json"

def load_stream_url():
    default_urls = [
        "http://10.227.221.**:8080/video",
        "http://10.227.221.**:8080/live", 
        "http://10.227.221.**:8080/video_feed"
    ]
    try:
        if SETTINGS_FILE.exists():
            settings = json.loads(SETTINGS_FILE.read_text())
            primary = settings.get("stream_url", default_urls[0])  # ← FIXED
            return [primary] + [u for u in default_urls if u != primary]
    except:
        pass
    return default_urls


import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

def safe_print(*args, **kwargs):
    text = " ".join(str(a) for a in args)
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="ignore").decode("ascii"), **kwargs)

# =========================
# GOOGLE SHEETS
# =========================
GOOGLE_SHEETS_ENABLED = False
SHEET_GC = None
SHEETS_LIVE = False
SHEET_URL = "https://docs.google.com/spreadsheets/d/1C1V4Rl4WjzfIxYU0jT4IokiWUq8Qrpc_WM54-******/edit"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file"
]

try:
    import gspread
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.auth.exceptions import RefreshError
    import pickle as pickle_module
    GOOGLE_SHEETS_ENABLED = True
    print("Google Sheets OAuth2 LOADED")        #✅ 
except ImportError:
    print("⚠️ Install: pip install gspread google-auth-oauthlib google-auth-httplib2")

# =========================
# PATHS
# =========================
ROOT = Path(".")
ATTENDANCE_DIR = ROOT / "Attendance"
ATTENDANCE_DIR.mkdir(exist_ok=True)

STATUS_FILE = ROOT / "system_status.json"
TOKEN_PATH = ROOT / "token.pickle"
CREDS_PATH = ROOT / "credentials.json"
SETTINGS_FILE = ROOT / "settings.json"
ENCODINGS_FILE = ROOT / "encodings.pickle"

SYSTEM_START_TS = time.time()





# =========================
# CONSTANTS / HEADERS
# =========================
ATTENDANCE_HEADERS = [
    "row_id",
    "timestamp",
    "date",
    "time",
    "name",
    "confidence",
    "session_duration_min",
    "worked_outside"
]

DEFAULT_SETTINGS = {
    # Admin UI compatible keys
    "recognition_threshold": 75.0,     # percent
    "cooldown_seconds": 300,             # admin_ui label says seconds
    "min_images_per_person": 3,
    "stream_url": "http://10.227.221.**:8080/video",

    # Internal advanced runtime keys
    "sheet_sync_interval_sec": 20,
    "max_faces_per_frame": 8,
    "process_every_n": 2,
    "status_write_interval_sec": 3,
    "reconnect_wait_sec": 2,
    "sheet_retry_limit": 3,
    "sheet_batch_size": 20,
    "det_size_w": 640,
    "det_size_h": 480,
    "det_thresh": 0.55,
    "match_threshold": 0.60,          # legacy/fallback
    "log_threshold": 0.75,            # legacy/fallback
    #"session_cooldown_min": max(1, int(float(SETTINGS.get("cooldown_minutes", 5))))         
}

# =========================
# GLOBALS
# =========================
cap = None
current_url = ""
camera_urls = []
frame = None
fps = 0.0
total_matches = 0
total_attendance_logs = 0
last_status_write = 0
last_recognized_name = ""
last_recognized_conf = ""
known_names = []
known_encodings = None
window_name = "AMS Live Feed"

sheet_buffer = []
last_sheet_sync = 0
sheet_failed_attempts = 0

last_log_time = {}
last_detections = []

logged_row_ids = set()
sheet_synced_row_ids = set()

current_day_str = datetime.now().strftime("%Y-%m-%d")
DAILY_FILE = ATTENDANCE_DIR / f"{current_day_str}.csv"
MAIN_FILE = ATTENDANCE_DIR / "attendance.csv"

settings = DEFAULT_SETTINGS.copy()

# =========================
# SETTINGS
# =========================
def load_settings():
    global settings

    admin_defaults = {
        "recognition_threshold": 75.0,
        "cooldown_seconds": 300,  # Matches admin_ui slider (5 minutes)
        "min_images_per_person": 3,
        "stream_url": "http://10.227.221.**:8080/video"  # underscore!
    }

    if not SETTINGS_FILE.exists():
        base = DEFAULT_SETTINGS.copy()
        base.update(admin_defaults)
        SETTINGS_FILE.write_text(json.dumps(base, indent=2), encoding="utf-8")
        settings = base
        print("✅ Created settings.json")
        return

    try:
        raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        raw = raw if isinstance(raw, dict) else {}

        merged = DEFAULT_SETTINGS.copy()
        merged.update(raw)

        # Backward compatibility → admin_ui.py keys (underscore format)
        if "recognitionthreshold" in raw:
            merged["recognition_threshold"] = float(raw.get("recognitionthreshold", 75.0))
        if "cooldownseconds" in raw or "cooldown_minutes" in raw:
            merged["cooldown_seconds"] = max(1, int(raw.get("cooldownseconds", 300) or raw.get("cooldown_minutes", 5) * 60))
        if "minimagesperperson" in raw:
            merged["min_images_per_person"] = int(raw.get("minimagesperperson", 3))
        if "streamurl" in raw:  # ← THIS FIX
            merged["stream_url"] = str(raw.get("streamurl", "")).strip()

        settings = merged
        print("✅ Loaded settings.json")
        print(f"   Stream: {settings.get('stream_url', 'N/A')}")
        print(f"   Threshold: {settings.get('recognition_threshold', 'N/A')}%")

    except Exception as e:
        print(f"⚠️ Failed to load settings.json: {e}")
        settings = DEFAULT_SETTINGS.copy()

def get_match_threshold():
    try:
        if "match_threshold" in settings:
            return float(settings.get("match_threshold", 0.60))
    except Exception:
        pass
    return 0.60

def get_log_threshold():
    # Admin UI stores threshold as percentage, e.g. 75 => 0.75
    try:
        return float(settings.get("recognition_threshold", 75.0)) / 100.0
    except Exception:
        pass
    try:
        return float(settings.get("log_threshold", 0.75))
    except Exception:
        return 0.75

def get_cooldown_seconds():
    try:
        return max(1, int(settings.get("cooldown_seconds", 300)))
    except Exception:
        pass
    try:
        return max(1, int(float(settings.get("session_cooldown_min", 5)) * 60))
    except Exception:
        return 300

def get_stream_url():
    return str(settings.get("stream_url", "http://10.227.221.**:8080/video")).strip()

# =========================
# DATE / FILE HELPERS
# =========================
def refresh_day_files():
    global current_day_str, DAILY_FILE, MAIN_FILE
    current_day_str = datetime.now().strftime("%Y-%m-%d")
    DAILY_FILE = ATTENDANCE_DIR / f"{current_day_str}.csv"
    MAIN_FILE = ATTENDANCE_DIR / "attendance.csv"

def ensure_daily_rollover():
    global current_day_str
    new_day = datetime.now().strftime("%Y-%m-%d")
    if new_day != current_day_str:
        print(f"📅 Daily rollover: {current_day_str} -> {new_day}")
        current_day_str = new_day
        refresh_day_files()
        setup_csv_headers()

# =========================
# STATUS JSON
# =========================
def write_system_status(payload: dict):
    try:
        payload["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        STATUS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"⚠️ Status write failed: {e}")

def build_status(system_mode="starting", error_msg="", frame_shape=None):
    uptime_sec = int(time.time() - SYSTEM_START_TS)
    return {
        "system_mode": system_mode,
        "camera_connected": bool(cap is not None and cap.isOpened()),
        "stream_url": current_url,
        "configured_stream_url": get_stream_url(),
        "frame_shape": list(frame_shape) if frame_shape is not None else None,
        "gpu_enabled": bool(torch.cuda.is_available()),
        "google_sheets_loaded": GOOGLE_SHEETS_ENABLED,
        "google_sheets_live": SHEETS_LIVE,
        "sheet_buffer_size": len(sheet_buffer),
        "face_db_count": len(known_names) if known_names else 0,
        "matches": total_matches,
        "logs_written": total_attendance_logs,
        "fps": round(float(fps), 2) if fps else 0.0,
        "last_recognized": last_recognized_name,
        "last_confidence": last_recognized_conf,
        "uptime_sec": uptime_sec,
        "last_frame_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "recognition_threshold_percent": float(settings.get("recognition_threshold", 75.0)),
        "cooldown_seconds": get_cooldown_seconds(),
        "min_images_per_person": int(settings.get("min_images_per_person", 3)),
        "error_msg": error_msg
    }

# =========================
# CSV HELPERS
# =========================
def clean_existing_csv(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return

    cleaned_rows = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if not any(str(cell).strip() for cell in row):
                    continue
                cleaned_rows.append(row)

        if not cleaned_rows:
            cleaned_rows = [ATTENDANCE_HEADERS]

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(cleaned_rows)

    except Exception as e:
        print(f"⚠️ Could not clean {path}: {e}")

def setup_csv_headers():
    for file_path in [DAILY_FILE, MAIN_FILE]:
        if not file_path.exists() or file_path.stat().st_size == 0:
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(ATTENDANCE_HEADERS)
            print(f"✅ Created {file_path}")
        else:
            clean_existing_csv(file_path)

def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ CSV read failed for {path}: {e}")
    return pd.DataFrame(columns=ATTENDANCE_HEADERS)

def ensure_attendance_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ATTENDANCE_HEADERS:
        if col not in df.columns:
            df[col] = ""
    return df[ATTENDANCE_HEADERS]

def load_existing_row_ids():
    global logged_row_ids
    df = safe_read_csv(MAIN_FILE)
    if "row_id" in df.columns:
        ids = (
            df["row_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        logged_row_ids = set(x for x in ids if x)
    else:
        logged_row_ids = set()
    print(f"✅ Loaded {len(logged_row_ids)} existing row_ids")

def append_csv_row(row_dict):
    row = [str(row_dict.get(col, "")).strip() for col in ATTENDANCE_HEADERS]

    if not any(row):
        return

    for file_path in [DAILY_FILE, MAIN_FILE]:
        with open(file_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(row)
            f.flush()

def update_previous_session_duration(person_name: str, current_ts: datetime):
    try:
        df = safe_read_csv(MAIN_FILE)
        if df.empty:
            return

        df = ensure_attendance_columns(df)

        if "name" not in df.columns or "timestamp" not in df.columns:
            return

        person_mask = df["name"].fillna("").astype(str).str.strip() == person_name
        person_df = df[person_mask].copy()

        if person_df.empty:
            return

        person_df["parsed_ts"] = pd.to_datetime(person_df["timestamp"], errors="coerce")
        person_df = person_df.dropna(subset=["parsed_ts"]).sort_values("parsed_ts")

        if person_df.empty:
            return

        last_idx = person_df.index[-1]
        last_ts = person_df.loc[last_idx, "parsed_ts"]

        old_duration = str(df.loc[last_idx, "session_duration_min"]).strip()
        if old_duration not in ["", "0", "0.0", "nan", "None"]:
            return

        duration_min = max(0, round((current_ts - last_ts).total_seconds() / 60.0, 2))

        if duration_min <= 0:
            return

        df.loc[last_idx, "session_duration_min"] = duration_min
        df = ensure_attendance_columns(df)
        df.to_csv(MAIN_FILE, index=False)

        day_str = pd.to_datetime(df.loc[last_idx, "timestamp"], errors="coerce")
        if pd.notna(day_str):
            prev_daily = ATTENDANCE_DIR / f"{day_str.strftime('%Y-%m-%d')}.csv"
            if prev_daily.exists():
                ddf = safe_read_csv(prev_daily)
                ddf = ensure_attendance_columns(ddf)
                rid = str(df.loc[last_idx, "row_id"]).strip()
                if "row_id" in ddf.columns and rid:
                    match = ddf["row_id"].fillna("").astype(str).str.strip() == rid
                    if match.any():
                        ddf.loc[match, "session_duration_min"] = duration_min
                        ddf = ensure_attendance_columns(ddf)
                        ddf.to_csv(prev_daily, index=False)

        print(f"⏱️ Updated previous session for {person_name}: {duration_min} min")

    except Exception as e:
        print(f"⚠️ Failed updating session_duration_min: {e}")

# =========================
# GOOGLE SHEETS HELPERS
# =========================
def build_sheet_row(row_dict):
    return [row_dict.get(col, "") for col in ATTENDANCE_HEADERS]

def setup_google_sheets():
    global SHEET_GC, SHEETS_LIVE

    if not GOOGLE_SHEETS_ENABLED:
        return False

    try:
        creds = None

        if TOKEN_PATH.exists():
            with open(TOKEN_PATH, "rb") as token:
                creds = pickle_module.load(token)

        if creds:
            try:
                if not creds.valid:
                    if creds.expired and creds.refresh_token:
                        print("🔄 Refreshing Google token...")
                        creds.refresh(Request())
                    else:
                        creds = None
            except RefreshError as e:
                if "invalid_grant" in str(e).lower():
                    print("🔓 Token expired/revoked. Re-authenticating...")
                    try:
                        TOKEN_PATH.unlink(missing_ok=True)
                    except Exception:
                        pass
                    creds = None
                else:
                    raise

        if not creds:
            if not CREDS_PATH.exists():
                raise FileNotFoundError("credentials.json not found")
            print("🔐 Opening browser for Google login...")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
            with open(TOKEN_PATH, "wb") as token:
                pickle_module.dump(creds, token)

        gc = gspread.authorize(creds)
        SHEET_GC = gc.open_by_url(SHEET_URL)
        sheet = SHEET_GC.sheet1

        all_values = sheet.get_all_values()
        headers = all_values[0] if all_values else []

        if headers != ATTENDANCE_HEADERS:
            print("🛠️ Fixing Google Sheet headers to match CSV...")
            sheet.clear()
            sheet.append_row(ATTENDANCE_HEADERS)

        print(f"✅ Google Sheets LIVE: {SHEET_GC.title}")
        SHEETS_LIVE = True
        return True

    except Exception as e:
        print(f"⚠️ Sheets setup failed: {e}")
        SHEETS_LIVE = False
        return False

def flush_sheet_buffer():
    global sheet_buffer, last_sheet_sync, sheet_failed_attempts, SHEETS_LIVE

    if not SHEET_GC or not sheet_buffer:
        return

    batch_size = int(settings.get("sheet_batch_size", 20))
    retry_limit = int(settings.get("sheet_retry_limit", 3))

    for attempt in range(1, retry_limit + 1):
        try:
            batch = sheet_buffer[:batch_size]
            SHEET_GC.sheet1.append_rows(batch, value_input_option="USER_ENTERED")
            synced_ids = [str(r[0]).strip() for r in batch if len(r) > 0 and str(r[0]).strip()]
            for rid in synced_ids:
                sheet_synced_row_ids.add(rid)

            print(f"☁️ SYNCED BATCH: {len(batch)} rows")
            sheet_buffer = sheet_buffer[len(batch):]
            last_sheet_sync = time.time()
            sheet_failed_attempts = 0
            SHEETS_LIVE = True

            if sheet_buffer:
                flush_sheet_buffer()
            return

        except Exception as e:
            sheet_failed_attempts += 1
            SHEETS_LIVE = False
            print(f"☁️ Sheets batch error attempt {attempt}/{retry_limit}: {e}")
            time.sleep(min(2 * attempt, 5))

    print("⚠️ Google Sheets sync failed after retries; rows kept in local buffer")

def queue_sheet_row(row_dict):
    row_id = str(row_dict.get("row_id", "")).strip()
    if not row_id:
        return

    if row_id in sheet_synced_row_ids:
        return

    for row in sheet_buffer:
        if len(row) > 0 and str(row[0]).strip() == row_id:
            return

    sheet_buffer.append(build_sheet_row(row_dict))

# =========================
# CAMERA HELPERS
# =========================
def connect_camera():
    global cap, current_url, frame

    safe_release_camera()

    for url in camera_urls:
        print(f"🔄 Trying {url}...")
        test_cap = cv2.VideoCapture(url)
        test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        test_cap.set(cv2.CAP_PROP_FPS, 25)

        connected = False
        temp_frame = None

        for _ in range(10):
            ret, temp_frame = test_cap.read()
            if ret and temp_frame is not None and temp_frame.size > 0:
                test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"✅ CONNECTED! {url} -> {temp_frame.shape}")
                connected = True
                break

        if connected:
            cap = test_cap
            current_url = url
            frame = temp_frame
            return True
        else:
            test_cap.release()

    cap = None
    current_url = ""
    return False

def auto_reconnect_camera():
    wait_sec = int(settings.get("reconnect_wait_sec", 2))
    print("📡 Camera lost. Attempting reconnect...")
    write_system_status(build_status(system_mode="reconnecting", frame_shape=None))
    time.sleep(wait_sec)

    ok = connect_camera()
    if ok:
        print("✅ Camera reconnected")
        write_system_status(build_status(
            system_mode="running",
            frame_shape=frame.shape if frame is not None else None
        ))
        return True

    print("❌ Camera reconnect failed")
    return False

# =========================
# WINDOW / CLEANUP
# =========================
def safe_release_camera():
    global cap
    try:
        if cap is not None:
            cap.release()
            cap = None
    except Exception:
        pass

def safe_cv2_cleanup():
    safe_release_camera()

    try:
        cv2.waitKey(50)
    except Exception:
        pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    try:
        cv2.waitKey(50)
    except Exception:
        pass

    time.sleep(0.1)

def cleanup():
    try:
        if SHEET_GC and sheet_buffer:
            flush_sheet_buffer()
    except Exception:
        pass

    try:
        write_system_status(build_status(
            system_mode="stopped",
            frame_shape=frame.shape if frame is not None else None
        ))
    except Exception:
        pass

    safe_cv2_cleanup()
    print("\n✅ CLEAN EXIT")

atexit.register(cleanup)

# =========================
# STARTUP
# =========================
load_settings()
refresh_day_files()
setup_csv_headers()
load_existing_row_ids()
setup_google_sheets()
write_system_status(build_status(system_mode="initializing"))

# =========================
# CUDA / DLL SETUP
# =========================
dll_paths = [r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"]
for dll_path in dll_paths:
    if os.path.exists(dll_path):
        try:
            os.add_dll_directory(dll_path)
        except Exception:
            pass

print(f"✅ PyTorch CUDA: {torch.cuda.is_available()}")
print(f"✅ CUDA providers: {ort.get_available_providers()}")

if "CUDAExecutionProvider" not in ort.get_available_providers():
    msg = "CUDAExecutionProvider not found"
    print(f"💥 {msg}")
    write_system_status(build_status(system_mode="error", error_msg=msg))
    raise SystemExit

# =========================
# CAMERA URLS
# =========================
primary_stream = get_stream_url()
camera_urls = [primary_stream] if primary_stream else []
# Add fallbacks only if primary fails
fallback_urls = [
    "http://10.227.221.**:8080/video",
    "http://10.227.221.**:8080/live", 
    "http://10.227.221.**:8080/video_feed"
]
camera_urls.extend(fallback_urls)

if not connect_camera():
    print("❌ All streams failed")
    write_system_status(build_status(system_mode="error", error_msg="All streams failed"))
    raise SystemExit

print("🚀 LIVE FEED READY!")
write_system_status(build_status(
    system_mode="camera_ready",
    frame_shape=frame.shape if frame is not None else None
))

# =========================
# GPU MODELS
# =========================
det_w = int(settings.get("det_size_w", 640))
det_h = int(settings.get("det_size_h", 480))
det_thresh = float(settings.get("det_thresh", 0.55))

print("\n🔥 LOADING GPU buffalo_l...")
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(det_w, det_h), det_thresh=det_thresh)

for task, model in app.models.items():
    if hasattr(model, "session"):
        print(f"✅ {task}: {model.session.get_providers()}")

test_frame = np.random.randint(0, 255, (det_h, det_w, 3), dtype=np.uint8)
_ = app.get(test_frame)
print("✅ GPU TEST PASSED!")
print("✅ ALL GPU MODELS LOADED!")
write_system_status(build_status(system_mode="models_loaded"))

# =========================
# FACE DATABASE
# =========================
try:
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    known_encodings = normalize(np.array(data["encodings"]), norm="l2", axis=1)
    known_names = data["names"]

    if len(known_encodings) == 0 or len(known_names) == 0:
        raise ValueError("Face database is empty")

    if len(known_encodings) != len(known_names):
        raise ValueError("Mismatch between encodings and names count")

    print(f"✅ Face DB: {len(known_names)} identities")
    print(f"✅ Admin setting min_images_per_person = {int(settings.get('min_images_per_person', 3))}")

except Exception as e:
    print(f"💥 Face DB load failed: {e}")
    write_system_status(build_status(system_mode="error", error_msg=f"Face DB load failed: {e}"))
    raise SystemExit

write_system_status(build_status(system_mode="face_db_loaded"))

# =========================
# DISPLAY WINDOW
# =========================
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(window_name, 1280, 720)
cv2.moveWindow(window_name, 50, 50)
first_frame = True

frame_count = 0
fps_start = time.time()

print("\n🏆 PRODUCTION LIVE v4.9 (ADMIN_UI MATCHED) - Press 'q' to quit")

# =========================
# MAIN LOOP
# =========================
try:
    while True:
        ensure_daily_rollover()

        ret, frame = cap.read() if cap is not None else (False, None)
        if not ret or frame is None:
            if not auto_reconnect_camera():
                time.sleep(0.5)
                continue
            ret, frame = cap.read() if cap is not None else (False, None)
            if not ret or frame is None:
                time.sleep(0.05)
                continue

        frame_count += 1
        detections = last_detections.copy()

        if frame_count % int(settings.get("process_every_n", 2)) == 0:
            detections = []
            frame_gpu = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)

            try:
                faces = app.get(frame_gpu)
            except Exception as e:
                print(f"⚠️ Face inference error: {e}")
                faces = []

            for face in faces[:int(settings.get("max_faces_per_frame", 8))]:
                try:
                    emb = face.normed_embedding
                    sims = np.dot(known_encodings, emb)
                    confidence = float(sims.max())
                    best_idx = int(np.argmax(sims))

                    match_threshold = get_match_threshold()
                    log_threshold = get_log_threshold()
                    cooldown_sec = get_cooldown_seconds()

                    person = known_names[best_idx] if confidence > match_threshold else "Unknown"
                    now = time.time()

                    if person != "Unknown":
                        last_recognized_name = person
                        last_recognized_conf = f"{confidence:.1%}"

                    if person != "Unknown" and confidence >= log_threshold:
                        elapsed_sec = now - last_log_time.get(person, 0)

                        if elapsed_sec >= cooldown_sec:
                            ts = datetime.now()

                            update_previous_session_duration(person, ts)

                            row_id = (
                                f"{ts.strftime('%Y%m%d_%H%M%S')}__"
                                f"{person.replace(' ', '_')}__"
                                f"{int(ts.microsecond / 1000)}"
                            )

                            row_dict = {
                                "row_id": row_id,
                                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                                "date": ts.strftime("%Y-%m-%d"),
                                "time": ts.strftime("%H:%M"),
                                "name": person,
                                "confidence": f"{confidence * 100:.1f}",
                                "session_duration_min": "",
                                "worked_outside": ""
                            }

                            if row_id not in logged_row_ids:
                                append_csv_row(row_dict)
                                logged_row_ids.add(row_id)

                                if SHEET_GC:
                                    queue_sheet_row(row_dict)

                                total_attendance_logs += 1
                                last_log_time[person] = now
                                print(
                                    f"📝 LOGGED: {person} {confidence:.1%} "
                                    f"-> CSV + SHEETS (8 cols) | "
                                    f"threshold={settings.get('recognition_threshold', 75.0)}% "
                                    f"| cooldown={cooldown_sec}s"
                                )
                            else:
                                print(f"⚠️ Duplicate prevented for row_id={row_id}")

                    h, w = frame.shape[:2]
                    scale_x = w / float(det_w)
                    scale_y = h / float(det_h)

                    x1 = max(0, int(face.bbox[0] * scale_x))
                    y1 = max(0, int(face.bbox[1] * scale_y))
                    x2 = min(w - 1, int(face.bbox[2] * scale_x))
                    y2 = min(h - 1, int(face.bbox[3] * scale_y))

                    color = (0, 255, 0) if confidence > match_threshold else (0, 0, 255)

                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "person": person,
                        "conf": confidence * 100,
                        "color": color
                    })

                    if confidence > match_threshold:
                        total_matches += 1

                except Exception as e:
                    print(f"⚠️ Detection processing error: {e}")
                    continue

            last_detections = detections

        display = frame.copy()
        fps = frame_count / max((time.time() - fps_start), 1e-6)
        uptime = (time.time() - SYSTEM_START_TS) / 60.0
        sheets_status = (
            f"SHEETS:LIVE BUF:{len(sheet_buffer)}"
            if SHEET_GC and SHEETS_LIVE else
            f"SHEETS:QUEUE BUF:{len(sheet_buffer)}"
            if SHEET_GC else
            "LOCAL"
        )

        overlay_text = (
            f"GPU AMS v4.9 | {fps:.1f} FPS | Matches:{total_matches} | "
            f"Logs:{total_attendance_logs} | {sheets_status} | "
            f"Thr:{float(settings.get('recognition_threshold', 75.0)):.0f}% | "
            f"CD:{get_cooldown_seconds()}s | Up:{uptime:.1f}m"
        )

        cv2.rectangle(display, (10, 10), (1240, 78), (15, 40, 120), -1)
        cv2.putText(
            display,
            overlay_text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(display, (x1, y1), (x2, y2), det["color"], 3)

            label = f"{det['person']} {det['conf']:.0f}%"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            y_label_top = max(y1 - label_h - 10, 0)
            cv2.rectangle(display, (x1, y_label_top), (x1 + label_w + 12, y1), det["color"], -1)
            cv2.putText(
                display,
                label,
                (x1 + 8, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        cv2.imshow(window_name, display)

        if first_frame:
            try:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            except cv2.error:
                pass
            first_frame = False

        now_ts = time.time()

        if SHEET_GC and sheet_buffer and (
            now_ts - last_sheet_sync >= int(settings.get("sheet_sync_interval_sec", 20))
            or len(sheet_buffer) >= int(settings.get("sheet_batch_size", 20))
        ):
            flush_sheet_buffer()

        if now_ts - last_status_write >= int(settings.get("status_write_interval_sec", 3)):
            write_system_status(build_status(
                system_mode="running",
                frame_shape=frame.shape if frame is not None else None
            ))
            last_status_write = now_ts

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("\n⌨️ Interrupted by user")

except Exception as e:
    print(f"\n💥 SYSTEM ERROR: {e}")
    write_system_status(build_status(
        system_mode="error",
        error_msg=str(e),
        frame_shape=frame.shape if frame is not None else None
    ))
    raise

finally:
    print("\n🏆 SESSION COMPLETE!")
    print(f"📊 FPS: {fps:.1f} | Matches: {total_matches} | Logs: {total_attendance_logs}")
    print(f"📁 Daily: {DAILY_FILE}")
    print(f"📁 Main: {MAIN_FILE}")
    if SHEET_GC:
        print(f"☁️ Pending Sheets rows before exit: {len(sheet_buffer)}")
    #cleanup()