import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import load_stream_url

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AMS Admin Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset"
ENCODE_LOG = ROOT / "encode_log.txt"
ENCODINGS_FILE = ROOT / "encodings.pickle"
ENCODE_SCRIPT = ROOT / "encode_faces.py"
SYSTEM_SCRIPT = ROOT / "system.py"
ATTENDANCE_DIR = ROOT / "Attendance"
STATUS_FILE = ROOT / "system_status.json"
MAIN_ATTENDANCE_FILE = ATTENDANCE_DIR / "attendance.csv"
SETTINGS_FILE = ROOT / "settings.json"
SYSTEM_PID_FILE = ROOT / "system.pid"
SYSTEM_STDOUT_LOG = ROOT / "system_stdout.log"
SYSTEM_STDERR_LOG = ROOT / "system_stderr.log"
CAPTURE_SCRIPT = ROOT / "capture_photos.py"

FINAL_ATTENDANCE_COLUMNS = [
    "row_id",
    "timestamp",
    "date",
    "time",
    "name",
    "confidence",
    "session_duration_min",
    "worked_outside",
]

HELPER_COLUMNS = [
    "confidence_float",
    "hour",
]

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True
if "refresh_sec" not in st.session_state:
    st.session_state.refresh_sec = 2
if "screen_refreshed" not in st.session_state:
    st.session_state.screen_refreshed = False

# =========================
# CSS
# =========================
st.markdown("""
<style>
:root {
    --bg-primary: #0a0f1a;
    --bg-secondary: #111827;
    --bg-card: #0f172a;
    --bg-glass: rgba(15, 23, 42, 0.78);
    --border: #334155;
    --border-hover: #475569;
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #64748b;
    --accent-primary: #0ea5e9;
    --accent-success: #10b981;
    --accent-warning: #f59e0b;
    --accent-danger: #ef4444;
    --radius: 14px;
    --shadow-sm: 0 4px 10px rgba(0,0,0,.18);
    --shadow-md: 0 10px 24px rgba(0,0,0,.22);
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0f1a 0%, #172033 100%);
}

.block-container {
    max-width: 98%;
    padding-top: 1.5rem;
}

h1 {
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}

h2, h3, h4 {
    color: var(--text-primary);
    font-weight: 700;
}

div[data-testid="metric-container"], .card {
    background: var(--bg-glass);
    backdrop-filter: blur(14px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-md);
}

.card {
    padding: 1rem 1.1rem;
}

div[data-testid="metric-container"]:hover, .card:hover {
    border-color: var(--border-hover);
}

div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.5rem;
    padding: 0.4rem;
    background: rgba(30, 41, 59, 0.45);
    border-radius: var(--radius);
}

div[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 12px !important;
    padding: 0.75rem 1.25rem !important;
    font-weight: 600;
}

div[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--accent-primary) !important;
    color: white !important;
}

div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
}

.status-ok { color: #10b981; font-weight: 700; }
.status-warn { color: #f59e0b; font-weight: 700; }
.status-bad { color: #ef4444; font-weight: 700; }
.status-muted { color: #94a3b8; font-weight: 700; }

.small-note {
    color: #94a3b8;
    font-size: 0.9rem;
}

.stream-box {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 0.8rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def get_today_file():
    return ATTENDANCE_DIR / f"{date.today().strftime('%Y-%m-%d')}.csv"

def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()

def normalize_attendance_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=FINAL_ATTENDANCE_COLUMNS)

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    aliases = {
        "Timestamp": "timestamp",
        "Name": "name",
        "Confidence": "confidence",
        "Session": "session_duration_min",
        "duration_min": "session_duration_min",
    }
    df.rename(columns=aliases, inplace=True)

    for col in FINAL_ATTENDANCE_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    df["date"] = ts.dt.strftime("%Y-%m-%d").fillna("")
    df["time"] = ts.dt.strftime("%H:%M").fillna("")

    for col in ["row_id", "name", "confidence", "session_duration_min", "worked_outside"]:
        df[col] = df[col].fillna("").astype(str).replace(["nan", "NaN", "None", "NaT"], "")

    df = df[~((df["timestamp"].str.strip() == "") & (df["name"].str.strip() == ""))].copy()
    df = df.reindex(columns=FINAL_ATTENDANCE_COLUMNS)

    sort_ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.assign(_sort_ts=sort_ts).sort_values("_sort_ts", ascending=False).drop(columns="_sort_ts")

    return df

def ensure_row_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp" not in df.columns:
        df["timestamp"] = ""
    if "name" not in df.columns:
        df["name"] = ""

    ts_series = df["timestamp"].fillna("").astype(str).str.strip()
    name_series = df["name"].fillna("").astype(str).str.strip()

    def clean_text(x):
        x = str(x).strip()
        return "" if x.lower() in ["nan", "nat", "none"] else x

    ts_series = ts_series.apply(clean_text)
    name_series = name_series.apply(clean_text)

    generated_ids = [
        f"{ts}__{name}__{i}" if (ts or name) else ""
        for i, (ts, name) in enumerate(zip(ts_series, name_series))
    ]

    if "row_id" not in df.columns:
        df["row_id"] = generated_ids
    else:
        df["row_id"] = df["row_id"].fillna("").astype(str).str.strip()
        bad_mask = (
            df["row_id"].str.lower().isin(["", "nan", "nat", "none"])
            | df["row_id"].str.startswith("nan__")
            | df["row_id"].str.startswith("nat__")
        )
        for idx in df.index[bad_mask]:
            df.at[idx, "row_id"] = generated_ids[idx]

    return df

def prepare_attendance_export(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in HELPER_COLUMNS if c in df.columns], errors="ignore")

    for col in FINAL_ATTENDANCE_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    for col in FINAL_ATTENDANCE_COLUMNS:
        df[col] = df[col].fillna("").astype(str).replace(["nan", "NaN", "None", "NaT"], "")

    if "timestamp" in df.columns and "name" in df.columns:
        df = df[~((df["timestamp"].str.strip() == "") & (df["name"].str.strip() == ""))].copy()

    return df.reindex(columns=FINAL_ATTENDANCE_COLUMNS)

def load_system_status():
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def load_attendance_data():
    df = safe_read_csv(MAIN_ATTENDANCE_FILE)
    df = normalize_attendance_df(df)
    df = ensure_row_id(df)
    return prepare_attendance_export(df)

def load_today_data():
    df = safe_read_csv(get_today_file())
    df = normalize_attendance_df(df)
    df = ensure_row_id(df)
    return prepare_attendance_export(df)

def read_encode_log():
    if not ENCODE_LOG.exists():
        return pd.DataFrame(columns=["timestamp", "level", "message"])

    try:
        lines = ENCODE_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return pd.DataFrame(columns=["timestamp", "level", "message"])

    records = []
    patterns = [
        r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?(INFO|WARNING|ERROR).*?(.*)',
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?(INFO|WARNING|ERROR)\s*-?\s*(.*)',
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (INFO|WARNING|ERROR) - (.*)',
    ]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        matched = False
        for pattern in patterns:
            m = re.match(pattern, line, re.IGNORECASE)
            if m:
                timestamp, level, msg = m.groups()
                records.append({
                    "timestamp": timestamp,
                    "level": level.upper(),
                    "message": msg.strip()
                })
                matched = True
                break

        if not matched:
            records.append({
                "timestamp": "N/A",
                "level": "RAW",
                "message": line
            })

    return pd.DataFrame(records)

def get_dataset_summary():
    if not DATASET_PATH.exists():
        return pd.DataFrame(columns=["name", "image_count"])

    rows = []
    for person_dir in sorted(DATASET_PATH.iterdir()):
        if person_dir.is_dir():
            image_count = sum(
                1 for p in person_dir.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            )
            rows.append({"name": person_dir.name, "image_count": image_count})

    return pd.DataFrame(rows)

def convert_df_to_csv_bytes(df: pd.DataFrame):
    return prepare_attendance_export(df).to_csv(index=False).encode("utf-8")

def run_encode():
    if not ENCODE_SCRIPT.exists():
        st.error("encode_faces.py not found.")
        return

    with st.spinner("Running encode_faces.py..."):
        result = subprocess.run(
            [sys.executable, str(ENCODE_SCRIPT)],
            cwd=str(ROOT),
            capture_output=True,
            text=True
        )

    if result.returncode == 0:
        st.success("✅ Encoding completed successfully.")
        st.rerun()
    else:
        st.error("❌ Encoding failed.")
        if result.stderr:
            st.code(result.stderr)

def delete_person(person_name):
    person_dir = DATASET_PATH / person_name
    if person_dir.exists():
        shutil.rmtree(person_dir)
        st.success(f"✅ Deleted {person_name}")
        st.rerun()

def system_health(status):
    if not status:
        return "UNKNOWN"
    if status.get("system_mode") == "error":
        return "ERROR"
    if status.get("camera_connected") and status.get("gpu_enabled"):
        return "HEALTHY"
    if status.get("camera_connected"):
        return "DEGRADED"
    return "OFFLINE"

def uptime_text(seconds):
    seconds = int(seconds or 0)
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h}h {m}m {s}s"

def fmt_ts(ts):
    try:
        return pd.to_datetime(ts).strftime("%d %b %Y %H:%M:%S")
    except Exception:
        return str(ts)

def save_attendance_updates(edited_df: pd.DataFrame):
    if edited_df.empty:
        st.warning("No edited data to save.")
        return

    main_df = safe_read_csv(MAIN_ATTENDANCE_FILE)
    if main_df.empty:
        st.error("Main attendance CSV is empty or missing.")
        return

    main_df = normalize_attendance_df(main_df)
    main_df = ensure_row_id(main_df)

    edited_df = normalize_attendance_df(edited_df)
    edited_df = ensure_row_id(edited_df)

    main_df = main_df.set_index("row_id")
    edited_df = edited_df.set_index("row_id")

    common_ids = edited_df.index.intersection(main_df.index)
    if len(common_ids) == 0:
        st.warning("No matching rows found to update.")
        return

    for col in ["name", "worked_outside", "session_duration_min"]:
        if col in edited_df.columns and col in main_df.columns:
            main_df.loc[common_ids, col] = edited_df.loc[common_ids, col]

    main_df = main_df.reset_index()
    export_df = prepare_attendance_export(main_df)
    export_df = export_df.dropna(how="all")
    export_df = export_df[
        export_df.apply(lambda r: any(str(v).strip() for v in r.values), axis=1)
    ].copy()

    export_df.to_csv(MAIN_ATTENDANCE_FILE, index=False, encoding="utf-8", lineterminator="\n")
    st.success("Attendance CSV updated successfully.")

def load_settings():
    default_settings = {
        "recognition_threshold": 75.0,
        "cooldown_minutes": 5,
        "min_images_per_person": 3,
        "stream_url": "http://10.227.221.34:8080/video"
    }

    if not SETTINGS_FILE.exists():
        SETTINGS_FILE.write_text(json.dumps(default_settings, indent=2), encoding="utf-8")
        return default_settings

    try:
        settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))

        if "recognition_threshold" not in settings and "recognitionthreshold" in settings:
            settings["recognition_threshold"] = settings["recognitionthreshold"]

        if "cooldown_minutes" not in settings and "cooldownminutes" in settings:
            settings["cooldown_minutes"] = settings["cooldownminutes"]

        if "min_images_per_person" not in settings and "minimagesperperson" in settings:
            settings["min_images_per_person"] = settings["minimagesperperson"]

        if "stream_url" not in settings and "streamurl" in settings:
            settings["stream_url"] = settings["streamurl"]

        if "cooldown_minutes" not in settings and "cooldown_seconds" in settings:
            old_seconds = int(settings.get("cooldown_seconds", 300))
            settings["cooldown_minutes"] = max(1, round(old_seconds / 60))

        for k, v in default_settings.items():
            settings.setdefault(k, v)

        return settings
    except Exception:
        return default_settings

def save_settings(settings: dict):
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")

def get_pid_from_file():
    try:
        if SYSTEM_PID_FILE.exists():
            raw = SYSTEM_PID_FILE.read_text(encoding="utf-8").strip()
            if raw:
                return int(raw)
    except Exception:
        return None
    return None

def remove_pid_file():
    try:
        if SYSTEM_PID_FILE.exists():
            SYSTEM_PID_FILE.unlink()
    except Exception:
        pass

def is_process_running(pid: int) -> bool:
    if not pid:
        return False
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True
            )
            return str(pid) in result.stdout
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False

def get_system_pid():
    pid = get_pid_from_file()
    if pid and is_process_running(pid):
        return pid
    if pid and not is_process_running(pid):
        remove_pid_file()
    return None

def start_system():
    if not SYSTEM_SCRIPT.exists():
        return False, "system.py not found."

    current_pid = get_system_pid()
    if current_pid:
        return False, f"system.py is already running with PID {current_pid}."

    try:
        out_f = open(SYSTEM_STDOUT_LOG, "a", encoding="utf-8")
        err_f = open(SYSTEM_STDERR_LOG, "a", encoding="utf-8")

        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            proc = subprocess.Popen(
                [sys.executable, str(SYSTEM_SCRIPT)],
                cwd=str(ROOT),
                stdout=out_f,
                stderr=err_f,
                universal_newlines=True,  
                encoding="utf-8",         
                errors="replace",         
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                close_fds=False
            )
        else:
            proc = subprocess.Popen(
                [sys.executable, str(SYSTEM_SCRIPT)],
                cwd=str(ROOT),
                stdout=out_f,
                stderr=err_f,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True
            )

        time.sleep(3)

        if proc.poll() is not None:
            err_tail = read_recent_text_log(SYSTEM_STDERR_LOG, 60)
            out_tail = read_recent_text_log(SYSTEM_STDOUT_LOG, 60)
            return False, (
                "system.py started but exited immediately.\n\n"
                f"STDERR:\n{err_tail or 'No stderr'}\n\n"
                f"STDOUT:\n{out_tail or 'No stdout'}"
            )

        SYSTEM_PID_FILE.write_text(str(proc.pid), encoding="utf-8")
        return True, f"Started system.py with PID {proc.pid}."

    except Exception as e:
        return False, f"Failed to start system.py: {e}"

def stop_system():
    pid = get_system_pid()
    if not pid:
        remove_pid_file()
        return False, "system.py is not running."

    try:
        if os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                return False, result.stderr.strip() or f"Failed to stop PID {pid}."
        else:
            os.kill(pid, signal.SIGTERM)

        time.sleep(1)
        remove_pid_file()
        return True, f"Stopped system.py (PID {pid})."
    except Exception as e:
        return False, f"Failed to stop system.py: {e}"

def read_recent_text_log(path: Path, tail_lines: int = 80):
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-tail_lines:])
    except Exception:
        return ""

def open_capture_terminal(clean_name: str):
    if not CAPTURE_SCRIPT.exists():
        st.error("capture_photos.py not found.")
        return

    try:
        if os.name == "nt":
            cmd = f'start cmd /k "cd /d {ROOT} && python "{CAPTURE_SCRIPT.name}" "{clean_name}""'
            os.system(cmd)
            st.success(f"🎥 Terminal opened for '{clean_name}' - capture now!")
        else:
            subprocess.Popen(
                [sys.executable, str(CAPTURE_SCRIPT), clean_name],
                cwd=str(ROOT),
                start_new_session=True
            )
            st.success(f"🎥 Capture script started for '{clean_name}'")
    except Exception as e:
        st.error(f"Failed to open capture process: {e}")

def render_stream_block(url: str, label: str):
    st.markdown(f"**{label}**")
    st.code(url, language="text")
    stream_html = f"""
    <div style="border-radius:12px; overflow:hidden; border:1px solid #334155;">
        <img src="{url}" style="width:100%;height:220px;object-fit:cover;display:block;" />
    </div>
    """
    st.markdown(stream_html, unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
status = load_system_status()
attendance_df = load_attendance_data()
today_df = load_today_data()
dataset_df = get_dataset_summary()
log_df = read_encode_log()
today_file = get_today_file()
system_pid = get_system_pid()
system_running = system_pid is not None

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## AMS Admin")
    st.caption("Live dashboard for attendance + system monitoring")

    st.divider()
    st.session_state.auto_refresh = st.toggle("🔄 Auto Refresh", value=st.session_state.auto_refresh)
    refresh_options = [2, 5, 10, 15, 30]
    default_index = refresh_options.index(st.session_state.refresh_sec) if st.session_state.refresh_sec in refresh_options else 0
    st.session_state.refresh_sec = st.selectbox("Refresh every (sec)", refresh_options, index=default_index)

    st.divider()
    st.markdown("### Filters")

    available_names = sorted(
        attendance_df["name"].replace("", pd.NA).dropna().unique().tolist()
    ) if not attendance_df.empty else []

    selected_names = st.multiselect("People", available_names)

    valid_dates = pd.to_datetime(attendance_df["date"], errors="coerce").dropna() if not attendance_df.empty else pd.Series(dtype="datetime64[ns]")

    if valid_dates.empty:
        min_date = date.today()
        max_date = date.today()
    else:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

    date_range = st.date_input("Date range", (min_date, max_date))
    min_conf = st.slider("Min confidence %", 0, 100, 0)

    st.divider()
    if st.button("🔄 Refresh now", use_container_width=True):
        st.rerun()

# =========================
# FILTER DATA
# =========================
filtered_df = attendance_df.copy()

if not filtered_df.empty:
    if selected_names:
        filtered_df = filtered_df[filtered_df["name"].isin(selected_names)]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df["date_dt"] = pd.to_datetime(filtered_df["date"], errors="coerce").dt.normalize()
        filtered_df = filtered_df[
            filtered_df["date_dt"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        ].drop(columns=["date_dt"], errors="ignore")

    filtered_df["confidence_float"] = pd.to_numeric(
        filtered_df["confidence"].astype(str).str.rstrip("%").str.strip(),
        errors="coerce"
    ).fillna(0)

    filtered_df = filtered_df[filtered_df["confidence_float"] >= min_conf]

# =========================
# HEADER
# =========================
st.markdown("# AMS Admin Dashboard")
st.caption("📊 Real-time attendance monitoring • Live system status • CSV + Daily log aware")

# =========================
# KPI ROWS
# =========================
health = system_health(status)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("🩺 Health", health)
with c2:
    st.metric("📹 Camera", "✅ LIVE" if status.get("camera_connected") else "❌ OFFLINE")
with c3:
    st.metric("💻 GPU", "✅ CUDA" if status.get("gpu_enabled") else "❌ CPU")
with c4:
    st.metric("⚡ FPS", f"{float(status.get('fps', 0) or 0):.1f}")
with c5:
    st.metric("👥 Matches", int(status.get("matches", 0) or 0))
with c6:
    st.metric("📝 Logs Written", int(status.get("logs_written", 0) or 0))

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.metric("📈 Total Entries", len(filtered_df))
with m2:
    st.metric("👤 Unique People", filtered_df["name"].replace("", pd.NA).dropna().nunique() if not filtered_df.empty else 0)
with m3:
    avg_conf = filtered_df["confidence_float"].mean() if ("confidence_float" in filtered_df.columns and not filtered_df.empty) else 0
    st.metric("🎯 Avg Confidence", f"{0 if pd.isna(avg_conf) else avg_conf:.1f}%")
with m4:
    st.metric("🗂️ Dataset IDs", len(dataset_df))
with m5:
    st.metric("📅 Today Records", len(today_df))
with m6:
    st.metric("📄 Main CSV Rows", len(attendance_df))

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "👥 Attendance", "🔴 Live Monitor", "🗂️ Dataset", "⚙️ Actions", "📹 Screen"
])

# =========================
# TAB 1 - OVERVIEW
# =========================
with tab1:
    a, b, c = st.columns([1.05, 1.25, 1.0])

    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### System Files")
        st.markdown(f"Dataset: <span class='status-ok'>OK</span>" if DATASET_PATH.exists() else "Dataset: <span class='status-bad'>Missing</span>", unsafe_allow_html=True)
        st.markdown(f"Encodings: <span class='status-ok'>Ready</span>" if ENCODINGS_FILE.exists() else "Encodings: <span class='status-warn'>Pending</span>", unsafe_allow_html=True)
        st.markdown(f"Attendance dir: <span class='status-ok'>Active</span>" if ATTENDANCE_DIR.exists() else "Attendance dir: <span class='status-bad'>Missing</span>", unsafe_allow_html=True)
        st.markdown(f"Main CSV: <span class='status-ok'>{MAIN_ATTENDANCE_FILE.name}</span>" if MAIN_ATTENDANCE_FILE.exists() else "Main CSV: <span class='status-warn'>Pending</span>", unsafe_allow_html=True)
        st.markdown(f"Today CSV: <span class='status-ok'>{today_file.name}</span>" if today_file.exists() else f"Today CSV: <span class='status-warn'>{today_file.name} not yet created</span>", unsafe_allow_html=True)
        st.markdown(f"Status JSON: <span class='status-ok'>Available</span>" if STATUS_FILE.exists() else "Status JSON: <span class='status-warn'>Missing</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Recent Activity")

        if not filtered_df.empty:
            recent_df = prepare_attendance_export(ensure_row_id(normalize_attendance_df(filtered_df.copy())))
            display_cols = ["timestamp", "name", "confidence", "session_duration_min", "worked_outside"]
            recent_display = recent_df[[c for c in display_cols if c in recent_df.columns]].copy()

            recent_display["timestamp_sort"] = pd.to_datetime(recent_display["timestamp"], errors="coerce")
            recent_display = recent_display.sort_values("timestamp_sort", ascending=False).drop(columns="timestamp_sort")

            st.dataframe(
                recent_display.head(12),
                use_container_width=True,
                height=310,
                hide_index=True
            )
        else:
            st.info("No attendance rows match current filters.")

        st.markdown('</div>', unsafe_allow_html=True)

    with c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Quick Export")

        if not today_df.empty:
            st.download_button(
                "📅 Download Today CSV",
                convert_df_to_csv_bytes(today_df),
                file_name=f"attendance_{date.today()}.csv",
                mime="text/csv",
                use_container_width=True
            )

        if not filtered_df.empty:
            export_filtered = prepare_attendance_export(filtered_df)
            st.download_button(
                "🔍 Download Filtered CSV",
                convert_df_to_csv_bytes(export_filtered),
                file_name="attendance_filtered.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-note'>Main file: {MAIN_ATTENDANCE_FILE}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-note'>Today file: {today_file}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Attendance Trend")
    if not filtered_df.empty:
        trend_df = filtered_df.copy()
        trend_df["date_dt"] = pd.to_datetime(trend_df["date"], errors="coerce")
        trend_df = trend_df.dropna(subset=["date_dt"])

        if not trend_df.empty:
            daily = trend_df.groupby("date_dt").size().reset_index(name="entries")
            fig = px.line(daily, x="date_dt", y="entries", markers=True, title="Daily attendance entries")
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, key="attendance_trend")
        else:
            st.info("Not enough attendance data for trend chart.")
    else:
        st.info("Not enough attendance data for trend chart.")

# =========================
# TAB 2 - ATTENDANCE
# =========================
with tab2:
    st.markdown("### Attendance Analytics")

    working_df = attendance_df.copy()
    if not working_df.empty:
        working_df["confidence_float"] = pd.to_numeric(
            working_df["confidence"].astype(str).str.rstrip("%").str.strip(),
            errors="coerce"
        ).fillna(0)
        working_df["hour"] = pd.to_datetime(working_df["timestamp"], errors="coerce").dt.hour

        if selected_names:
            working_df = working_df[working_df["name"].isin(selected_names)]

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            working_df["date_dt"] = pd.to_datetime(working_df["date"], errors="coerce").dt.normalize()
            working_df = working_df[
                working_df["date_dt"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
            ]

        if min_conf > 0:
            working_df = working_df[working_df["confidence_float"] >= min_conf]
    else:
        working_df = pd.DataFrame()

    name_filter = st.text_input("Filter by name", "").strip()
    if name_filter and not working_df.empty:
        working_df = working_df[working_df["name"].str.contains(name_filter, case=False, na=False)]

    if working_df.empty:
        st.warning("No attendance data matches current filters.")
    else:
        r1, r2 = st.columns(2)

        with r1:
            top_people = (
                working_df["name"]
                .replace("", pd.NA)
                .dropna()
                .value_counts()
                .head(12)
                .reset_index()
            )
            top_people.columns = ["name", "entries"]

            fig = px.bar(
                top_people,
                x="entries",
                y="name",
                orientation="h",
                title="Top Attendees",
                color="entries",
                color_continuous_scale="viridis"
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True, key="top_attendees")

        with r2:
            valid_conf = working_df[
                (working_df["confidence_float"] >= 70) &
                (working_df["confidence_float"] <= 100)
            ]

            if len(valid_conf) > 0:
                fig = px.histogram(
                    valid_conf,
                    x="confidence_float",
                    nbins=15,
                    title="🎯 Confidence Distribution (70-100%)",
                    color_discrete_sequence=["#10b981"],
                    labels={"confidence_float": "Confidence (%)"}
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#f8fafc",
                    height=400,
                    bargap=0.05,
                    xaxis_title="Confidence (%)",
                    yaxis_title="Count"
                )
                fig.update_xaxes(range=[65, 102], dtick=5)
                st.plotly_chart(fig, use_container_width=True, key="confidence_hist")
            else:
                st.info("❌ No confidence data in 70-100% range.")

        r3, r4 = st.columns(2)

        with r3:
            hourly = (
                working_df.dropna(subset=["hour"])
                .groupby("hour")
                .size()
                .reset_index(name="entries")
            )

            if not hourly.empty:
                fig = px.bar(
                    hourly,
                    x="hour",
                    y="entries",
                    title="Attendance by Hour",
                    color="entries",
                    color_continuous_scale="viridis"
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#f8fafc",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True, key="hourly_attendance")
            else:
                st.info("No hourly data available.")

        with r4:
            summary_df = working_df.copy()
            summary_df["session_duration_num"] = pd.to_numeric(summary_df["session_duration_min"], errors="coerce")

            agg_dict = {
                "entries": ("name", "size"),
                "avg_confidence": ("confidence_float", "mean"),
                "avg_session_min": ("session_duration_num", "mean"),
            }

            summary = (
                summary_df.groupby("name", dropna=False)
                .agg(**agg_dict)
                .reset_index()
                .round(2)
            )
            summary["name"] = summary["name"].replace("", "UNKNOWN")

            st.dataframe(
                summary.sort_values("entries", ascending=False),
                use_container_width=True,
                height=400
            )

        st.markdown("### 📝 Attendance Records")

        editable_base = prepare_attendance_export(ensure_row_id(normalize_attendance_df(working_df.copy())))
        show_cols = [c for c in FINAL_ATTENDANCE_COLUMNS if c in editable_base.columns]
        df_display = editable_base[show_cols].copy()

        edited_df = st.data_editor(
            df_display,
            num_rows="fixed",
            use_container_width=True,
            height=420,
            disabled=[
                c for c in ["row_id", "timestamp", "date", "time", "confidence", "session_duration_min"]
                if c in df_display.columns
            ],
            column_config={
                "row_id": st.column_config.TextColumn("Row ID"),
                "timestamp": st.column_config.TextColumn("Timestamp"),
                "date": st.column_config.TextColumn("Date"),
                "time": st.column_config.TextColumn("Time"),
                "name": st.column_config.TextColumn("Name", required=True),
                "confidence": st.column_config.TextColumn("Confidence"),
                "session_duration_min": st.column_config.TextColumn("Session (min)"),
                "worked_outside": st.column_config.TextColumn("Worked Outside"),
            },
            key="attendance_editor"
        )

        if st.button("💾 Save Attendance Changes", use_container_width=True):
            save_attendance_updates(edited_df)
            st.rerun()

        with st.expander("📋 Read-only filtered table", expanded=False):
            st.dataframe(df_display, use_container_width=True, height=380)

# =========================
# TAB 3 - LIVE MONITOR
# =========================
with tab3:
    st.markdown("### Live System Monitor")

    top_a, top_b, top_c = st.columns([1.1, 1.1, 1.4])

    with top_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ▶️ System Control")

        st.write(f"**system.py:** {'Running' if system_running else 'Stopped'}")
        st.write(f"**PID:** {system_pid if system_pid else 'N/A'}")
        st.write(f"**Script path:** `{SYSTEM_SCRIPT}`")

        start_disabled = system_running or (not SYSTEM_SCRIPT.exists())
        if st.button("▶️ Start system.py", use_container_width=True, disabled=start_disabled):
            ok, msg = start_system()
            if ok:
                st.success(msg)
            else:
                st.error(msg)
            st.rerun()

        stop_disabled = not system_running
        if st.button("⏹️ Stop system.py", use_container_width=True, disabled=stop_disabled):
            ok, msg = stop_system()
            if ok:
                st.success(msg)
            else:
                st.warning(msg)
            st.rerun()

        if not SYSTEM_SCRIPT.exists():
            st.error("system.py not found in project root.")

        st.markdown('</div>', unsafe_allow_html=True)

    with top_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📡 Runtime Snapshot")
        st.metric("Process State", "RUNNING" if system_running else "STOPPED")
        st.metric("System Mode", str(status.get("system_mode", "unknown")))
        st.metric("Camera Connected", "YES" if status.get("camera_connected") else "NO")
        st.metric("GPU Enabled", "YES" if status.get("gpu_enabled") else "NO")
        st.markdown('</div>', unsafe_allow_html=True)

    with top_c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📝 system.py Output")
        stdout_tail = read_recent_text_log(SYSTEM_STDOUT_LOG, 30)
        stderr_tail = read_recent_text_log(SYSTEM_STDERR_LOG, 30)

        if stdout_tail:
            st.caption("Stdout")
            st.code(stdout_tail, language="text")
        else:
            st.info("No stdout log yet.")

        if stderr_tail:
            st.caption("Stderr")
            st.code(stderr_tail, language="text")
        st.markdown('</div>', unsafe_allow_html=True)

    l1, l2 = st.columns([1.0, 1.2])

    with l1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if health == "HEALTHY":
            st.markdown("### 🟢 Healthy")
        elif health == "DEGRADED":
            st.markdown("### 🟡 Degraded")
        elif health == "OFFLINE":
            st.markdown("### 🔴 Offline")
        else:
            st.markdown("### ⚪ Unknown")

        if status:
            frame_shape = status.get("frame_shape", "N/A")
            frame_shape_display = " × ".join(map(str, frame_shape)) if isinstance(frame_shape, (list, tuple)) else str(frame_shape)

            st.metric("Mode", str(status.get("system_mode", "unknown")))
            st.metric("Stream", str(status.get("stream_url", "N/A")))
            st.metric("Frame", frame_shape_display)
            st.metric("Uptime", uptime_text(status.get("uptime_sec", 0)))
            st.metric("Updated", fmt_ts(status.get("updated_at", "")))

            last_person = status.get("last_recognized", "")
            if last_person:
                st.success(f"👤 Last recognized: {last_person}")

            err = status.get("error_msg", "")
            if err:
                st.error(f"⚠️ Error: {err}")
        else:
            st.info("No system_status.json data available.")

        st.markdown('</div>', unsafe_allow_html=True)

    with l2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Encode Logs")

        if log_df.empty:
            st.info("No encode logs available.")
        else:
            levels = sorted(log_df["level"].dropna().unique())
            selected_levels = st.multiselect("Levels", levels, default=levels, key="log_levels")
            search_text = st.text_input("Search logs", key="log_search")

            filtered_logs = log_df[log_df["level"].isin(selected_levels)]
            if search_text:
                filtered_logs = filtered_logs[
                    filtered_logs["message"].astype(str).str.contains(search_text, case=False, na=False)
                ]

            st.dataframe(filtered_logs.tail(100), use_container_width=True, height=400)

        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Raw Status JSON"):
        st.json(status if status else {"status": "No live data"})

# =========================
# TAB 4 - DATASET
# =========================
with tab4:
    st.markdown("### Identity Manager")

    if dataset_df.empty:
        st.warning("Dataset folder is empty.")
    else:
        fig = px.bar(
            dataset_df.sort_values("image_count", ascending=False).head(15),
            x="image_count", y="name", orientation="h", title="Images per identity"
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
        st.plotly_chart(fig, use_container_width=True, key="dataset_chart")

        with st.expander("📋 Dataset Table", expanded=False):
            st.dataframe(
                dataset_df.sort_values("image_count", ascending=False),
                use_container_width=True,
                height=320
            )

        with st.expander("🖼️ Preview Identity", expanded=False):
            selected_person = st.selectbox(
                "Select identity",
                dataset_df["name"].tolist(),
                key="dataset_preview_person"
            )

            person_images = sorted((DATASET_PATH / selected_person).glob("*"))
            preview_images = [
                str(img) for img in person_images
                if img.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ][:12]

            if preview_images:
                cols = st.columns(4)
                for i, img_path in enumerate(preview_images):
                    with cols[i % 4]:
                        st.image(
                            img_path,
                            use_container_width=True,
                            caption=Path(img_path).name
                        )
            else:
                st.info("No images found for this identity.")

        with st.expander("➕ Add New Identity", expanded=False):
            new_name = st.text_input("👤 New person name", key="new_identity_name")
            clean_name = re.sub(r'[^A-Za-z0-9_-]+', '_', new_name.strip())

            col_mode1, col_mode2 = st.columns(2)

            with col_mode1:
                uploaded_files = st.file_uploader(
                    "📁 Browse photos", type=["jpg", "jpeg", "png"],
                    accept_multiple_files=True, key="browse_upload"
                )
                if uploaded_files and st.button("🚀 Create from Upload", key="create_upload_btn", use_container_width=True):
                    if not clean_name:
                        st.error("Enter person name")
                    else:
                        person_dir = DATASET_PATH / clean_name
                        person_dir.mkdir(parents=True, exist_ok=True)
                        saved = 0
                        for f in uploaded_files:
                            with open(person_dir / f.name, "wb") as file:
                                file.write(f.getbuffer())
                            saved += 1
                        st.success(f"✅ {clean_name}: {saved} images")
                        st.rerun()

            with col_mode2:
                if st.button("🎥 Launch Capture Photos", key="launch_capture_btn", use_container_width=True):
                    if not clean_name:
                        st.error("Enter person name first")
                    else:
                        open_capture_terminal(clean_name)
                        st.balloons()

            st.caption("*Capture mode reuses system.py's buffalo_l + IP camera pipeline*")

        with st.expander("✏️ Rename or Delete Identity", expanded=False):
            person_to_manage = st.selectbox(
                "Choose identity",
                dataset_df["name"].tolist(),
                key="manage_identity_person"
            )

            new_person_name = st.text_input(
                "Rename selected identity to",
                value=person_to_manage,
                key="rename_identity_input"
            )

            col_a, col_b = st.columns(2)

            with col_a:
                if st.button("✏️ Rename Identity", key="rename_identity_btn", use_container_width=True):
                    old_path = DATASET_PATH / person_to_manage
                    clean_new_name = re.sub(r'[^A-Za-z0-9_-]+', '_', new_person_name.strip())
                    new_path = DATASET_PATH / clean_new_name

                    if not clean_new_name:
                        st.warning("New name cannot be empty.")
                    elif clean_new_name == person_to_manage:
                        st.info("No changes detected.")
                    elif new_path.exists():
                        st.error("Another identity with this name already exists.")
                    else:
                        old_path.rename(new_path)
                        st.success(f"✅ Renamed {person_to_manage} → {clean_new_name}")
                        st.rerun()

            with col_b:
                confirm_delete = st.checkbox(
                    f"Confirm delete: {person_to_manage}",
                    key="confirm_delete_identity"
                )

                if st.button("💥 Delete Identity", key="delete_identity_btn", use_container_width=True):
                    if confirm_delete:
                        delete_person(person_to_manage)
                    else:
                        st.warning("Please confirm deletion first.")

        with st.expander("📤 Upload More Photos", expanded=False):
            person_for_upload = st.selectbox(
                "Select identity to add photos",
                dataset_df["name"].tolist(),
                key="upload_more_person"
            )

            more_files = st.file_uploader(
                "Choose more photos",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="upload_more_files"
            )

            if st.button("📸 Upload Photos", key="upload_more_btn", use_container_width=True):
                if not more_files:
                    st.warning("No files selected.")
                else:
                    person_dir = DATASET_PATH / person_for_upload
                    person_dir.mkdir(parents=True, exist_ok=True)

                    saved_count = 0
                    for file in more_files:
                        save_path = person_dir / file.name
                        with open(save_path, "wb") as f:
                            f.write(file.getbuffer())
                        saved_count += 1

                    st.success(f"✅ Uploaded {saved_count} image(s) to {person_for_upload}.")
                    st.rerun()

# =========================
# TAB 5 - ACTIONS
# =========================
with tab5:
    st.markdown("### ⚙️ System Settings & Maintenance")

    c1, c2 = st.columns(2)
    settings = load_settings()

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Recognition Settings")

        threshold = st.slider(
            "Log threshold %",
            min_value=50,
            max_value=95,
            value=int(settings.get("recognition_threshold", 75)),
            step=1
        )
        cooldown = st.slider(
            "Cooldown (minutes)",
            min_value=1,
            max_value=60,
            value=int(settings.get("cooldown_minutes", 5)),
            step=1
        )
        min_images = st.slider(
            "Minimum images per person",
            min_value=1,
            max_value=20,
            value=int(settings.get("min_images_per_person", 3)),
            step=1
        )

        st.divider()
        st.markdown("**📹 Camera URLs**")

        primary_url = st.text_input(
            "Primary stream",
            value=str(settings.get("stream_url", "http://10.227.221.34:8080/video")),
            key="primary_stream_input"
        )

        if st.button("🔌 Test Primary Stream", key="test_stream"):
            try:
                import requests
                r = requests.head(primary_url, timeout=5, allow_redirects=True)
                if r.status_code < 400:
                    st.success("✅ Stream reachable")
                else:
                    st.warning(f"⚠️ HTTP {r.status_code}")
            except Exception:
                st.error("❌ Stream unreachable")

        stream_url = st.text_input(
            "Default stream URL",
            value=str(settings.get("stream_url", "http://10.227.221.34:8080/video")),
            key="stream_url_input"
        )

        if st.button("💾 Save Settings", use_container_width=True):
            settings["recognition_threshold"] = float(threshold)
            settings["cooldown_minutes"] = int(cooldown)
            settings["min_images_per_person"] = int(min_images)
            settings["stream_url"] = stream_url.strip() or primary_url.strip()
            save_settings(settings)
            st.success("✅ Settings saved! Restart system.py to apply.")

        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🔧 Maintenance")

        if st.button("🚀 Rebuild Encodings", use_container_width=True):
            run_encode()

        if st.button("📊 Check Google Sheets Sync", use_container_width=True):
            st.info("✅ Manual sync check placeholder.")

        if st.button("📄 Reload Dashboard Data", use_container_width=True):
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# TAB 6 - SCREEN
# =========================
with tab6:
    st.markdown("## 📹 LIVE-CAM")

    current_status = load_system_status()
    camera_urls = load_stream_url()

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        st.metric("🖥️ Status", str(current_status.get("system_mode", "unknown")).upper())
    with top2:
        st.metric("📸 FPS", f"{float(current_status.get('fps', 0) or 0):.1f}")
    with top3:
        st.metric("🎯 Matches", int(current_status.get("matches", 0) or 0))
    with top4:
        st.metric("📝 Logs", int(current_status.get("logs_written", 0) or 0))

    st.markdown("---")
    st.markdown("### Live Stream")

    if not camera_urls:
        st.warning("No camera URLs found from load_stream_url().")
    else:
        stream_cols = st.columns(min(3, len(camera_urls)))
        for i, url in enumerate(camera_urls[:3]):
            with stream_cols[i]:
                try:
                    render_stream_block(url, f"Stream {i+1}")
                except Exception as e:
                    st.error(f"❌ Stream {i+1} offline: {e}")

    st.markdown("---")
    st.markdown("### System Status")

    refresh_col1, refresh_col2 = st.columns([1, 3])
    with refresh_col1:
        if st.button("🔄 Refresh Screen Status", key="refresh_screen_status", use_container_width=True):
            st.session_state.screen_refreshed = True
            st.rerun()

    latest_status = load_system_status()
    if latest_status:
        st.json(latest_status)

        col_fps, col_matches, col_logs, col_faces = st.columns(4)
        with col_fps:
            st.metric("⚡ FPS", f"{float(latest_status.get('fps', 0) or 0):.1f}")
        with col_matches:
            st.metric("🎯 Matches", int(latest_status.get('matches', 0) or 0))
        with col_logs:
            st.metric("📝 Logs", int(latest_status.get('logs_written', 0) or 0))
        with col_faces:
            st.metric("👥 Face DB", int(latest_status.get('face_db_count', 0) or 0))
    else:
        st.error("⚠️ Status file not found - system.py not running?")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("**AMS Admin Dashboard** • Jwel Sharma • 2026")

# =========================
# AUTO REFRESH
# =========================
if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_sec)
    st.rerun()