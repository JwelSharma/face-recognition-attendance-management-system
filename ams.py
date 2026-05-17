#!/usr/bin/env python3
"""
🎯 JWEL AMS CLI - One command for everything
Usage: python ams.py [capture|encode|system|run|admin|status|clean]
"""

import sys
import subprocess
import argparse
from pathlib import Path
import shutil
from datetime import datetime
import csv
import codecs

# Fix Windows stdout encoding
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

SUPPRESS_SYSTEM_STDERR = True       #False for debugging

AMS_ROOT = Path(".")

SCRIPTS = {
    "capture": AMS_ROOT / "capture_photos.py",
    "encode": AMS_ROOT / "encode_faces.py",
    "system": AMS_ROOT / "system.py",
    "admin": AMS_ROOT / "admin_ui.py",
}

DATASET_PATH = AMS_ROOT / "dataset"
ENCODINGS_FILE = AMS_ROOT / "encodings.pickle"
LOG_FILE = AMS_ROOT / "encode_log.txt"

ATTENDANCE_DIR = AMS_ROOT / "Attendance"
TODAY = datetime.now().strftime("%Y-%m-%d")
DAILY_ATTENDANCE_FILE = ATTENDANCE_DIR / f"{TODAY}.csv"
MAIN_ATTENDANCE_FILE = ATTENDANCE_DIR / "attendance.csv"


def print_banner():
    print("""
🚀 JWEL AMS CLI v1.0
═══════════════════════════════════════════════
📸 capture     → capture_photos.py
🧬 encode      → encode_faces.py
🎥 system/run  → system.py (live recognition)
🖥️  admin      → admin_ui.py (Streamlit dashboard)
📊 status      → Check dataset/encodings/logs
🗑️  clean      → Delete dataset/encodings/logs
═══════════════════════════════════════════════
    """)


def resolve_command(command: str) -> str:
    aliases = {
        "run": "system",
    }
    return aliases.get(command, command)


def run_script(script_name):
    """Run AMS script with error handling."""
    script_name = resolve_command(script_name)
    script_path = SCRIPTS.get(script_name)

    if not script_path or not script_path.exists():
        print(f"❌ ERROR: {script_name}.py not found!")
        print(f"   Expected: {script_path}")
        sys.exit(1)

    print(f"🔥 Running {script_name}.py...")

    if script_name == "admin":
        result = subprocess.run(
            ["streamlit", "run", str(script_path)],
            cwd=str(AMS_ROOT)
        )
        return result.returncode == 0

    if script_name == "system":
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(AMS_ROOT),
            text=True,
            capture_output=True
        )

        print(f"\n↩ Return code: {result.returncode}")

        if result.stdout:
            print("\n--- SYSTEM STDOUT ---")
            print(result.stdout)

        if result.stderr:
            print("\n--- SYSTEM STDERR ---")
            print(result.stderr)

        return result.returncode == 0

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(AMS_ROOT)
    )
    return result.returncode == 0

def get_file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            return sum(1 for _ in reader)
    except Exception:
        return 0


def print_status():
    """Print pipeline status."""
    print("\n📊 AMS PIPELINE STATUS")
    print("=" * 50)

    dataset_exists = DATASET_PATH.exists()
    dataset_folders = len([p for p in DATASET_PATH.glob("*") if p.is_dir()]) if dataset_exists else 0
    total_images = sum(
        len(list(p.glob("*.jpg"))) + len(list(p.glob("*.jpeg"))) + len(list(p.glob("*.png")))
        for p in DATASET_PATH.glob("*") if p.is_dir()
    )

    encode_log_exists = LOG_FILE.exists()
    encode_log_bytes = get_file_size(LOG_FILE)

    daily_exists = DAILY_ATTENDANCE_FILE.exists()
    daily_bytes = get_file_size(DAILY_ATTENDANCE_FILE)
    daily_rows = count_csv_rows(DAILY_ATTENDANCE_FILE)

    main_exists = MAIN_ATTENDANCE_FILE.exists()
    main_bytes = get_file_size(MAIN_ATTENDANCE_FILE)
    main_rows = count_csv_rows(MAIN_ATTENDANCE_FILE)

    print(f"📁 Dataset:        {'✅' if dataset_exists else '❌'} ({dataset_folders} folders, {total_images} images)")
    print(f"🧬 Encodings:      {'✅' if ENCODINGS_FILE.exists() else '❌'}")
    print(f"📝 Encode log:     {'✅' if encode_log_exists else '❌'} ({encode_log_bytes} bytes)")
    print(f"📝 Daily CSV:      {'✅' if daily_exists else '❌'} ({daily_bytes} bytes, {daily_rows} rows)")
    print(f"📝 Main CSV:       {'✅' if main_exists else '❌'} ({main_bytes} bytes, {main_rows} rows)")

    for name, path in SCRIPTS.items():
        status = "✅" if path.exists() else "❌"
        print(f"📜 {name:9s}:     {status} {path.name}")


def clean_all():
    """Clean dataset, encodings, and optional logs."""
    confirm = input("🗑️  Delete dataset/, encodings.pickle, encode_log.txt? (y/N): ")
    if confirm.lower() != "y":
        print("❌ Cancelled.")
        return

    if DATASET_PATH.exists():
        shutil.rmtree(DATASET_PATH)
        print("✅ Deleted dataset/")

    if ENCODINGS_FILE.exists():
        ENCODINGS_FILE.unlink()
        print("✅ Deleted encodings.pickle")

    if LOG_FILE.exists():
        LOG_FILE.unlink()
        print("✅ Deleted encode_log.txt")

    print("🧹 Clean complete!")


def main():
    parser = argparse.ArgumentParser(description="JWEL AMS CLI", add_help=False)
    parser.add_argument("command", nargs="?", help="Command: capture, encode, system, run, admin, status, clean")
    parser.add_argument("--help", "-h", action="store_true", help="Show help")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.help:
        print_banner()
        print_status()
        return

    if not args.command:
        print_banner()
        print_status()
        return

    command = resolve_command(args.command.lower())

    if command == "status":
        print_banner()
        print_status()

    elif command == "clean":
        clean_all()

    elif command in SCRIPTS:
        success = run_script(command)
        if success:
            print(f"\n✅ {command.upper()} completed!")
            print_status()
        else:
            print(f"\n❌ {command.upper()} failed!")
            sys.exit(1)

    else:
        print(f"❌ Unknown command: {args.command}")
        print("Valid: capture, encode, system, run, admin, status, clean")
        sys.exit(1)


if __name__ == "__main__":
    main()