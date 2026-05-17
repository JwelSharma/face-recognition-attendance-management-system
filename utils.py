import json
from pathlib import Path

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
            settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            primary = settings.get("streamurl", default_urls[0])
            return [primary] + [u for u in default_urls if u != primary]
    except Exception:
        pass
    return default_urls