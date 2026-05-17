# JWEL AMS

A face recognition based attendance management system built with Python, InsightFace, OpenCV, Streamlit, and Google Sheets integration.

## Features

- Real-time face recognition using InsightFace (`buffalo_l`)
- Attendance logging to CSV
- Streamlit admin dashboard
- Identity dataset management
- Face encoding pipeline
- Camera-based face capture utility
- Optional Google Sheets sync

## Project Structure

```text
JWEL-AMS/
├── ams.py
├── system.py
├── admin_ui.py
├── encode_faces.py
├── capture_photos.py
├── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/jwel-ams.git
   cd jwel-ams
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   ```

   On Windows:
   ```bash
   venv\Scripts\activate
   ```

   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

Before running the project, prepare the following locally:

- `dataset/` folder with one subfolder per person
- `credentials.json` for Google Sheets integration, if used
- camera stream URL in `settings.json` or in your local utility settings
- CUDA-compatible environment if using GPU acceleration

These local/private files are intentionally not included in the public repository.

## Usage

### Run the admin dashboard

```bash
streamlit run admin_ui.py
```

### Run the attendance system

```bash
python system.py
```

### Build face encodings

```bash
python encode_faces.py
```

### Capture photos for a new identity

```bash
python capture_photos.py
```

Or pass a person name directly:

```bash
python capture_photos.py John_Doe
```

## Notes

- This repository excludes private runtime files, attendance logs, dataset images, and credentials.
- Update camera URLs and any local network settings before use.
- If GPU execution is unavailable, the system may fall back to CPU depending on the installed runtime.

## Tech Stack

- Python
- OpenCV
- InsightFace
- ONNX Runtime GPU
- Streamlit
- Plotly
- Pandas
- Google Sheets API
