# VeriFact Deployment Guide

This guide is for running the web app in a browser after cloning the repository.

## 1. Quick Start (Windows PowerShell)

Run all commands from the project root (the folder that contains `app.py`, `templates/`, `static/`, `verifact/`, and `outputs/`).

### 1.1 Clone and open the repo

```powershell
git clone https://github.com/<owner>/<repo>.git
cd <repo>
```

### 1.2 Create and activate virtual environment

Use Python 3.10 or 3.11. Recommended: 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 1.3 Start the Flask server

```powershell
flask --app app.py run --host=127.0.0.1 --port=5000 --debug
```

### 1.4 Open in browser

Open:

- `http://127.0.0.1:5000`

Do not open `templates/index.html` directly in Live Server.

## 2. Quick Start (macOS/Linux)

```bash
git clone https://github.com/<owner>/<repo>.git
cd <repo>
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
flask --app app.py run --host=127.0.0.1 --port=5000 --debug
```

Then open:

- `http://127.0.0.1:5000`

## 3. Verify You Are In The Correct Folder

Before running install/start commands, confirm you are in repo root.

Windows:

```powershell
Get-ChildItem
```

macOS/Linux:

```bash
ls
```

You should see `app.py` in this same directory.

## 4. Required Model Artifacts

The app expects these files in `outputs/`:

- `best_clf.pkl`
- `scaler.pkl`
- `fake_news_lstm_UP.h5`
- `tokenizer_UP.pkl`
- `branch3_v2_model.pkl`
- `branch3_v2_snippet_vectorizer.pkl`
- `branch3_v2_domain_vectorizer.pkl`

Quick check:

Windows:

```powershell
Get-ChildItem .\outputs
```

macOS/Linux:

```bash
ls outputs
```

## 5. How To Use The Browser App

1. Open `http://127.0.0.1:5000`.
2. Paste a claim/headline in the text box.
3. Optionally attach an image (enables Branch 1).
4. Keep `Fast mode` enabled for quicker response in demos.
5. Click `Run Analysis`.
6. Read:
   - Final label and confidence
   - Branch cards
   - OSINT sources and signals

## 6. Common Problems And Fixes

### 6.1 `ModuleNotFoundError` for Flask/tensorflow/ddgs/etc.

You are likely not using the activated virtual environment.

Fix:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 6.2 Python 3.13 compatibility issues

Use Python 3.10/3.11 instead of 3.13.

### 6.3 Branch 3 disabled or unstable due package mismatch

Run:

```powershell
pip install --upgrade --force-reinstall numpy==1.26.4 scikit-learn==1.6.1
```

### 6.4 UI says artifacts missing

Make sure all required files in Section 4 are present under `outputs/`.

## 7. Optional Speed Tuning

For faster OSINT responses in demo mode (Windows PowerShell):

```powershell
$env:VERIFACT_FAST_DDG_TEXT_MAX_RESULTS="2"
$env:VERIFACT_FAST_DDG_RETRY_ATTEMPTS="1"
$env:VERIFACT_DDG_CACHE_TTL_SEC="1800"
flask --app app.py run --host=127.0.0.1 --port=5000 --debug
```

## 8. Diagnostics Endpoints

Use these in browser while server is running:

- `/api/health`
- `/api/diagnostics`

These show runtime/package status and Branch load diagnostics.

## 9. Stop Server And Exit Environment

- Stop Flask: `Ctrl + C`
- Deactivate environment:

```powershell
deactivate
```
