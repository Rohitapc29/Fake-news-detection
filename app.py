"""
VeriFact web UI — Flask backend serving the tri-modal detector from VeriFact_Final.ipynb.

Place trained artifacts under ./outputs/ (same filenames as notebook) or set VERIFACT_BASE_DIR.

Run (after installing requirements):
    set FLASK_APP=app.py
    flask run --debug
Or:
    python app.py
"""
from __future__ import annotations

import os
import platform
import sys
import traceback
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MiB uploads

BASE_DIR = Path(os.environ.get("VERIFACT_BASE_DIR", Path(__file__).resolve().parent))
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _runtime_info() -> dict[str, object]:
    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "allow_cpu_blip": os.environ.get("VERIFACT_ALLOW_CPU_BLIP", "0"),
    }
    packages = [
        "numpy",
        "scikit-learn",
        "joblib",
        "ddgs",
        "duckduckgo-search",
        "torch",
        "transformers",
        "tensorflow",
    ]
    info["packages"] = {pkg: _pkg_version(pkg) for pkg in packages}

    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        info["cuda_available"] = False
        info["cuda_device_count"] = 0

    return info


def artifact_paths(base: Path) -> dict[str, Path]:
    out = base / "outputs"
    return {
        "b1_clf": out / "best_clf.pkl",
        "b1_scaler": out / "scaler.pkl",
        "b2_model": out / "fake_news_lstm_UP.h5",
        "b2_tokenizer": out / "tokenizer_UP.pkl",
        "b3_model": out / "branch3_v2_model.pkl",
        "b3_snippet": out / "branch3_v2_snippet_vectorizer.pkl",
        "b3_domain": out / "branch3_v2_domain_vectorizer.pkl",
    }


@app.route("/")
def index():
    paths = artifact_paths(BASE_DIR)
    missing = [name for name, p in paths.items() if not p.exists()]
    return render_template(
        "index.html",
        base_dir=str(BASE_DIR),
        artifact_status="ready" if not missing else "incomplete",
        missing_artifacts=", ".join(missing) if missing else "",
    )


@app.route("/api/analyze", methods=["POST"])
def analyze():
    text = (request.form.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "Please enter some text to analyze."}), 400

    threshold_str = request.form.get("threshold")
    try:
        threshold = float(threshold_str) if threshold_str else 0.5
    except ValueError:
        threshold = 0.5
    threshold = max(0.0, min(1.0, threshold))

    fast_mode = str(request.form.get("fast_mode") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    image_path = None
    if "image" in request.files and request.files["image"].filename:
        f = request.files["image"]
        ext = Path(f.filename or "").suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
            return jsonify({"ok": False, "error": "Unsupported image type. Use jpg, png, webp, gif, or bmp."}), 400
        uid = uuid.uuid4().hex
        save_path = UPLOAD_DIR / f"{uid}{ext}"
        f.save(save_path)
        image_path = str(save_path)

    try:
        from verifact.engine import analyze_text_image

        result = analyze_text_image(
            text,
            image_path=image_path,
            threshold=threshold,
            base_dir=BASE_DIR,
            fast_mode=fast_mode,
        )
        result["ok"] = True
        result["input_mode"] = "text_and_image" if image_path else "text_only"
        result["fast_mode"] = fast_mode
        return jsonify(result)
    except FileNotFoundError as fnf:
        return jsonify({"ok": False, "error": str(fnf)}), 503
    except Exception:
        app.logger.exception("/api/analyze failed")
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Inference failed — see detail below or the Flask console.",
                    "detail": traceback.format_exc(),
                }
            ),
            500,
        )
    finally:
        if image_path and Path(image_path).exists():
            try:
                Path(image_path).unlink()
            except OSError:
                pass


@app.route("/api/health")
def health():
    paths = artifact_paths(BASE_DIR)
    missing = {k: str(v) for k, v in paths.items() if not v.exists()}
    return jsonify(
        {
            "base_dir": str(BASE_DIR),
            "artifacts_ok": len(missing) == 0,
            "missing": missing,
            "runtime": _runtime_info(),
            "diagnostics_endpoint": "/api/diagnostics",
        }
    )


@app.route("/api/diagnostics")
def diagnostics():
    log_path = BASE_DIR / "outputs" / "sklearn_load_debug.log"
    log_tail = ""
    if log_path.exists():
        try:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
        except OSError:
            log_tail = "Could not read diagnostics log."

    return jsonify(
        {
            "base_dir": str(BASE_DIR),
            "runtime": _runtime_info(),
            "diagnostics_log_path": str(log_path),
            "diagnostics_log_tail": log_tail,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=True)
