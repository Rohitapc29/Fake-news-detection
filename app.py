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
import traceback
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from verifact.engine import artifact_paths

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MiB uploads

BASE_DIR = Path(os.environ.get("VERIFACT_BASE_DIR", Path(__file__).resolve().parent))
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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

        result = analyze_text_image(text, image_path=image_path, threshold=threshold, base_dir=BASE_DIR)
        result["ok"] = True
        result["input_mode"] = "text_and_image" if image_path else "text_only"
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
    return jsonify({"base_dir": str(BASE_DIR), "artifacts_ok": len(missing) == 0, "missing": missing})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=True)
