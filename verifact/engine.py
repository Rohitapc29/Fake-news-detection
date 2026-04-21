"""
Inference pipeline aligned with VeriFact_Final.ipynb:
  Branch 1: BLIP-2 caption + MiniLM cosine sim → sklearn clf
  Branch 2: BiLSTM on cleaned text
  Branch 3: DuckDuckGo OSINT + hybrid rules + RandomForest
  Ensemble: 0.45*p1 + 0.05*p2 + 0.5*p3 (probabilities are P(fake))
"""
from __future__ import annotations

import io
import os
import pickle
import re
import threading
import time
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import imagehash
except ImportError:  # pragma: no cover - optional runtime dependency
    imagehash = None
import joblib
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import torch

# ── Numpy 1.x/2.x compatibility patch for pickled models ──────────────
# Handle models pickled with numpy 2.x when running on numpy 1.x
if not hasattr(np, "_core"):
    np._core = np

# Aggressive patch for BitGenerator pickle compatibility
try:
    import numpy.random._pickle as np_pickle
    from numpy.random import MT19937
    from numpy.random.bit_generator import BitGenerator
    
    # Patch 1: Fix __bit_generator_ctor to handle class objects
    _original_bit_gen_ctor = getattr(np_pickle, '__bit_generator_ctor', None)
    
    def _compat_bit_generator_ctor(bit_gen_name):
        """Handle both string names and class objects from numpy 2.x pickles."""
        if isinstance(bit_gen_name, type):
            return bit_gen_name
        if isinstance(bit_gen_name, str) and 'MT19937' in str(bit_gen_name):
            return MT19937
        if _original_bit_gen_ctor:
            return _original_bit_gen_ctor(bit_gen_name)
        raise ValueError(f'{bit_gen_name} is not a known BitGenerator module.')
    
    np_pickle.__bit_generator_ctor = _compat_bit_generator_ctor
    
    # Patch 2: Fix BitGenerator.__setstate__ to handle tuple state format
    _original_setstate = BitGenerator.__setstate__
    
    def _compat_setstate(self, state):
        """Handle both dict (1.x) and tuple (2.x) state formats."""
        if isinstance(state, dict):
            _original_setstate(self, state)
        elif isinstance(state, tuple) and len(state) > 0:
            # Extract dict from tuple
            state_dict = state[0] if isinstance(state[0], dict) else {}
            if state_dict:
                _original_setstate(self, state_dict)
            else:
                # If no dict found, skip setstate to avoid error
                pass
        else:
            # Empty tuple or other - skip
            pass
    
    BitGenerator.__setstate__ = _compat_setstate
except Exception:
    pass

try:
    from ddgs import DDGS  # noqa: ICN003
except ImportError:
    from duckduckgo_search import DDGS  # type: ignore[no-redef]
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings("ignore")

# Branch 2 sequence length (must match tokenizer padding and fake_news_lstm.h5 training).
B2_MAX_LEN = 100


def _load_branch2_bilstm_from_weights(path: str) -> Any:
    """
    Recreate the IFND BiLSTM from VeriFact_Final / ifndtest and load weights only.
    Full HDF5 saves from Keras 2 often fail on Keras 3 (InputLayer batch_shape, Embedding extras, etc.).
    Architecture matches the notebook checkpoint under outputs/fake_news_lstm.h5.
    """
    m = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(20000, 128),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="branch2_bilstm",
    )
    m.build((None, B2_MAX_LEN))
    m.load_weights(path)
    return m


def _load_keras2_h5(path: str) -> Any:
    """Prefer native load_model; fall back to fixed architecture + load_weights for Keras-3 stacks."""
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as native_err:
        try:
            return _load_branch2_bilstm_from_weights(path)
        except Exception as weight_err:
            raise RuntimeError(
                "Could not load Branch 2 BiLSTM (outputs/fake_news_lstm.h5). "
                "Tried keras.load_model then the notebook architecture + load_weights. "
                f"load_model error: {native_err!r}; rebuild error: {weight_err!r}"
            ) from weight_err


ENSEMBLE_W1, ENSEMBLE_W2, ENSEMBLE_W3 = 0.45, 0.05, 0.5
B1_CAPTION_MODEL_ID = os.environ.get(
    "VERIFACT_B1_CAPTION_MODEL",
    "Salesforce/blip-image-captioning-base",
)
SIM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


DDG_TEXT_MAX_RESULTS = _env_int("VERIFACT_DDG_TEXT_MAX_RESULTS", 5)
DDG_IMAGE_MAX_RESULTS = _env_int("VERIFACT_DDG_IMAGE_MAX_RESULTS", 1)
DDG_RETRY_ATTEMPTS = _env_int("VERIFACT_DDG_RETRY_ATTEMPTS", 3)
DDG_RETRY_DELAY_SEC = _env_float("VERIFACT_DDG_RETRY_DELAY_SEC", 0.35)

# Fast mode is meant for live demos where response time matters more than depth.
FAST_DDG_TEXT_MAX_RESULTS = _env_int("VERIFACT_FAST_DDG_TEXT_MAX_RESULTS", 3)
FAST_DDG_RETRY_ATTEMPTS = _env_int("VERIFACT_FAST_DDG_RETRY_ATTEMPTS", 1)
FAST_SKIP_IMAGE_HASH = os.environ.get("VERIFACT_FAST_SKIP_IMAGE_HASH", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Cache DDG text responses to avoid repeated network latency on similar queries.
DDG_CACHE_TTL_SEC = _env_float("VERIFACT_DDG_CACHE_TTL_SEC", 900.0)
DDG_CACHE_MAX_SIZE = _env_int("VERIFACT_DDG_CACHE_MAX_SIZE", 200)
B1_ALLOW_SIMILARITY_FALLBACK = _env_bool("VERIFACT_B1_ALLOW_SIMILARITY_FALLBACK", True)
SINGLE_BRANCH_FAKE_THRESHOLD = _env_float("VERIFACT_SINGLE_BRANCH_FAKE_THRESHOLD", 0.62)
B3_RF_EVIDENCE_FLOOR = _env_float("VERIFACT_B3_RF_EVIDENCE_FLOOR", 0.35)
SATIRE_DOMAIN_BLACKLIST = {
    "theonion.com",
    "babylonbee.com",
    "clickhole.com",
    "duffelblog.com",
    "waterfordwhispersnews.com",
    "thebeaverton.com",
    "newsthump.com",
    "fakingnews.com",
}
SATIRE_MIN_MATCH_HITS = _env_int("VERIFACT_SATIRE_MIN_MATCH_HITS", 1)
SATIRE_TOKEN_MATCH_THRESHOLD = _env_float("VERIFACT_SATIRE_TOKEN_MATCH_THRESHOLD", 0.6)
SATIRE_OVERRIDE_PROB_FAKE_SINGLE = _env_float("VERIFACT_SATIRE_OVERRIDE_PROB_FAKE_SINGLE", 0.92)
SATIRE_OVERRIDE_PROB_FAKE_MULTI = _env_float("VERIFACT_SATIRE_OVERRIDE_PROB_FAKE_MULTI", 0.98)
STRICT_CONSENSUS_OVERLAP = _env_float("VERIFACT_STRICT_CONSENSUS_OVERLAP", 0.75)
STRICT_CONSENSUS_MIN_RESULTS = _env_int("VERIFACT_STRICT_CONSENSUS_MIN_RESULTS", 2)
B3_LOW_CORROBORATION_MIN_HITS = _env_int("VERIFACT_B3_LOW_CORROBORATION_MIN_HITS", 3)
B3_LOW_CORROBORATION_MIN_TOKENS = _env_int("VERIFACT_B3_LOW_CORROBORATION_MIN_TOKENS", 8)
B3_LOW_CORROBORATION_FAKE_PROB = _env_float("VERIFACT_B3_LOW_CORROBORATION_FAKE_PROB", 0.82)


def _domain_in_blacklist(domain: str, blacklist: set[str]) -> bool:
    d = str(domain or "").strip().lower().lstrip(".")
    if not d:
        return False
    return any(d == b or d.endswith(f".{b}") for b in blacklist)


def _load_joblib_with_pickle_fallback(path: Path, label: str, debug_log: list[str] | None = None) -> Any:
    """
    Load a pickled artifact with a compatibility fallback for NumPy BitGenerator state issues.
    """
    try:
        return joblib.load(path)
    except TypeError as exc:
        err_txt = str(exc)
        if "BitGenerator" not in err_txt or "__setstate__" not in err_txt:
            raise
        if debug_log is not None:
            debug_log.append(f"{label} BitGenerator compatibility fallback engaged")
        with open(path, "rb") as fh:
            return pickle.load(fh)


def _base_dir() -> Path:
    return Path(os.environ.get("VERIFACT_BASE_DIR", Path(__file__).resolve().parent.parent))


def artifact_paths(base: Path | None = None) -> dict[str, Path]:
    root = base or _base_dir()
    out = root / "outputs"
    return {
        "b1_clf": out / "best_clf.pkl",
        "b1_scaler": out / "scaler.pkl",
        "b2_model": out / "fake_news_lstm_UP.h5",
        "b2_tokenizer": out / "tokenizer_UP.pkl",
        "b3_model": out / "branch3_v2_model.pkl",
        "b3_snippet": out / "branch3_v2_snippet_vectorizer.pkl",
        "b3_domain": out / "branch3_v2_domain_vectorizer.pkl",
    }


class VeriFactEngine:
    """Lazy-loaded singleton for all models."""

    _instance: VeriFactEngine | None = None
    _lock = threading.Lock()

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or _base_dir()
        paths = artifact_paths(self.base_dir)
        self.paths = paths
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._b1_caption_model_id = B1_CAPTION_MODEL_ID
        self._b1_allow_similarity_fallback = B1_ALLOW_SIMILARITY_FALLBACK
        self._allow_cpu_blip = os.environ.get("VERIFACT_ALLOW_CPU_BLIP", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }

        self._tf_ready = False
        self._b2_model = None
        self._b2_tokenizer = None

        self._torch_ready = False
        self._blip2_processor = None
        self._blip2_model = None
        self._sim_model = None

        self._sk_ready = False
        self._b1_clf = None
        self._b1_scaler = None
        self._b3_model = None
        self._b3_snippet_vocab = None
        self._b3_domain_vocab = None
        self._ddgs: DDGS | None = None
        self._b1_available = False  # Track if Branch 1 sklearn files exist
        self._b3_available = False  # Track if Branch 3 sklearn files exist
        self._b1_load_error: str | None = None
        self._b3_load_error: str | None = None
        self._b1_runtime_error: str | None = None
        self._ddgs_error: str | None = None
        self._ddg_text_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    @classmethod
    def get(cls, base_dir: Path | None = None) -> VeriFactEngine:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(base_dir)
            return cls._instance

    def ensure_tf(self) -> None:
        if self._tf_ready:
            return
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        path = self.paths["b2_model"]
        tok_path = self.paths["b2_tokenizer"]
        if not path.exists():
            raise FileNotFoundError(f"Missing Branch 2 model: {path}")
        if not tok_path.exists():
            raise FileNotFoundError(f"Missing tokenizer: {tok_path}")
        self._b2_model = _load_keras2_h5(str(path))
        with open(tok_path, "rb") as f:
            self._b2_tokenizer = pickle.load(f)
        self._tf_ready = True

    def ensure_torch_branch1(self) -> None:
        if self._torch_ready:
            return

        if self.device != "cuda" and not self._allow_cpu_blip:
            self._b1_runtime_error = (
                "Branch 1 caption model is disabled on CPU-only runtime for reliability. "
                "Set VERIFACT_ALLOW_CPU_BLIP=1 to force CPU inference (slow), "
                "or deploy on GPU."
            )
            return

        model_id = str(self._b1_caption_model_id)
        use_blip2 = "blip2" in model_id.lower()

        if use_blip2:
            from transformers import Blip2ForConditionalGeneration as CaptionModelClass
            from transformers import Blip2Processor as CaptionProcessorClass
        else:
            from transformers import BlipForConditionalGeneration as CaptionModelClass
            from transformers import BlipProcessor as CaptionProcessorClass
        from sentence_transformers import SentenceTransformer

        print(f"Loading Branch 1 caption model: {model_id}...")
        try:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._blip2_processor = CaptionProcessorClass.from_pretrained(model_id)
            self._blip2_model = CaptionModelClass.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
            )
            self._blip2_model.eval()
            if self.device != "cuda":
                self._blip2_model.to(self.device)
            self._sim_model = SentenceTransformer(SIM_MODEL_ID, device=self.device)
            self._b1_runtime_error = None
            self._torch_ready = True
        except Exception as exc:
            self._b1_runtime_error = (
                f"Branch 1 caption model init failed for '{model_id}' ({type(exc).__name__}): {exc}. "
                "Deploy on a GPU node with sufficient VRAM, or temporarily disable Branch 1."
            )
            self._torch_ready = False

    def _init_ddgs(self, debug_log: list[str] | None = None) -> None:
        try:
            try:
                self._ddgs = DDGS(timeout=10)
            except TypeError:
                # Older implementations may not support timeout argument.
                self._ddgs = DDGS()
            self._ddgs_error = None
            if debug_log is not None:
                debug_log.append("DDGS initialized")
        except Exception as exc:
            self._ddgs = None
            self._ddgs_error = f"{type(exc).__name__}: {exc}"
            if debug_log is not None:
                debug_log.append(f"DDGS init error: {self._ddgs_error}")

    @staticmethod
    def _normalize_query(query: str) -> str:
        return re.sub(r"\s+", " ", str(query).strip().lower())

    def _ddg_cache_get(self, query: str, max_results: int) -> list[dict[str, Any]] | None:
        key = self._normalize_query(query)
        cached = self._ddg_text_cache.get(key)
        if not cached:
            return None
        ts, rows = cached
        if (time.time() - ts) > DDG_CACHE_TTL_SEC:
            self._ddg_text_cache.pop(key, None)
            return None
        return list(rows[:max_results])

    def _ddg_cache_set(self, query: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        key = self._normalize_query(query)
        self._ddg_text_cache[key] = (time.time(), rows)
        if len(self._ddg_text_cache) <= DDG_CACHE_MAX_SIZE:
            return
        oldest_key = min(self._ddg_text_cache.items(), key=lambda item: item[1][0])[0]
        self._ddg_text_cache.pop(oldest_key, None)

    def _ddg_text_search(
        self,
        query: str,
        max_results: int,
        retry_attempts: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        errors: list[str] = []
        attempt_limit = max(1, retry_attempts or DDG_RETRY_ATTEMPTS)

        cached = self._ddg_cache_get(query, max_results=max_results)
        if cached is not None:
            return cached, errors

        for attempt in range(1, attempt_limit + 1):
            if not self._ddgs:
                self._init_ddgs()
            if not self._ddgs:
                errors.append(f"attempt {attempt}: {self._ddgs_error or 'DDGS unavailable'}")
                time.sleep(DDG_RETRY_DELAY_SEC)
                continue
            try:
                results = list(self._ddgs.text(query, max_results=max_results))
                self._ddg_cache_set(query, results)
                return results, errors
            except Exception as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                self._ddgs = None
                time.sleep(DDG_RETRY_DELAY_SEC)
        return [], errors

    def _ddg_image_search(
        self,
        query: str,
        max_results: int,
        retry_attempts: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        errors: list[str] = []
        attempt_limit = max(1, retry_attempts or DDG_RETRY_ATTEMPTS)

        for attempt in range(1, attempt_limit + 1):
            if not self._ddgs:
                self._init_ddgs()
            if not self._ddgs:
                errors.append(f"attempt {attempt}: {self._ddgs_error or 'DDGS unavailable'}")
                time.sleep(DDG_RETRY_DELAY_SEC)
                continue
            try:
                results = list(self._ddgs.images(query, max_results=max_results))
                return results, errors
            except Exception as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                self._ddgs = None
                time.sleep(DDG_RETRY_DELAY_SEC)
        return [], errors

    def ensure_sklearn_branches(self) -> None:
        if self._sk_ready:
            return
        
        debug_log = []
        
        # Try to load Branch 1 sklearn files (optional)
        self._b1_available = False
        self._b1_load_error = None
        try:
            if self.paths["b1_clf"].exists() and self.paths["b1_scaler"].exists():
                debug_log.append("Starting B1 loads...")
                self._b1_clf = _load_joblib_with_pickle_fallback(
                    self.paths["b1_clf"],
                    "B1 classifier",
                    debug_log,
                )
                debug_log.append("B1 clf loaded OK")
                self._b1_scaler = _load_joblib_with_pickle_fallback(
                    self.paths["b1_scaler"],
                    "B1 scaler",
                    debug_log,
                )
                debug_log.append("B1 scaler loaded OK")
                
                self._b1_available = True
                debug_log.append("✓ B1 LOADED")
            else:
                missing = [
                    str(self.paths["b1_clf"].name) if not self.paths["b1_clf"].exists() else "",
                    str(self.paths["b1_scaler"].name) if not self.paths["b1_scaler"].exists() else "",
                ]
                missing_txt = ", ".join([m for m in missing if m])
                self._b1_load_error = f"Missing Branch 1 artifacts: {missing_txt}"
                debug_log.append(self._b1_load_error)
        except Exception as e:
            import traceback
            debug_log.append(f"B1 LOAD FAILED: {str(e)}")
            debug_log.append(traceback.format_exc())
            self._b1_load_error = f"B1 artifact load failed: {type(e).__name__}: {e}"
        
        # Try to load Branch 3 sklearn files (optional)
        self._b3_available = False
        self._b3_load_error = None
        try:
            debug_log.append(f"B3 model exists: {self.paths['b3_model'].exists()}")
            debug_log.append(f"B3 snippet exists: {self.paths['b3_snippet'].exists()}")
            debug_log.append(f"B3 domain exists: {self.paths['b3_domain'].exists()}")
            
            if (self.paths["b3_model"].exists() and 
                self.paths["b3_snippet"].exists() and 
                self.paths["b3_domain"].exists()):
                debug_log.append("Starting B3 loads...")
                self._b3_model = joblib.load(self.paths["b3_model"])
                debug_log.append("B3 model loaded OK")
                self._b3_snippet_vocab = joblib.load(self.paths["b3_snippet"])
                debug_log.append("B3 snippet loaded OK")
                self._b3_domain_vocab = joblib.load(self.paths["b3_domain"])
                debug_log.append("B3 domain loaded OK")
                self._b3_available = True
            else:
                missing = [
                    str(self.paths["b3_model"].name) if not self.paths["b3_model"].exists() else "",
                    str(self.paths["b3_snippet"].name) if not self.paths["b3_snippet"].exists() else "",
                    str(self.paths["b3_domain"].name) if not self.paths["b3_domain"].exists() else "",
                ]
                missing_txt = ", ".join([m for m in missing if m])
                self._b3_load_error = f"Missing Branch 3 artifacts: {missing_txt}"
                debug_log.append(self._b3_load_error)
        except Exception as e:
            import traceback
            debug_log.append(f"B3 LOAD FAILED: {str(e)}")
            debug_log.append(traceback.format_exc())
            self._b3_load_error = f"B3 artifact load failed: {type(e).__name__}: {e}"
        
        self._init_ddgs(debug_log)

        # Write debug log to file after DDGS init so runtime state is captured in one place.
        try:
            log_path = self.base_dir / "outputs" / "sklearn_load_debug.log"
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(debug_log))
        except OSError:
            pass
        
        self._sk_ready = True

    def load_all(self, use_branch1_vision: bool = True) -> None:
        self.ensure_tf()
        if use_branch1_vision:
            self.ensure_torch_branch1()
        self.ensure_sklearn_branches()

    @staticmethod
    def b2_clean_text(text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text).lower()
        for pattern in [
            r"^fact check[:-]?\s*",
            r"^viral[:-]?\s*",
            r"^fake[:-]?\s*",
        ]:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text.strip()

    @torch.no_grad()
    def _b1_generate_caption(self, image: Image.Image | None) -> str:
        if image is None:
            return ""
        try:
            inputs = self._blip2_processor(images=image, return_tensors="pt")
            inputs = inputs.to(self.device)
            pv = inputs["pixel_values"]
            pv = pv.to(dtype=torch.float16 if self.device == "cuda" else torch.float32)
            inputs["pixel_values"] = pv
            out = self._blip2_model.generate(**inputs, max_new_tokens=40)
            return self._blip2_processor.decode(out[0], skip_special_tokens=True).strip()
        except Exception:
            return ""

    @staticmethod
    def _b1_similarity_to_prob_fake(similarity: float) -> float:
        # High caption-text similarity generally indicates stronger consistency, thus lower fake probability.
        return float(np.clip(0.5 - (0.6 * similarity), 0.05, 0.95))

    @staticmethod
    def _friendly_b1_load_error(raw_error: str | None) -> str:
        if not raw_error:
            return "Branch 1 classifier artifact could not be loaded in this environment."

        txt = str(raw_error).lower()
        if "bitgenerator" in txt or "__setstate__" in txt or "legacy mt19937 state" in txt:
            return (
                "Legacy Branch 1 classifier artifact is not compatible with the current Python/NumPy stack."
            )

        if "inconsistentversionwarning" in txt or "version" in txt:
            return "Branch 1 classifier artifact version does not match current sklearn runtime."

        return "Branch 1 classifier artifact could not be loaded in this environment."

    def branch1_detailed(self, image_path: str | None, text: str) -> dict[str, Any]:
        detail: dict[str, Any] = {
            "branch": "Branch 1 — Vision (caption + similarity)",
            "image_provided": bool(image_path),
            "caption_model": self._b1_caption_model_id,
            "fallback_enabled": self._b1_allow_similarity_fallback,
        }
        if not image_path:
            detail["disabled"] = True
            detail["reason"] = "No image provided; vision branch is not run and gets no ensemble weight."
            return detail

        if not self._torch_ready:
            detail["disabled"] = True
            detail["reason"] = self._b1_runtime_error or "Branch 1 caption model runtime unavailable."
            detail["prob_fake"] = 0.5
            return detail

        if not self._b1_available and not self._b1_allow_similarity_fallback:
            detail["disabled"] = True
            detail["reason"] = self._b1_load_error or "Branch 1 sklearn artifacts unavailable."
            detail["prob_fake"] = 0.5
            return detail

        if not self._b1_available and self._b1_allow_similarity_fallback:
            detail["warning"] = (
                "Branch 1 sklearn artifacts could not be loaded. "
                "Using similarity fallback probability for this request."
            )
            detail["fallback_reason"] = self._friendly_b1_load_error(self._b1_load_error)
            detail["fallback_reason_debug"] = self._b1_load_error or "Unknown sklearn load error."

        from sentence_transformers import util

        try:
            img = Image.open(image_path).convert("RGB")
            caption = self._b1_generate_caption(img)
            detail["caption"] = caption
            emb_c = self._sim_model.encode(caption, convert_to_tensor=True)
            emb_t = self._sim_model.encode(str(text), convert_to_tensor=True)
            sim = float(util.cos_sim(emb_c, emb_t).item())
            detail["cosine_similarity_caption_vs_text"] = round(sim, 6)
            wc = len(caption.split()) if caption else 0
            detail["caption_word_count"] = wc

            if self._b1_available:
                detail["classifier_mode"] = "artifact_classifier"
                feats = self._b1_scaler.transform([[sim, wc]])
                detail["scaled_features"] = [round(float(feats[0][0]), 6), float(wc)]
                prob_fake = float(self._b1_clf.predict_proba(feats)[0][1])
            else:
                detail["classifier_mode"] = "similarity_fallback"
                prob_fake = self._b1_similarity_to_prob_fake(sim)

            detail["prob_fake"] = round(prob_fake, 4)
        except Exception as exc:
            detail["error"] = str(exc)
            detail["prob_fake"] = 0.5
        return detail

    def branch2_detailed(self, text: str) -> dict[str, Any]:
        detail: dict[str, Any] = {"branch": "Branch 2 — BiLSTM (IFND-style text)"}
        try:
            clean = self.b2_clean_text(text)
            detail["cleaned_text_preview"] = clean[:400] + ("…" if len(clean) > 400 else "")
            seq = self._b2_tokenizer.texts_to_sequences([clean])
            padded = pad_sequences(seq, maxlen=B2_MAX_LEN, padding="post")
            lstm_real_score = float(self._b2_model.predict(padded, verbose=0)[0][0])
            detail["lstm_output_prob_real"] = round(lstm_real_score, 4)
            prob_fake = float(1.0 - lstm_real_score)
            detail["prob_fake"] = round(prob_fake, 4)
        except Exception as exc:
            detail["error"] = str(exc)
            detail["prob_fake"] = 0.5
        return detail

    def branch3_detailed(self, text: str, image_path: str | None, fast_mode: bool = False) -> dict[str, Any]:
        detail: dict[str, Any] = {"branch": "Branch 3 — OSINT + Random Forest"}

        text_max_results = FAST_DDG_TEXT_MAX_RESULTS if fast_mode else DDG_TEXT_MAX_RESULTS
        ddg_retry_attempts = FAST_DDG_RETRY_ATTEMPTS if fast_mode else DDG_RETRY_ATTEMPTS
        skip_image_hash = fast_mode and FAST_SKIP_IMAGE_HASH
        detail["fast_mode"] = fast_mode
        detail["ddg_budget"] = {
            "max_text_results": text_max_results,
            "retry_attempts": ddg_retry_attempts,
            "skip_image_hash": skip_image_hash,
        }
        
        if not self._b3_available:
            detail["disabled"] = True
            detail["reason"] = self._b3_load_error or "Branch 3 sklearn artifacts unavailable."
            detail["prob_fake"] = 0.5
            return detail
        
        if not self._ddgs:
            self._init_ddgs()

        if not self._ddgs:
            detail["disabled"] = True
            detail["reason"] = (
                "DuckDuckGo search not available "
                f"(DDGS init failed: {self._ddgs_error or 'unknown'})."
            )
            detail["prob_fake"] = 0.5
            return detail

        debunk_words = [
            "false","fake","hoax","debunked","misleading",
            "altered","satire","scam","unsupported","unproven"
        ]

        debunk_phrases = [
            "no evidence",
            "not true",
            "false claim",
            "does not cause",
            "not caused by",
            "no link",
            "no proven link",
            "rumor",
            "myth",
        ]

        true_words = ["true","correct","accurate","authentic","verified","real"]
        fact_check_phrases = ["fact check", "fact-check", "factcheck", "verified by", "verification"]

        try:
            # ================================
            # SMART QUERY
            # ================================
            clean_text = re.sub(r'http\S+', '', str(text))
            clean_text = re.sub(r'[^a-zA-Z0-9\s"\'\-]', ' ', clean_text)

            stopwords = {
                "a","an","the","is","are","was","were","in","on","at","to","for",
                "of","who","that","this","it","with","and","by","from"
            }

            words = [w for w in clean_text.split() if w.lower() not in stopwords]
            headline = " ".join(words[:10])
            if not headline:
                headline = re.sub(r"\s+", " ", str(text)).strip()[:120] or "latest news claim"
            detail["search_query_headline"] = headline
            query_variants = [headline]
            if len(headline.split()) >= 4:
                quoted = f'"{headline}"'
                if quoted not in query_variants:
                    query_variants.append(quoted)
            detail["search_query_variants"] = query_variants

            # ================================
            # SEARCH + DOMAIN LOGIC
            # ================================
            snippets_combined, domains_combined = "", ""
            agreeing_results = 0
            strong_agreeing_results = 0

            # Try an exact phrase variant if the primary query is weak, then merge unique results.
            results: list[dict[str, Any]] = []
            ddg_errors: list[str] = []
            seen_result_ids: set[str] = set()
            for idx, query_text in enumerate(query_variants):
                variant_results, variant_errors = self._ddg_text_search(
                    query_text,
                    max_results=text_max_results,
                    retry_attempts=ddg_retry_attempts,
                )
                if variant_errors:
                    ddg_errors.extend([f"{query_text} :: {err}" for err in variant_errors])

                for item in variant_results:
                    href = str(item.get("href", "")).strip().lower()
                    if href:
                        result_id = href
                    else:
                        title = str(item.get("title", "")).strip().lower()
                        body = str(item.get("body", "")).strip().lower()[:120]
                        result_id = f"{title}::{body}"

                    if result_id in seen_result_ids:
                        continue
                    seen_result_ids.add(result_id)
                    results.append(item)
                    if len(results) >= text_max_results:
                        break

                if len(results) >= text_max_results:
                    break
                # If the first query already returned enough material, skip the fallback query.
                if idx == 0 and len(results) >= max(2, text_max_results // 2):
                    break

            detail["ddg_text_hits"] = len(results)
            if ddg_errors:
                detail["ddg_errors"] = ddg_errors

            preview_snips = []
            preview_domains = []
            preview_links = []
            seen_domains = set()

            # Measure source consensus directly from retrieved text instead of hardcoding domain trust.
            query_noise_tokens = {
                "urgent", "msg", "plz", "please", "fwd", "forward", "share", "contacts",
                "contact", "immediately", "today", "morning", "breaking", "alert", "indians",
                "india", "announcement", "announce", "tv", "save", "life", "everyone", "all",
            }
            query_tokens: list[str] = []
            for w in words:
                token = str(w).lower().strip()
                if len(token) < 3 or token in query_noise_tokens:
                    continue
                if token not in query_tokens:
                    query_tokens.append(token)
            query_tokens = query_tokens[:24]
            query_token_set = set(query_tokens)
            headline_norm = re.sub(r"\s+", " ", headline.lower()).strip()
            satire_match_hits = 0
            satire_domains_found: set[str] = set()
            strict_agreeing_results = 0
            contradicting_results = 0
            strict_overlap_threshold = min(1.0, max(0.0, STRICT_CONSENSUS_OVERLAP))

            claim_text_lower = str(text).lower()
            forwarding_markers = ["plz", "please", "fwd", "forward", "share", "urgent", "alert"]
            risk_terms = {
                "hiv", "aids", "virus", "viral", "blood", "infect", "infection",
                "contaminated", "poison", "toxic", "cancer", "dies", "death",
            }
            has_forwarding_marker = any(marker in claim_text_lower for marker in forwarding_markers)
            has_risk_term = any(term in query_token_set for term in risk_terms)

            for res in results:
                body = str(res.get("body", ""))
                href = res.get("href", "")
                title = str(res.get("title", "")).strip()
                combined_text = f"{title} {body}".lower()

                dom = urlparse(href).netloc.replace("www.", "").lower()

                snippets_combined += combined_text + " || "

                if any(dw in combined_text for dw in debunk_words) or any(
                    phrase in combined_text for phrase in debunk_phrases
                ):
                    contradicting_results += 1

                if isinstance(href, str) and href.startswith(("http://", "https://")):
                    preview_links.append(
                        {
                            "title": title,
                            "url": href,
                            "domain": dom,
                            "snippet": body[:200],
                        }
                    )

                overlap_ratio = 0.0
                if query_token_set:
                    overlap = sum(1 for t in query_token_set if t in combined_text)
                    overlap_ratio = overlap / max(1, len(query_token_set))
                    if overlap_ratio >= 0.45:
                        agreeing_results += 1
                    if overlap_ratio >= 0.70:
                        strong_agreeing_results += 1
                    if overlap_ratio >= strict_overlap_threshold:
                        strict_agreeing_results += 1

                # Satire override should only trigger if headline/text materially matches the claim.
                if _domain_in_blacklist(dom, SATIRE_DOMAIN_BLACKLIST):
                    headline_exact_match = bool(headline_norm and headline_norm in combined_text)
                    satire_threshold = min(1.0, max(0.0, SATIRE_TOKEN_MATCH_THRESHOLD))
                    headline_similar_match = headline_exact_match or overlap_ratio >= satire_threshold
                    if headline_similar_match and dom not in satire_domains_found:
                        satire_domains_found.add(dom)
                        satire_match_hits += 1

                if dom and dom not in seen_domains:
                    domains_combined += dom + " "
                    preview_snips.append(body[:150])
                    preview_domains.append(dom)
                    seen_domains.add(dom)

            detail["result_snippets_preview"] = preview_snips
            detail["result_domains"] = preview_domains
            detail["result_links"] = preview_links

            unique_domains = len(preview_domains)
            text_hits = len(results)
            consensus_ratio = (agreeing_results / text_hits) if text_hits > 0 else 0.0
            strict_consensus_ratio = (strict_agreeing_results / text_hits) if text_hits > 0 else 0.0

            # ================================
            # HYBRID OVERRIDE
            # ================================
            has_fc = any(phrase in snippets_combined for phrase in fact_check_phrases)
            has_debunk = any(dw in snippets_combined for dw in debunk_words) or any(
                phrase in snippets_combined for phrase in debunk_phrases
            )
            has_true = any(tw in snippets_combined for tw in true_words)
            strict_min_results = max(1, STRICT_CONSENSUS_MIN_RESULTS)

            detail["signals"] = {
                "supporting_sources": agreeing_results,
                "contradicting_sources": contradicting_results,
                "text_hits": text_hits,
                "agreeing_results": agreeing_results,
                "strong_agreeing_results": strong_agreeing_results,
                "strict_agreeing_results": strict_agreeing_results,
                "unique_domains": unique_domains,
                "consensus_ratio": round(consensus_ratio, 4),
                "strict_consensus_ratio": round(strict_consensus_ratio, 4),
                "query_token_count": len(query_token_set),
                "fact_checker_hit": has_fc,
                "debunk_language": has_debunk,
                "verification_language": has_true,
                "satire_blacklist_hits": satire_match_hits,
                "satire_blacklist_domains": sorted(satire_domains_found),
                "satire_blacklist_triggered": satire_match_hits >= SATIRE_MIN_MATCH_HITS,
                # Backward-compatible keys used by some older templates.
                "trusted_sources": agreeing_results,
                "suspicious_sources": contradicting_results,
                "fact_checker_domain_hit": has_fc,
                "debunk_language_in_snippets": has_debunk,
                "verification_language_in_snippets": has_true,
            }

            if has_fc and has_debunk:
                detail["decision_path"] = "Override → fact-check debunk"
                detail["prob_fake"] = 0.95
                return detail

            if satire_match_hits >= SATIRE_MIN_MATCH_HITS:
                detail["decision_path"] = "Override → satire blacklist"
                satire_prob = (
                    SATIRE_OVERRIDE_PROB_FAKE_MULTI
                    if satire_match_hits >= 2
                    else SATIRE_OVERRIDE_PROB_FAKE_SINGLE
                )
                detail["prob_fake"] = round(float(np.clip(satire_prob, 0.5, 0.999)), 4)
                return detail

            low_corroboration = (
                text_hits >= max(1, B3_LOW_CORROBORATION_MIN_HITS)
                and len(query_token_set) >= max(1, B3_LOW_CORROBORATION_MIN_TOKENS)
                and strict_agreeing_results == 0
                and strict_consensus_ratio < 0.2
                and contradicting_results == 0
            )
            if low_corroboration and has_forwarding_marker and has_risk_term:
                detail["decision_path"] = "Override → unsupported viral scare claim"
                detail["prob_fake"] = round(float(np.clip(B3_LOW_CORROBORATION_FAKE_PROB, 0.5, 0.999)), 4)
                return detail

            # If multiple sources independently align with the query and no debunk signal appears,
            # treat it as verification-style evidence.
            if (
                text_hits >= strict_min_results
                and unique_domains >= strict_min_results
                and strict_agreeing_results >= strict_min_results
                and contradicting_results == 0
                and not has_debunk
            ):
                if strict_agreeing_results == text_hits or strict_consensus_ratio >= 0.85:
                    detail["decision_path"] = "Override → multi-source consensus"
                    detail["prob_fake"] = 0.08
                else:
                    detail["decision_path"] = "Override → partial source consensus"
                    detail["prob_fake"] = 0.2
                return detail

            if (
                has_fc
                and has_true
                and not has_debunk
                and contradicting_results == 0
                and strict_consensus_ratio >= 0.5
            ):
                detail["decision_path"] = "Override → fact-check verified"
                detail["prob_fake"] = 0.05
                return detail

            # ================================
            # RANDOM FOREST FALLBACK
            # ================================
            detail["decision_path"] = "RandomForest fallback"

            live_snippet_features = self._b3_snippet_vocab.transform([snippets_combined]).toarray()
            live_domain_features = self._b3_domain_vocab.transform([domains_combined]).toarray()

            hash_dist, has_exif = -1, 0

            if image_path and imagehash is not None and not skip_image_hash:
                try:
                    local_hash = imagehash.phash(Image.open(image_path))
                    ddg_imgs, ddg_img_errors = self._ddg_image_search(
                        headline,
                        max_results=DDG_IMAGE_MAX_RESULTS,
                        retry_attempts=ddg_retry_attempts,
                    )
                    if ddg_img_errors:
                        detail["ddg_image_errors"] = ddg_img_errors
                    if ddg_imgs:
                        resp = requests.get(
                            ddg_imgs[0]["image"],
                            timeout=5,
                            headers={"User-Agent": "VeriFact/1.0 (+osint-image-check)"},
                        )
                        resp.raise_for_status()
                        hash_dist = local_hash - imagehash.phash(Image.open(io.BytesIO(resp.content)))
                except Exception:
                    pass
            elif image_path and skip_image_hash:
                detail["image_hash_skipped_reason"] = "Fast mode enabled; OSINT image hash check skipped."
            elif image_path and imagehash is None:
                detail["image_hash_skipped_reason"] = "ImageHash package not installed; OSINT image hash check skipped."

            fc_count = sum(snippets_combined.count(p) for p in fact_check_phrases)
            debunk_count = sum(snippets_combined.count(w) for w in debunk_words)
            legit_count = sum(snippets_combined.count(w) for w in true_words)

            X_live = np.hstack((
                live_snippet_features,
                live_domain_features,
                [[hash_dist]],
                [[has_exif]],
                [[fc_count]],
                [[debunk_count]],
                [[legit_count]]
            ))

            classes = list(self._b3_model.classes_)
            fake_idx = classes.index(0)

            raw_prob_fake = float(self._b3_model.predict_proba(X_live)[0][fake_idx])

            evidence_strength = (
                0.25 * min(1.0, text_hits / max(1.0, float(text_max_results)))
                + 0.35 * min(1.0, max(0.0, consensus_ratio))
                + 0.20 * min(1.0, unique_domains / 4.0)
                + 0.20 * float(has_fc or has_debunk or has_true)
            )
            evidence_floor = min(1.0, max(0.0, B3_RF_EVIDENCE_FLOOR))
            evidence_strength = min(1.0, max(evidence_floor, evidence_strength))
            prob_fake = 0.5 + ((raw_prob_fake - 0.5) * evidence_strength)

            if evidence_strength < 0.45:
                detail["decision_path"] = "RandomForest fallback (low-evidence smoothing)"

            detail["rf_raw_prob_fake"] = round(raw_prob_fake, 4)
            detail["evidence_strength"] = round(evidence_strength, 4)
            detail["prob_fake"] = round(prob_fake, 4)

        except Exception as exc:
            detail["error"] = str(exc)
            detail["prob_fake"] = 0.5

        return detail

    def analyze(
        self,
        text: str,
        image_path: str | None = None,
        threshold: float = 0.5,
        fast_mode: bool = False,
    ) -> dict[str, Any]:
        use_b1 = bool(image_path)  # Whether image was provided
        self.load_all(use_branch1_vision=use_b1)

        d1 = self.branch1_detailed(image_path, text)
        d2 = self.branch2_detailed(text)
        d3 = self.branch3_detailed(text, image_path, fast_mode=fast_mode)

        # Check which branches are actually available (not disabled)
        b1_disabled = d1.get("disabled", False)
        b3_disabled = d3.get("disabled", False)

        p2 = float(d2.get("prob_fake", 0.5))
        p1 = float(d1.get("prob_fake", 0.5)) if not b1_disabled else None
        p3 = float(d3.get("prob_fake", 0.5)) if not b3_disabled else None

        # Dynamically renormalize weights based on available branches
        if not b1_disabled and not b3_disabled:
            # All branches available - use original weights
            w1, w2, w3 = ENSEMBLE_W1, ENSEMBLE_W2, ENSEMBLE_W3
            final_score = (w1 * p1) + (w2 * p2) + (w3 * p3)
        elif b1_disabled and not b3_disabled:
            # Only B2 & B3 (no image) - renormalize
            denom = ENSEMBLE_W2 + ENSEMBLE_W3
            w1 = 0.0
            w2 = ENSEMBLE_W2 / denom
            w3 = ENSEMBLE_W3 / denom
            final_score = (w2 * p2) + (w3 * p3)
        elif not b1_disabled and b3_disabled:
            # Only B1 & B2 (B3 missing) - renormalize
            denom = ENSEMBLE_W1 + ENSEMBLE_W2
            w1 = ENSEMBLE_W1 / denom
            w2 = ENSEMBLE_W2 / denom
            w3 = 0.0
            final_score = (w1 * p1) + (w2 * p2)
        else:
            # Only B2 available - all 100% weight
            w1 = 0.0
            w2 = 1.0
            w3 = 0.0
            final_score = p2

        active_branch_count = int(not b1_disabled) + int(not b3_disabled) + 1
        effective_threshold = float(threshold)
        decision_notes: list[str] = []

        if active_branch_count == 1:
            effective_threshold = max(effective_threshold, min(1.0, SINGLE_BRANCH_FAKE_THRESHOLD))
            decision_notes.append(
                "Only one branch is active, so a stricter fake threshold was used to reduce false positives."
            )
        if b3_disabled:
            decision_notes.append(
                f"OSINT branch unavailable: {d3.get('reason', 'Branch 3 disabled.')}"
            )

        final_label = "FAKE" if final_score >= effective_threshold else "REAL"
        confidence = final_score if final_label == "FAKE" else (1.0 - final_score)

        wc1 = round(w1 * p1, 6) if p1 is not None else 0.0
        wc3 = round(w3 * p3, 6) if p3 is not None else 0.0

        return {
            "threshold": threshold,
            "effective_threshold": round(float(effective_threshold), 4),
            "fast_mode": fast_mode,
            "decision_context": {
                "active_branches": active_branch_count,
                "notes": decision_notes,
            },
            "nominal_weights": {
                "branch1": ENSEMBLE_W1,
                "branch2": ENSEMBLE_W2,
                "branch3": ENSEMBLE_W3,
            },
            "weights": {"branch1": w1, "branch2": w2, "branch3": w3},
            "branches": {"branch1": d1, "branch2": d2, "branch3": d3},
            "branch_probs_fake": {
                "branch1": round(p1, 4) if p1 is not None else None,
                "branch2": round(p2, 4),
                "branch3": round(p3, 4) if p3 is not None else None,
            },
            "ensemble": {
                "weighted_components": {
                    "w1_times_p1": wc1,
                    "w2_times_p2": round(w2 * p2, 6),
                    "w3_times_p3": wc3,
                },
                "final_score_fake": round(float(final_score), 4),
                "final_label": final_label,
                "confidence_for_label": round(float(confidence), 4),
            },
        }


def analyze_text_image(
    text: str,
    image_path: str | None = None,
    threshold: float = 0.5,
    base_dir: Path | None = None,
    fast_mode: bool = False,
) -> dict[str, Any]:
    engine = VeriFactEngine.get(base_dir)
    return engine.analyze(text, image_path=image_path, threshold=threshold, fast_mode=fast_mode)
