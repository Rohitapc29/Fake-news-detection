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
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import imagehash
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
    from ddgs import DDGS 
      # noqa: ICN003
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
BLIP2_MODEL_ID = "Salesforce/blip2-opt-2.7b"
SIM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


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
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        from sentence_transformers import SentenceTransformer

        print("Loading BLIP-2 (Branch 1)...")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._blip2_processor = Blip2Processor.from_pretrained(BLIP2_MODEL_ID)
        self._blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            BLIP2_MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        self._blip2_model.eval()
        if self.device != "cuda":
            self._blip2_model.to(self.device)
        self._sim_model = SentenceTransformer(SIM_MODEL_ID, device=self.device)
        self._torch_ready = True

    def ensure_sklearn_branches(self) -> None:
        if self._sk_ready:
            return
        
        debug_log = []
        
        # Try to load Branch 1 sklearn files (optional)
        self._b1_available = False
        try:
            if self.paths["b1_clf"].exists() and self.paths["b1_scaler"].exists():
                debug_log.append("Starting B1 loads...")
                try:
                    self._b1_clf = joblib.load(self.paths["b1_clf"])
                    debug_log.append("B1 clf loaded OK")
                except TypeError as e:
                    if "BitGenerator" in str(e) and "__setstate__" in str(e):
                        debug_log.append(f"B1 clf BitGenerator error (expected): {str(e)[:100]}")
                        debug_log.append("Attempting direct load with fallback unpickler...")
                        # Try with protocol 2 reading manually
                        with open(self.paths["b1_clf"], "rb") as f:
                            import pickle
                            try:
                                self._b1_clf = pickle.load(f)
                                debug_log.append("B1 clf loaded via pickle (fallback)")
                            except:
                                raise
                
                try:
                    self._b1_scaler = joblib.load(self.paths["b1_scaler"])
                    debug_log.append("B1 scaler loaded OK")
                except TypeError as e:
                    if "BitGenerator" in str(e) and "__setstate__" in str(e):
                        debug_log.append(f"B1 scaler BitGenerator error (expected): {str(e)[:100]}")
                        with open(self.paths["b1_scaler"], "rb") as f:
                            import pickle
                            try:
                                self._b1_scaler = pickle.load(f)
                                debug_log.append("B1 scaler loaded via pickle (fallback)")
                            except:
                                raise
                
                self._b1_available = True
                debug_log.append("✓ B1 LOADED")
            else:
                debug_log.append("B1 files missing")
        except Exception as e:
            import traceback
            debug_log.append(f"B1 LOAD FAILED: {str(e)}")
            debug_log.append(traceback.format_exc())
        
        # Try to load Branch 3 sklearn files (optional)
        self._b3_available = False
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
                debug_log.append("B3 files missing")
        except Exception as e:
            import traceback
            debug_log.append(f"B3 LOAD FAILED: {str(e)}")
            debug_log.append(traceback.format_exc())
        
        # Write debug log to file
        try:
            log_path = self.base_dir / "outputs" / "sklearn_load_debug.log"
            with open(log_path, "w") as f:
                f.write("\n".join(debug_log))
        except:
            pass
        
        try:
            self._ddgs = DDGS()
            debug_log.append("✓ DDGS initialized")
        except Exception as e:
            debug_log.append(f"DDGS init error: {str(e)}")
            self._ddgs = None
        
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

    def branch1_detailed(self, image_path: str | None, text: str) -> dict[str, Any]:
        detail: dict[str, Any] = {
            "branch": "Branch 1 — Vision (BLIP-2 + similarity)",
            "image_provided": bool(image_path),
        }
        if not image_path:
            detail["disabled"] = True
            detail["reason"] = "No image provided; vision branch is not run and gets no ensemble weight."
            return detail
        
        if not self._b1_available:
            detail["disabled"] = True
            detail["reason"] = "Branch 1 sklearn files not available (best_clf.pkl, scaler.pkl missing)."
            detail["prob_fake"] = 0.5
            return detail

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
            feats = self._b1_scaler.transform([[sim, wc]])
            detail["scaled_features"] = [round(float(feats[0][0]), 6), float(wc)]
            prob_fake = float(self._b1_clf.predict_proba(feats)[0][1])
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

    def branch3_detailed(self, text: str, image_path: str | None) -> dict[str, Any]:
        detail: dict[str, Any] = {"branch": "Branch 3 — OSINT + Random Forest"}
        
        if not self._b3_available:
            detail["disabled"] = True
            detail["reason"] = "Branch 3 sklearn files not available (branch3_v2_*.pkl missing)."
            detail["prob_fake"] = 0.5
            return detail
        
        if not self._ddgs:
            detail["disabled"] = True
            detail["reason"] = "DuckDuckGo search not available (DDGS init failed)."
            detail["prob_fake"] = 0.5
            return detail

        fact_checkers = [
            "snopes","politifact","factcheck","reuters",
            "apnews","leadstories","fullfact","afp","usatoday"
        ]

        debunk_words = [
            "false","fake","hoax","debunked","misleading",
            "altered","satire","scam","unsupported","unproven"
        ]

        true_words = ["true","correct","accurate","authentic","verified","real"]

        whitelist = ['reuters.com','apnews.com','bbc.com','thehindu.com','nytimes.com','cnn.com','ndtv.com']
        blacklist = ['theonion.com','infowars.com','breitbart.com','babylonbee.com']

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
            detail["search_query_headline"] = headline

            # ================================
            # SEARCH + DOMAIN LOGIC
            # ================================
            snippets_combined, domains_combined = "", ""
            trusted_score, suspicious_score = 0, 0

            results = list(self._ddgs.text(headline, max_results=5))
            detail["ddg_text_hits"] = len(results)

            preview_snips = []
            preview_domains = []
            seen_domains = set()

            for res in results:
                body = str(res.get("body", ""))
                href = res.get("href", "")

                dom = urlparse(href).netloc.replace("www.", "").lower()

                snippets_combined += body.lower() + " || "

                if dom and dom not in seen_domains:
                    domains_combined += dom + " "
                    preview_snips.append(body[:150])
                    preview_domains.append(dom)
                    seen_domains.add(dom)

                    if any(w in dom for w in whitelist):
                        trusted_score += 1
                    if any(b in dom for b in blacklist):
                        suspicious_score += 1

            detail["result_snippets_preview"] = preview_snips
            detail["result_domains"] = preview_domains

            # ================================
            # HYBRID OVERRIDE
            # ================================
            has_fc = any(fc in domains_combined for fc in fact_checkers)
            has_debunk = any(dw in snippets_combined for dw in debunk_words)
            has_true = any(tw in snippets_combined for tw in true_words)

            detail["signals"] = {
                "trusted_sources": trusted_score,
                "suspicious_sources": suspicious_score,
                "fact_checker_hit": has_fc,
                "debunk_language": has_debunk,
                "verification_language": has_true
            }

            if suspicious_score >= 2:
                detail["decision_path"] = "Override → suspicious domains"
                detail["prob_fake"] = 0.9
                return detail

            if trusted_score >= 2 and not has_debunk:
                detail["decision_path"] = "Override → trusted domains"
                detail["prob_fake"] = 0.1
                return detail

            if has_fc and has_debunk:
                detail["decision_path"] = "Override → fact-check debunk"
                detail["prob_fake"] = 0.95
                return detail

            if has_fc and has_true and not has_debunk:
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

            if image_path:
                try:
                    local_hash = imagehash.phash(Image.open(image_path))
                    ddg_imgs = list(self._ddgs.images(headline, max_results=1))
                    if ddg_imgs:
                        resp = requests.get(ddg_imgs[0]["image"], timeout=3)
                        hash_dist = local_hash - imagehash.phash(Image.open(io.BytesIO(resp.content)))
                except:
                    pass

            fc_count = sum(1 for d in fact_checkers if d in domains_combined)
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

            prob_fake = float(self._b3_model.predict_proba(X_live)[0][fake_idx])
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
    ) -> dict[str, Any]:
        use_b1 = bool(image_path)  # Whether image was provided
        self.load_all(use_branch1_vision=use_b1)

        d1 = self.branch1_detailed(image_path, text)
        d2 = self.branch2_detailed(text)
        d3 = self.branch3_detailed(text, image_path)

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

        final_label = "FAKE" if final_score >= threshold else "REAL"
        confidence = final_score if final_label == "FAKE" else (1.0 - final_score)

        wc1 = round(w1 * p1, 6) if p1 is not None else 0.0
        wc3 = round(w3 * p3, 6) if p3 is not None else 0.0

        return {
            "threshold": threshold,
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
) -> dict[str, Any]:
    engine = VeriFactEngine.get(base_dir)
    return engine.analyze(text, image_path=image_path, threshold=threshold)

print(list(DDGS().text("test")))
