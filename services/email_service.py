"""
Email model: phishing detector (TF-IDF + calibrated LinearSVC).

Accepts a .eml file, a standalone .pdf, or pasted text. Parsing and cleaning
follow the original predict_email.py; the only change is that files are read
from memory (uploaded bytes) instead of from disk paths.

clean_text() MUST match the cleaning used during training.
"""
import hashlib
import io
import json
import os
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path

import joblib

try:
    import pdfplumber
    HAVE_PDF = True
except ImportError:  # app still starts; PDF input is reported as unavailable
    HAVE_PDF = False

try:
    import docx  # python-docx, optional
    HAVE_DOCX = True
except ImportError:
    HAVE_DOCX = False

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "email"

DANGEROUS_EXTENSIONS = {
    ".exe", ".scr", ".bat", ".cmd", ".com", ".pif", ".vbs", ".vbe", ".js",
    ".jse", ".wsf", ".wsh", ".ps1", ".msi", ".jar", ".lnk", ".cpl", ".hta",
    ".docm", ".xlsm", ".pptm", ".dotm", ".xltm", ".iso", ".img",
}
MACRO_EXTENSIONS = {".docm", ".xlsm", ".pptm", ".dotm", ".xltm"}
MAX_PASTE_CHARS = 200_000


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URLTOKEN ", text)
    text = re.sub(r"\S+@\S+", " EMAILTOKEN ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    return text


def extract_attachment_text(filename: str, content: bytes) -> str:
    """Best-effort text extraction from common attachment types."""
    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == ".pdf" and HAVE_PDF:
            parts = []
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages[:20]:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
            return "\n".join(parts)
        if ext == ".docx" and HAVE_DOCX:
            d = docx.Document(io.BytesIO(content))
            return "\n".join(p.text for p in d.paragraphs)
        if ext in (".txt", ".csv", ".log"):
            return content.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    return ""


class EmailService:
    key = "email"

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.meta = {}
        self.error = None

    # ------------------------------------------------------------------ load
    def load(self):
        model_path = MODEL_DIR / "phishing_model.pkl"
        vec_path = MODEL_DIR / "tfidf_vectorizer.pkl"
        meta_path = MODEL_DIR / "model_meta.json"
        for p in (model_path, vec_path):
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vec_path)
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))

    @property
    def ready(self):
        return self.model is not None and self.vectorizer is not None

    @property
    def summary(self):
        m = self.meta
        parts = []
        if "test_accuracy" in m:
            parts.append(f"{m['test_accuracy'] * 100:.1f}% test accuracy")
        if "best_model" in m:
            parts.append(m["best_model"])
        return " \u00b7 ".join(parts) or "TF-IDF + linear classifier"

    # --------------------------------------------------------------- parsing
    def parse_eml(self, data: bytes) -> dict:
        msg = BytesParser(policy=policy.default).parsebytes(data)
        body_parts, attachments = [], []

        def walk(part):
            if part.is_multipart():
                for sub in part.iter_parts():
                    walk(sub)
                return
            ctype = part.get_content_type()
            disposition = (part.get_content_disposition() or "").lower()
            filename = part.get_filename()

            if filename:
                try:
                    payload = part.get_payload(decode=True) or b""
                except Exception:
                    payload = b""
                ext = os.path.splitext(filename)[1].lower()
                attachments.append({
                    "filename": filename,
                    "content_type": ctype,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest() if payload else "",
                    "dangerous_extension": ext in DANGEROUS_EXTENSIONS,
                    "macro_enabled_office": ext in MACRO_EXTENSIONS,
                    "extracted_text_snippet": extract_attachment_text(filename, payload)[:500],
                })
                return
            if disposition == "attachment":
                return
            try:
                if ctype == "text/plain":
                    body_parts.append(part.get_content())
                elif ctype == "text/html":
                    body_parts.append(strip_html(part.get_content()))
            except Exception:
                pass

        walk(msg)
        return {
            "subject": str(msg.get("subject", "") or ""),
            "from": str(msg.get("from", "") or ""),
            "body_text": "\n".join(body_parts).strip(),
            "attachments": attachments,
        }

    def parse_pdf(self, data: bytes, filename: str) -> dict:
        if not HAVE_PDF:
            raise RuntimeError("PDF support is not installed. Run: pip install pdfplumber")
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                pages = [p.extract_text() for p in pdf.pages[:30]]
                text = "\n".join(t for t in pages if t)
        except Exception:
            raise ValueError("That PDF could not be opened. It may be damaged or password-protected.")
        if not text.strip():
            raise ValueError("No text could be extracted from this PDF. "
                             "It may be a scanned image, which authen cannot read.")
        return {"subject": filename, "from": "", "body_text": text.strip(), "attachments": []}

    # --------------------------------------------------------------- predict
    def _classify(self, parsed: dict) -> dict:
        attachment_names = " ".join(a["filename"] for a in parsed["attachments"])
        snippets = " ".join(a["extracted_text_snippet"] for a in parsed["attachments"])
        feature_text = clean_text(" ".join([
            parsed["subject"], parsed["from"], parsed["body_text"], attachment_names, snippets,
        ]))
        if not feature_text:
            raise ValueError("There was no readable text in this message to check.")

        X = self.vectorizer.transform([feature_text])
        pred = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        fake_p = float(proba[classes.index(1)])
        label_map = self.meta.get("label_map", {"0": "legit", "1": "fake"})
        label = label_map.get(str(pred), "fake" if pred == 1 else "legit")

        risky = [a["filename"] for a in parsed["attachments"]
                 if a["dangerous_extension"] or a["macro_enabled_office"]]
        return {
            "prediction": label.upper(),                  # FAKE | LEGIT
            "confidence": round(max(proba) * 100, 2),
            "fake_probability": round(fake_p * 100, 2),
            "legit_probability": round((1 - fake_p) * 100, 2),
            "subject": parsed["subject"],
            "from": parsed["from"],
            "body_char_count": len(parsed["body_text"]),
            "body_preview": parsed["body_text"][:400],
            "attachments": [
                {"filename": a["filename"], "size_bytes": a["size_bytes"],
                 "risky": a["dangerous_extension"] or a["macro_enabled_office"],
                 "macro": a["macro_enabled_office"]}
                for a in parsed["attachments"]
            ],
            "risky_attachments": risky,
        }

    def predict_file(self, data: bytes, filename: str) -> dict:
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".eml":
            parsed = self.parse_eml(data)
        elif ext == ".pdf":
            parsed = self.parse_pdf(data, filename)
        else:
            raise ValueError("Unsupported file type. Upload a .eml or .pdf file.")
        result = self._classify(parsed)
        result["source"] = ext.lstrip(".")
        return result

    def predict_text(self, subject: str, sender: str, body: str) -> dict:
        body = (body or "").strip()
        if not body:
            raise ValueError("Paste the message text to check.")
        if len(body) > MAX_PASTE_CHARS:
            raise ValueError(f"That message is too long. The limit is {MAX_PASTE_CHARS:,} characters.")
        result = self._classify({
            "subject": (subject or "").strip(), "from": (sender or "").strip(),
            "body_text": body, "attachments": [],
        })
        result["source"] = "text"
        return result
