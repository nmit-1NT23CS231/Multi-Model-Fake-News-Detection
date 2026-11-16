"""
authen - check whether text, images and emails can be trusted.

Run:  python app.py        then open  http://127.0.0.1:8000
"""
import logging
import os

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from services.email_service import EmailService
from services.image_service import ImageService
from services.text_service import TextService

MAX_UPLOAD_MB = 15

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("authen")

SERVICES = {"text": TextService(), "image": ImageService(), "email": EmailService()}


def load_models():
    """Load every model once at start-up. One failing model never blocks the others."""
    for key, svc in SERVICES.items():
        try:
            log.info("Loading %s model ...", key)
            svc.load()
            log.info("  %s model ready", key)
        except Exception as exc:  # noqa: BLE001 - we want to report any failure
            svc.error = f"{type(exc).__name__}: {exc}"
            log.error("  %s model FAILED to load: %s", key, svc.error)


load_models()


def page_context():
    return {
        key: {"ready": s.ready, "error": s.error, "summary": s.summary if s.ready else ""}
        for key, s in SERVICES.items()
    }


# ----------------------------------------------------------------- pages
@app.get("/")
def home():
    return render_template("home.html", models=page_context())


@app.get("/text")
def text_page():
    return render_template("text.html", model=page_context()["text"], active="text")


@app.get("/image")
def image_page():
    return render_template("image.html", model=page_context()["image"], active="image")


@app.get("/email")
def email_page():
    return render_template("email.html", model=page_context()["email"], active="email")


# ------------------------------------------------------------------ API
def fail(message, status=400):
    return jsonify({"error": message}), status


def get_service(key):
    svc = SERVICES[key]
    if not svc.ready:
        return None, fail(f"The {key} model is not available. {svc.error or ''} "
                          "See the README troubleshooting section.", 503)
    return svc, None


@app.post("/api/text")
def api_text():
    svc, err = get_service("text")
    if err:
        return err
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(svc.predict(payload.get("text", "")))
    except ValueError as exc:
        return fail(str(exc))
    except Exception:
        log.exception("text prediction failed")
        return fail("Something went wrong while checking the text.", 500)


@app.post("/api/image")
def api_image():
    svc, err = get_service("image")
    if err:
        return err
    f = request.files.get("file")
    if not f or not f.filename:
        return fail("Choose an image to check.")
    try:
        return jsonify(svc.predict(f.read()))
    except ValueError as exc:
        return fail(str(exc))
    except Exception:
        log.exception("image prediction failed")
        return fail("Something went wrong while checking the image.", 500)


@app.post("/api/email")
def api_email():
    svc, err = get_service("email")
    if err:
        return err
    try:
        f = request.files.get("file")
        if f and f.filename:
            return jsonify(svc.predict_file(f.read(), f.filename))
        payload = request.get_json(silent=True) or {}
        if payload.get("body"):
            return jsonify(svc.predict_text(payload.get("subject"), payload.get("from"), payload["body"]))
        return fail("Upload a .eml or .pdf file, or paste the message text.")
    except ValueError as exc:
        return fail(str(exc))
    except Exception:
        log.exception("email prediction failed")
        return fail("Something went wrong while checking the email.", 500)


@app.get("/api/health")
def health():
    return jsonify({k: {"ready": s.ready, "error": s.error} for k, s in SERVICES.items()})


# --------------------------------------------------------------- errors
@app.errorhandler(HTTPException)
def http_error(exc):
    if request.path.startswith("/api/"):
        msg = f"That file is too large. The limit is {MAX_UPLOAD_MB} MB." if exc.code == 413 else exc.description
        return fail(msg, exc.code)
    return render_template("error.html", code=exc.code, message=exc.description), exc.code


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="127.0.0.1", port=port, debug=False)
