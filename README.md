

A website that checks whether a **news text**, an **image** or an **email** can be trusted.
It combines three trained models behind one Flask app:

| Page | You give it | Model | What you get back |
|------|-------------|-------|-------------------|
| **Text** | A pasted article or a `.txt` file | TF-IDF + Logistic Regression | Looks real / Looks fake, with probabilities |
| **Image** | A JPG, PNG, BMP or WebP | Small Keras CNN (reads 32×32 pixels) | Likely authentic / Likely AI-generated, with probability and confidence |
| **Email** | A `.eml` file, a PDF, or pasted text | TF-IDF + calibrated LinearSVC | Looks legitimate / Likely phishing, plus risky-attachment warnings |

The home page shows the title and three cards. Each card opens its own page.

---

## 1. What you need

- **Python 3.10, 3.11 or 3.12** (3.11 is the tested path). Check with `python --version`.
  Python 3.13 or newer will **not** work, because TensorFlow 2.16 does not support it yet.
- About **1.5 GB** of free disk space (most of it is TensorFlow).
- An internet connection for the first install. After that the app runs offline
  (only the Google Fonts are fetched; without internet the site falls back to system fonts).

## 2. Step-by-step: run it

### Step 1. Unzip and open the folder

Unzip `authen.zip`, then open the `authen` folder in VS Code (**File → Open Folder**) or in a terminal.
Open the VS Code terminal with **Terminal → New Terminal**. All commands below run from inside the `authen` folder.

### Step 2. Create a virtual environment

**Windows (PowerShell or Command Prompt)**
```
python -m venv .venv
.venv\Scripts\activate
```
If PowerShell refuses with a "running scripts is disabled" error, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux**
```
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`.

### Step 3. Install the dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```
This takes a few minutes because of TensorFlow. It installs the exact scikit-learn version
(1.6.1) the text and email models were trained with.

### Step 4. (Optional) Check that everything works

```
python tests/smoke_test.py
```
You should see `PASS` on every line and `All good.` at the end.

### Step 5. Start the website

```
python app.py
```
The first start takes **20 to 60 seconds** while TensorFlow loads. Wait until you see:

```
INFO    text model ready
INFO    image model ready
INFO    email model ready
 * Running on http://127.0.0.1:8000
```

### Step 6. Open it

Go to **http://127.0.0.1:8000** in your browser. Pick a card, add your input, and read the result.
Stop the server with `Ctrl + C`.

### Try it with the included samples

The `samples/` folder has ready-made files:

- `samples/images/` has 8 images for the Image page.
- `samples/emails/` has 5 `.eml` files (`phishing_example.eml` includes a risky `.exe` attachment).
- `samples/pdfs/` has PDFs for the Email page. `uber.pdf` is a scanned image with no text layer,
  so authen correctly reports that it can't read it.

The Text page has three one-click examples, and the Email page has a "Fill with an example" button.
Sample results are model predictions, not confirmed ground truth.

---

## 3. Project structure

```
authen/
├── app.py                     Flask app: pages + JSON API, loads the 3 models at start-up
├── requirements.txt
├── README.md
├── models/
│   ├── text/                  fake_news_pipeline.pkl, model_metadata.json
│   ├── image/                 fake_image_weights.npz + fake_image_meta.json (used)
│   │                          fake_image_detector.pkl (original, kept as fallback)
│   └── email/                 phishing_model.pkl, tfidf_vectorizer.pkl, model_meta.json
├── services/
│   ├── text_service.py        cleaning + prediction for the text model
│   ├── image_service.py       pre-processing + prediction for the image model
│   ├── email_service.py       .eml / PDF parsing + prediction for the email model
│   └── data/english_stopwords.txt   bundled, so no NLTK download is needed
├── templates/                 home, text, image, email pages
├── static/
│   ├── css/style.css
│   ├── js/                    common.js, text.js, image.js, email.js
│   └── img/guilloche.svg      the rosette behind the title
├── samples/                   images, .eml files and PDFs to test with
├── scripts/convert_image_model.py   how the image weights were converted (see below)
└── tests/smoke_test.py
```

## 4. How the models are wired in

Each model has a *service* class in `services/` with the same pattern: `load()` once at start-up,
then `predict(...)`. The pre-processing in each service is copied from your original code, so
predictions match what you saw in VS Code:

- **Text**: the same `clean_text()` as `Text_Model.ipynb` (lower-case, strip URLs / e-mail addresses /
  HTML / Reuters header, letters only, drop stopwords and words under 3 letters).
- **Image**: RGB, resize to 32×32, divide by 255. The High/Low confidence rule (distance from 50%
  at least 0.30) is the same as `vscode_inference.py`.
- **Email**: the same `clean_text()` and feature text (subject + sender + body + attachment names +
  attachment text) as `predict_email.py`. The only change is that files are read from memory
  (uploaded bytes) rather than from a path.

If one model fails to load, the other two still work, and the broken one is marked
"Unavailable" on the home page.

### JSON API (used by the pages, also usable directly)

| Method | URL | Body | Returns |
|--------|-----|------|---------|
| POST | `/api/text` | `{"text": "..."}` | `verdict`, `confidence`, `fake_probability`, `real_probability`, ... |
| POST | `/api/image` | form field `file` | `classification`, `ai_probability_percent`, `confidence`, ... |
| POST | `/api/email` | form field `file` (`.eml`/`.pdf`) **or** `{"subject","from","body"}` | `prediction`, `confidence`, `risky_attachments`, ... |
| GET | `/api/health` | none | which models loaded |

Example: `curl -X POST http://127.0.0.1:8000/api/text -H "Content-Type: application/json" -d "{\"text\": \"Your article here\"}"`

---

## 5. Troubleshooting

**`python` is not recognised / wrong version.** Install Python 3.11 from python.org (tick
"Add Python to PATH" on Windows). On macOS/Linux use `python3`. If several versions are installed,
create the environment with a specific one, e.g. `py -3.11 -m venv .venv` on Windows.

**TensorFlow fails to install.** This is almost always a Python version problem. Use 3.10, 3.11 or 3.12
in a fresh virtual environment, and make sure you are running 64-bit Python.

**The home page says the Image card is "Unavailable".** TensorFlow did not load. Run
`python -c "import tensorflow as tf; print(tf.__version__)"` to see the real error. The Text and Email
checkers keep working meanwhile.

**`ModuleNotFoundError: numpy._core`.** This happens if you point authen at the original
`fake_image_detector.pkl` in an environment whose NumPy is older than the one that saved it. authen avoids
this by loading `fake_image_weights.npz` + `fake_image_meta.json` first. Don't delete those two files.
If you retrain the image model, create a fresh pair with
`python scripts/convert_image_model.py models/image/fake_image_detector.pkl`
(run it in an environment that can read the new `.pkl`).

**`InconsistentVersionWarning` from scikit-learn.** You have a scikit-learn other than 1.6.1.
Run `pip install scikit-learn==1.6.1`.

**Port 8000 is already in use.** Start on another port:
Windows PowerShell `$env:PORT=8010; python app.py`, Windows Command Prompt `set PORT=8010 && python app.py`, macOS/Linux `PORT=8010 python app.py`.

**"No text could be extracted from this PDF."** The PDF is a scan (just pictures of text). authen reads
text layers only. OCR support was optional in your original script and is not included here.

**"That file is too large."** Uploads are limited to 15 MB (`MAX_UPLOAD_MB` in `app.py`).

**The first request feels slow.** The models are loaded at start-up, so only the very first start is slow.
Wait for the three "model ready" lines before opening the site.

## 6. Good to know (limitations)

- These are statistical models, so results are **probabilities, not proof**. The site says so on every result.
- **Text** judges writing style and vocabulary from its training data. It cannot fact-check a claim, and very
  short text gives unreliable results (authen warns you under 30 characters).
- **Image** looks at a 32×32 pixel version of the picture, which discards fine detail. Treat it as a hint.
  The result page shows you the tiny version the model actually saw.
- **Email** reads text only. It does not verify the sender (SPF/DKIM), follow links or open attachments
  (except to read text from PDF, DOCX and TXT attachments).
- `python app.py` uses Flask's built-in development server, bound to `127.0.0.1` (your machine only).
  For public hosting, put the app behind a production server such as `waitress` (Windows) or
  `gunicorn` (Linux/macOS), with a single worker or per-worker model loading.

## 7. Where to change things

| I want to... | Edit |
|--------------|------|
| Change colours or fonts | `:root` variables at the top of `static/css/style.css` |
| Change card text | `templates/home.html` |
| Change the image "High confidence" rule | `CONFIDENCE_THRESHOLD` in `services/image_service.py` |
| Change the upload limit | `MAX_UPLOAD_MB` in `app.py` |
| Swap in a retrained model | Replace the files in `models/<type>/` and keep the pre-processing in sync |
