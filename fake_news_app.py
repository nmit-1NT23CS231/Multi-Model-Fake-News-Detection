"""
Fake News Detection System — VS Code Application
=================================================
TF-IDF + Logistic Regression | Real-time Prediction

Usage:
  python fake_news_app.py                    → Interactive CLI mode
  python fake_news_app.py --api              → Start Flask REST API
  python fake_news_app.py --text "..."       → Single prediction
  python fake_news_app.py --file news.txt    → Predict from file

Prerequisites:
  pip install scikit-learn joblib flask nltk pandas
  Place fake_news_pipeline.pkl in ./model_artifacts/
"""

import os
import re
import sys
import json
import joblib
import argparse
import textwrap
import nltk
from pathlib import Path

# ── Optional Flask import (only needed for --api mode) ─────────────────────────
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# ── Download NLTK stopwords once ───────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH    = Path('model_artifacts/fake_news_pipeline.pkl')
METADATA_PATH = Path('model_artifacts/model_metadata.json')
MIN_TEXT_LEN  = 30   # Characters — reject inputs that are too short


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          TEXT PREPROCESSING                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def clean_text(text: str) -> str:
    """
    Identical preprocessing pipeline as used during training.
    Must stay in sync with the Colab notebook's clean_text().
    """
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)          # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)                    # Remove emails
    text = re.sub(r'<[^>]+>', '', text)                    # Remove HTML tags
    text = re.sub(r'^.*?\(reuters\)\s*[-–]?\s*', '', text) # Remove wire headers
    text = re.sub(r'[^a-z\s]', ' ', text)                 # Keep only letters
    text = re.sub(r'\s+', ' ', text).strip()              # Normalize whitespace
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         MODEL LOADER                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class FakeNewsDetector:
    """Wraps the trained sklearn pipeline with a clean prediction interface."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"\n❌ Model not found at: {model_path}\n"
                "   Please run the Colab notebook first and place\n"
                "   'fake_news_pipeline.pkl' inside the 'model_artifacts/' folder.\n"
            )
        self.pipeline = joblib.load(model_path)
        self.metadata = self._load_metadata()
        print(f"Model loaded  (test accuracy: {self.metadata.get('test_accuracy', 'N/A')})")

    def _load_metadata(self) -> dict:
        if METADATA_PATH.exists():
            with open(METADATA_PATH) as f:
                return json.load(f)
        return {}

    def predict(self, text: str) -> dict:
        """
        Predict whether the input text is FAKE or REAL news.

        Returns
        -------
        dict with keys:
            verdict          : 'FAKE' or 'REAL'
            confidence       : float (0-100), percentage confidence
            fake_probability : float (0-100)
            real_probability : float (0-100)
            cleaned_text     : str — the processed input
            word_count       : int
            warning          : str | None — if input is too short
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {'error': 'Input text is empty.'}

        warning = None
        if len(text.strip()) < MIN_TEXT_LEN:
            warning = f'Input is very short ({len(text.strip())} chars). Results may be less accurate.'

        cleaned = clean_text(text)

        if not cleaned:
            return {'error': 'Text became empty after cleaning. Please provide more content.'}

        proba = self.pipeline.predict_proba([cleaned])[0]
        fake_pct = round(proba[0] * 100, 2)
        real_pct = round(proba[1] * 100, 2)
        verdict = 'REAL' if real_pct >= fake_pct else 'FAKE'

        return {
            'verdict'         : verdict,
            'confidence'      : max(fake_pct, real_pct),
            'fake_probability': fake_pct,
            'real_probability': real_pct,
            'cleaned_text'    : cleaned[:200] + '...' if len(cleaned) > 200 else cleaned,
            'word_count'      : len(text.split()),
            'warning'         : warning,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Predict multiple articles at once."""
        return [self.predict(t) for t in texts]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         DISPLAY HELPERS                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

BANNER = r"""
╔══════════════════════════════════════════════════════╗
║      🔍  FAKE NEWS DETECTION SYSTEM  🔍              ║
║      TF-IDF + Logistic Regression                    ║
╚══════════════════════════════════════════════════════╝
"""

def print_result(result: dict, text: str = ''):
    """Pretty-print a prediction result to the terminal."""
    if 'error' in result:
        print(f"\nError: {result['error']}\n")
        return

    if result.get('warning'):
        print(f"\n{result['warning']}")

    verdict   = result['verdict']
    conf      = result['confidence']
    fake_pct  = result['fake_probability']
    real_pct  = result['real_probability']

    icon  = '🔴' if verdict == 'FAKE' else '🟢'
    color = '\033[91m' if verdict == 'FAKE' else '\033[92m'
    reset = '\033[0m'

    bar_width = 30
    fake_bar  = '█' * int(fake_pct / 100 * bar_width)
    real_bar  = '█' * int(real_pct / 100 * bar_width)

    if text:
        preview = textwrap.shorten(text, width=80, placeholder='...')
        print(f"\nInput Preview : {preview}")

    print(f"\n{'─'*52}")
    print(f"  {icon} Verdict    : {color}{verdict}{reset}  ({conf:.1f}% confidence)")
    print(f"{'─'*52}")
    print(f"  FAKE │{fake_bar:<{bar_width}}│ {fake_pct:.1f}%")
    print(f"  REAL │{real_bar:<{bar_width}}│ {real_pct:.1f}%")
    print(f"{'─'*52}")
    print(f"  Word Count : {result['word_count']}")
    if result.get('cleaned_text'):
        print(f"  Cleaned    : {result['cleaned_text'][:80]}...")
    print()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      INTERACTIVE CLI MODE                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_cli(detector: FakeNewsDetector):
    """Interactive command-line interface for predictions."""
    print(BANNER)

    # Print model stats
    meta = detector.metadata
    if meta:
        print(f"  Model      : {meta.get('algorithm', 'TF-IDF + LR')}")
        print(f"  Train Acc  : {meta.get('train_accuracy', 'N/A')}")
        print(f"  Test Acc   : {meta.get('test_accuracy', 'N/A')}")
        print(f"  CV Score   : {meta.get('cv_mean', 'N/A')} ± {meta.get('cv_std', 'N/A')}")
        print(f"  ROC-AUC    : {meta.get('roc_auc', 'N/A')}")
    print()

    print("Type or paste a news article below.")
    print("Commands: 'quit' to exit | 'demo' for examples | 'batch' for multiple articles\n")

    demo_articles = [
        {
            'label': 'Expected: REAL',
            'text': (
                'The Federal Reserve raised its benchmark interest rate by a quarter percentage '
                'point Wednesday, continuing its fight against inflation. Fed Chair Jerome Powell '
                'stated that the committee remains committed to returning inflation to its 2 percent '
                'target. Economists widely expected the move, following months of elevated price pressures.'
            )
        },
        {
            'label': 'Expected: FAKE',
            'text': (
                'BREAKING: Scientists confirm that 5G towers are activating dormant viruses '
                'in vaccinated people! The government is hiding this from the public. Share '
                'this before it gets taken down! The deep state does NOT want you to know '
                'that the real agenda behind 5G is population control — wake up!'
            )
        },
        {
            'label': 'Expected: REAL',
            'text': (
                'The World Health Organization released new guidelines on antibiotic resistance, '
                'urging governments to reduce overprescription of antimicrobials. The report, '
                'compiled by international health experts, warns that drug-resistant infections '
                'could cause 10 million deaths annually by 2050 without coordinated action.'
            )
        },
    ]

    while True:
        try:
            user_input = input("Enter article text (or command): ").strip()
        except (EOFError, KeyboardInterrupt):
            print('\n\nGoodbye! 👋')
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ('quit', 'exit', 'q'):
            print('\nGoodbye!\n')
            break

        elif cmd == 'demo':
            print('\nRunning demo predictions...\n')
            for item in demo_articles:
                print(f"[{item['label']}]")
                result = detector.predict(item['text'])
                print_result(result, item['text'])

        elif cmd == 'batch':
            print('\nBatch mode — enter articles one per line.')
            print("Type 'END' on a new line when done.\n")
            lines = []
            while True:
                try:
                    line = input()
                except (EOFError, KeyboardInterrupt):
                    break
                if line.strip().upper() == 'END':
                    break
                lines.append(line.strip())

            if lines:
                articles = [l for l in lines if l]
                print(f'\nProcessing {len(articles)} article(s)...\n')
                results = detector.predict_batch(articles)
                for i, (article, result) in enumerate(zip(articles, results), 1):
                    print(f'--- Article {i} ---')
                    print_result(result, article)

        else:
            result = detector.predict(user_input)
            print_result(result, user_input)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          FLASK REST API                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_api(detector: FakeNewsDetector, host='0.0.0.0', port=5000):
    """Start a Flask REST API server for integration with web frontends."""
    if not FLASK_AVAILABLE:
        print("Flask not installed. Run: pip install flask")
        sys.exit(1)

    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def index():
        return jsonify({
            'service'  : 'Fake News Detection API',
            'version'  : detector.metadata.get('model_version', '1.0.0'),
            'endpoints': {
                'POST /predict' : 'Predict single article',
                'POST /batch'   : 'Predict multiple articles',
                'GET  /health'  : 'Health check',
            }
        })

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok', 'model_loaded': True})

    @app.route('/predict', methods=['POST'])
    def predict():
        """
        POST /predict
        Body: {"text": "Your news article here..."}
        Returns: {"verdict": "FAKE"|"REAL", "confidence": 87.3, ...}
        """
        data = request.get_json(silent=True)
        if not data or 'text' not in data:
            return jsonify({'error': 'Request body must contain "text" field.'}), 400

        result = detector.predict(data['text'])
        return jsonify(result)

    @app.route('/batch', methods=['POST'])
    def batch_predict():
        """
        POST /batch
        Body: {"articles": ["text1", "text2", ...]}
        Returns: {"results": [...]}
        """
        data = request.get_json(silent=True)
        if not data or 'articles' not in data:
            return jsonify({'error': 'Request body must contain "articles" list.'}), 400

        articles = data['articles']
        if not isinstance(articles, list) or len(articles) == 0:
            return jsonify({'error': '"articles" must be a non-empty list.'}), 400

        if len(articles) > 50:
            return jsonify({'error': 'Maximum 50 articles per batch request.'}), 400

        results = detector.predict_batch(articles)
        return jsonify({'results': results, 'count': len(results)})

    print(BANNER)
    print(f"API running at http://{host}:{port}")
    print("Endpoints:")
    print(f"  POST http://localhost:{port}/predict")
    print(f"  POST http://localhost:{port}/batch")
    print(f"  GET  http://localhost:{port}/health\n")
    print("Example curl command:")
    print(f'  curl -X POST http://localhost:{port}/predict \\')
    print(f'       -H "Content-Type: application/json" \\')
    print(f'       -d \'{{"text": "Your news article here..."}}\'\n')

    app.run(host=host, port=port, debug=False)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              MAIN ENTRY                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description='Fake News Detection — TF-IDF + Logistic Regression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python fake_news_app.py                          # Interactive CLI
              python fake_news_app.py --api                    # Start REST API
              python fake_news_app.py --api --port 8080        # Custom port
              python fake_news_app.py --text "Breaking: ..."   # Quick prediction
              python fake_news_app.py --file article.txt       # Predict from file
        """)
    )
    parser.add_argument('--api',   action='store_true', help='Start Flask REST API server')
    parser.add_argument('--port',  type=int, default=5000, help='API port (default: 5000)')
    parser.add_argument('--host',  type=str, default='0.0.0.0', help='API host (default: 0.0.0.0)')
    parser.add_argument('--text',  type=str, help='Article text for single prediction')
    parser.add_argument('--file',  type=str, help='Path to text file for prediction')
    parser.add_argument('--model', type=str, default=str(MODEL_PATH), help='Path to model .pkl file')

    args = parser.parse_args()

    # Load model
    try:
        detector = FakeNewsDetector(Path(args.model))
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # ── Mode selection ──────────────────────────────────────────────────────────
    if args.text:
        # Single prediction from CLI argument
        result = detector.predict(args.text)
        print_result(result, args.text)

    elif args.file:
        # Predict from file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
        text = file_path.read_text(encoding='utf-8')
        result = detector.predict(text)
        print(f"\n📄 File: {args.file}")
        print_result(result, text)

    elif args.api:
        # Start REST API
        run_api(detector, host=args.host, port=args.port)

    else:
        # Default: interactive CLI
        run_cli(detector)


if __name__ == '__main__':
    main()
