"""
Quick end-to-end check that all three models load and answer.
Run from the project root:   python tests/smoke_test.py
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import app  # loads all three models  # noqa: E402

client = app.test_client()
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name} {detail}")
    if not ok:
        failures.append(name)


health = client.get("/api/health").get_json()
for key, st in health.items():
    check(f"{key} model loaded", st["ready"], st["error"] or "")

for path in ("/", "/text", "/image", "/email"):
    check(f"page {path}", client.get(path).status_code == 200)

r = client.post("/api/text", json={"text": "BREAKING: Share this before it gets taken down! "
                                          "The deep state does NOT want you to know the truth, wake up!"})
check("text prediction", r.status_code == 200 and r.get_json()["verdict"] in ("FAKE", "REAL"))

img = next((ROOT / "samples" / "images").glob("*"))
r = client.post("/api/image", data={"file": (io.BytesIO(img.read_bytes()), img.name)},
                content_type="multipart/form-data")
check("image prediction", r.status_code == 200 and r.get_json()["classification"] in ("FAKE", "REAL"))

eml = ROOT / "samples" / "emails" / "phishing_example.eml"
r = client.post("/api/email", data={"file": (io.BytesIO(eml.read_bytes()), eml.name)},
                content_type="multipart/form-data")
check("email prediction", r.status_code == 200 and r.get_json()["prediction"] in ("FAKE", "LEGIT"))

print("\nAll good." if not failures else f"\n{len(failures)} check(s) failed.")
sys.exit(1 if failures else 0)
