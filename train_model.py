"""
Optional training script.

Expected dataset structure:
data/
  train/
    0/
    1/
    2/
    3/
    4/

The folders correspond to:
0 = No DR
1 = Mild
2 = Moderate
3 = Severe
4 = Proliferative

This starter model is intentionally simple for a mini project:
- resize image to 64x64
- extract 11 numeric features
- train Random Forest
- save model/retinopathy_model.pkl

For a serious research project, use a validated retinal dataset and a CNN/transfer-learning
pipeline with proper patient-level train/validation/test splitting.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "train"
MODEL = BASE / "model" / "retinopathy_model.pkl"
EXTS = {".jpg", ".jpeg", ".png"}

def features(path):
    img = Image.open(path).convert("RGB").resize((64,64))
    a = np.asarray(img, dtype=np.float32) / 255.0
    g = a.mean(axis=2)
    return [
        g.mean(), g.std(), a[:,:,0].mean(), a[:,:,1].mean(), a[:,:,2].mean(),
        np.percentile(g,5), np.percentile(g,25), np.percentile(g,75),
        np.percentile(g,95), (g < .18).mean(), (g > .85).mean()
    ]

X, y = [], []
for class_id in range(5):
    folder = DATA / str(class_id)
    if not folder.exists():
        continue
    for p in folder.rglob("*"):
        if p.suffix.lower() in EXTS:
            try:
                X.append(features(p))
                y.append(class_id)
            except Exception:
                pass

if len(set(y)) < 2:
    raise SystemExit(
        "Need at least 2 classes of images. Put images in data/train/0 ... data/train/4."
    )

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, zero_division=0))

MODEL.parent.mkdir(exist_ok=True)
joblib.dump(clf, MODEL)
print("Saved:", MODEL)
