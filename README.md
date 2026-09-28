# Diabetic Retinopathy Classification Using an ML Model

A simple final/mini-project starter with:
- Flask backend
- Separate HTML, CSS and JavaScript
- Retinal image upload + preview
- ML prediction API
- 5-class output
- Responsive dashboard-style UI

## Important
This is an educational project template. The built-in fallback prediction is NOT medically valid.
For a real ML result, train the supplied model using an appropriate retinal dataset and validate it properly.

## 1. Install Python
Recommended: Python 3.10 or 3.11.

Check:
```bash
python --version
```

## 2. Open project in VS Code
Open this folder:
`diabetic_retinopathy_mini_project`

## 3. Create virtual environment

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

## 4. Install packages
```bash
pip install -r requirements.txt
```

## 5. Run
```bash
python app.py
```

Open:
http://127.0.0.1:5000

## 6. Add a real ML model (optional)
Place training images in:
```text
data/train/0
data/train/1
data/train/2
data/train/3
data/train/4
```

Then run:
```bash
python train_model.py
```

The trained file will be:
`model/retinopathy_model.pkl`

Restart:
```bash
python app.py
```

## 7. Suggested dataset
For academic work, use a reputable retinal fundus dataset such as:
- APTOS 2019 Blindness Detection
- EyePACS
- Messidor

Check each dataset's license/terms before use.

## Project flow
User -> Upload retinal image -> JavaScript preview -> Flask `/predict` -> feature extraction -> ML model -> class + confidence -> result card

## 5 Classes
0: No Diabetic Retinopathy
1: Mild
2: Moderate
3: Severe
4: Proliferative

## Mini-project modules
1. Home / Dashboard
2. Image Upload
3. Image Preview
4. Prediction
5. Result / Risk
6. About / Disclaimer
