# Phishing URL Detection System

Lexical-feature + Random Forest phishing URL classifier. The URL is never visited.

## Setup
```
pip install pandas numpy scikit-learn matplotlib
```
Download the dataset (Hannousse & Yahiouche) from
https://data.mendeley.com/datasets/c2gw7fy2j4/3 , take the CSV with `url` and `status`
columns and save it as `phish.csv` next to the script.

## Run
```
python phishing_url_detector.py            # uses phish.csv
python phishing_url_detector.py data.csv   # or give a path
```

## Output
- Console: accuracy, precision, recall, F1, confusion matrix, risk of 5 sample URLs
- `report.json`: metrics + per-URL prediction, probability, risk level, indicators
- `confusion_features.png`: confusion matrix and top 10 feature importances

## Pipeline
Load data -> extract 29 lexical features -> stratified 70/30 split -> Random Forest (200 trees)
-> evaluate -> analyse URLs (LOW < 0.5, MEDIUM 0.5-0.8, HIGH >= 0.8) -> rule-based indicators -> JSON report.
