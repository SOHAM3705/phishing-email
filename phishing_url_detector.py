"""
Phishing URL Detection System
Lexical features + Random Forest. The URL is never opened or requested.

Usage:
    python phishing_url_detector.py                 # uses phish.csv
    python phishing_url_detector.py path/to/data.csv
"""
import re
import sys
import json
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

REPORT_FILE = "report.json"
CHART_FILE = "confusion_features.png"

KEYWORDS = ["login", "signin", "verify", "secure", "account", "update", "bank",
            "confirm", "password", "paypal", "webscr", "billing", "wallet", "auth"]
SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly"]
SUSPICIOUS_TLDS = ["xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "work", "zip"]


def entropy(text):
    """Shannon entropy of a string."""
    if not text:
        return 0.0
    probs = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in probs)


def extract_features(url):
    """29 lexical features only - the URL is never opened or requested."""
    url = url.strip()
    try:
        p = urlparse(url if "://" in url else "http://" + url)
        host = (p.hostname or "").lower()
        port = p.port
    except ValueError:
        p, host, port = urlparse("http://invalid"), "", None
    path, query = p.path or "", p.query or ""
    parts = [x for x in host.split(".") if x]
    words = re.split(r"[\W_]+", url)
    low = url.lower()
    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": sum(c.isdigit() for c in url),
        "digit_ratio": sum(c.isdigit() for c in url) / max(len(url), 1),
        "num_at": url.count("@"),
        "num_question": url.count("?"),
        "num_equals": url.count("="),
        "num_ampersand": url.count("&"),
        "num_percent": url.count("%"),
        "has_ip": int(bool(re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", host))),
        "is_https": int(p.scheme == "https"),
        "num_subdomains": max(len(parts) - 2, 0),
        "host_hyphens": host.count("-"),
        "host_digits": sum(c.isdigit() for c in host),
        "num_keywords": sum(k in low for k in KEYWORDS),
        "is_shortener": int(host in SHORTENERS),
        "suspicious_tld": int(bool(parts) and parts[-1] in SUSPICIOUS_TLDS),
        "has_port": int(port is not None),
        "has_punycode": int("xn--" in host),
        "http_in_path": int("http" in path.lower() or "http" in query.lower()),
        "starts_with_www": int(host.startswith("www.")),
        "longest_word": max((len(w) for w in words), default=0),
        "url_entropy": entropy(url),
    }


def indicators(url, f):
    """Human-readable red flags used in the forensic report."""
    flags = []
    if f["has_ip"]:
        flags.append("IP address used instead of domain name")
    if f["num_at"]:
        flags.append("'@' symbol present in URL")
    if not f["is_https"]:
        flags.append("Connection is not HTTPS")
    if f["is_shortener"]:
        flags.append("URL shortening service used")
    if f["num_subdomains"] >= 3:
        flags.append("Excessive number of subdomains")
    if f["num_keywords"] >= 2:
        flags.append("Multiple sensitive keywords (login, verify, secure...)")
    if f["suspicious_tld"]:
        flags.append("Suspicious top-level domain")
    if f["url_length"] > 75:
        flags.append("Unusually long URL")
    if f["has_punycode"]:
        flags.append("Punycode (possible homograph attack)")
    return flags


def load_data(path):
    df = pd.read_csv(path)[["url", "status"]].dropna()
    X = pd.DataFrame([extract_features(u) for u in df["url"]])
    y = (df["status"] == "phishing").astype(int)
    return X, y


def train_and_evaluate(dataset_path="phish.csv"):
    X, y = load_data(dataset_path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1_score": f1_score(y_te, pred),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
    }
    return model, X.columns.tolist(), metrics, (len(X_tr), len(X_te))


def analyse_url(model, columns, url):
    f = extract_features(url)
    prob = float(model.predict_proba(pd.DataFrame([f])[columns])[0][1])
    risk = "HIGH" if prob >= 0.8 else "MEDIUM" if prob >= 0.5 else "LOW"
    return {"url": url,
            "prediction": "phishing" if prob >= 0.5 else "legitimate",
            "phishing_probability": round(prob, 3),
            "risk_level": risk,
            "indicators": indicators(url, f)}


def save_charts(model, columns, cm):
    """Figure 3: confusion matrix + top 10 feature importances."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    a1.imshow(cm, cmap="Blues")
    a1.set_title("Confusion Matrix (test set)")
    a1.set_xticks([0, 1], ["Legitimate", "Phishing"])
    a1.set_yticks([0, 1], ["Legitimate", "Phishing"])
    a1.set_xlabel("Predicted")
    a1.set_ylabel("Actual")
    top = max(max(r) for r in cm)
    for i in range(2):
        for j in range(2):
            a1.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > top / 2 else "black", fontsize=14)
    imp = pd.Series(model.feature_importances_, index=columns).nlargest(10)[::-1]
    a2.barh(imp.index, imp.values, color="#1f4e79")
    a2.set_title("Top 10 Feature Importances")
    a2.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150)
    plt.close()


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    dataset_path = args[0] if args else "phish.csv"
    model, columns, metrics, (n_train, n_test) = train_and_evaluate(dataset_path)
    print(f"Training samples: {n_train} | Testing samples: {n_test} | Features: {len(columns)}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:", metrics["confusion_matrix"])

    test_urls = [
        "https://www.wikipedia.org/wiki/Digital_forensics",
        "http://192.168.10.25/secure/login/verify-account.php",
        "http://paypal.com.account-verify.secure-login.example-update.xyz/signin?user=abc@mail.com",
        "https://docs.python.org/3/library/urllib.html",
        "http://bit.ly/3xYz12A",
    ]
    results = [analyse_url(model, columns, u) for u in test_urls]
    print("\nURL analysis results:")
    for r in results:
        print(f"[{r['risk_level']:<6}] {r['prediction']:<10} "
              f"p={r['phishing_probability']:.2f}  {r['url'][:60]}")

    report = {"model": "RandomForest (200 trees)", "dataset_size": n_train + n_test,
              "metrics": metrics, "url_reports": results}
    with open(REPORT_FILE, "w") as fh:
        json.dump(report, fh, indent=4)
    save_charts(model, columns, metrics["confusion_matrix"])
    print(f"\nPhishing URL analysis complete. Report saved to {REPORT_FILE}, chart to {CHART_FILE}")


if __name__ == "__main__":
    main()
