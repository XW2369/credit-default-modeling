"""模型可解释性：SHAP（树模型专用）。"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from credit_model.data import load_train_test

FIG_DIR = Path("reports/figures")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def run_explain() -> None:
    _, _, Xte, _ = load_train_test()
    model = joblib.load(Path("models/best_model.joblib"))
    explainer = shap.TreeExplainer(model)
    sample = Xte.sample(min(500, len(Xte)), random_state=42)
    sv = explainer.shap_values(sample)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(sv, sample, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=120)
    plt.close()
    print("saved", FIG_DIR / "shap_summary.png")


if __name__ == "__main__":
    run_explain()
