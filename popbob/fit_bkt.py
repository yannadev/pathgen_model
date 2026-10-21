"""Fit standard four-parameter BKT on train.csv and score test.csv once.

Run with the workspace's patched pyBKT 1.4.3 package:
    python fit_bkt.py

Outputs:
    skill_params.json       Deployable per-skill BKT parameters and metadata
    bkt_test_metrics.csv    Per-skill held-out comparison with borrowed parameters
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".deps"))
from pyBKT.models import Model  # noqa: E402


TRAIN = ROOT / "train.csv"
TEST = ROOT / "test.csv"
MAP = ROOT / "skill_map.csv"
JSON_OUT = ROOT / "skill_params.json"
METRICS_OUT = ROOT / "bkt_test_metrics.csv"
SKILLS = [f"S{i}" for i in range(1, 8)]
BORROWED = {"p_l0": 0.30, "p_t": 0.09, "p_g": 0.10, "p_s": 0.09}
DEFAULTS = {
    "user_id": "user_id",
    "skill_name": "skill_code",
    "correct": "y",
    "order_id": "time_done",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        dtype={"user_id": str, "skill_code": str, "time_done": "int64", "y": "int8"},
    )
    if data.columns.tolist() != ["user_id", "skill_code", "time_done", "y"]:
        raise ValueError(f"Unexpected columns in {path.name}")
    if data.isna().any().any() or not data.y.isin([0, 1]).all():
        raise ValueError(f"Missing or invalid values in {path.name}")
    return data


def extract_parameters(model: Model) -> dict[str, dict[str, float]]:
    params = model.params()
    key_map = {
        "p_l0": "prior",
        "p_t": "learns",
        "p_g": "guesses",
        "p_s": "slips",
    }
    fitted = {}
    for skill in SKILLS:
        values = {
            output_key: float(params.loc[(skill, internal_key, "default"), "value"])
            for output_key, internal_key in key_map.items()
        }
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in values.values()):
            raise ValueError(f"Invalid BKT probabilities for {skill}: {values}")
        fitted[skill] = values
    return fitted


def predict_sequence(outcomes: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    """One-step predictions; update mastery only after each observed outcome."""
    p = parameters["p_l0"]
    transition = parameters["p_t"]
    guess = parameters["p_g"]
    slip = parameters["p_s"]
    predictions = np.empty(len(outcomes), dtype=float)
    for i, outcome in enumerate(outcomes):
        chance_correct = p * (1 - slip) + (1 - p) * guess
        predictions[i] = chance_correct
        if outcome == 1:
            denominator = chance_correct
            posterior = p * (1 - slip) / denominator if denominator > 0 else p
        else:
            denominator = 1 - chance_correct
            posterior = p * slip / denominator if denominator > 0 else p
        p = posterior + (1 - posterior) * transition
        p = min(1.0, max(0.0, p))
    return predictions


def score(test_skill: pd.DataFrame, parameters: dict[str, float]) -> dict[str, float]:
    actual, predicted = [], []
    for _, group in test_skill.groupby("user_id", sort=False):
        outcomes = group.y.to_numpy()
        actual.extend(outcomes)
        predicted.extend(predict_sequence(outcomes, parameters))
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=float)
    clipped = np.clip(predicted, 1e-12, 1 - 1e-12)
    return {
        "auc": float(roc_auc_score(actual, predicted)),
        "brier": float(brier_score_loss(actual, predicted)),
        "log_loss": float(log_loss(actual, clipped, labels=[0, 1])),
    }


def main() -> None:
    train = load_rows(TRAIN)
    if set(train.skill_code) != set(SKILLS):
        raise ValueError("Training data must include all seven skills")
    print(f"Fitting standard BKT on {len(train):,} training answers...", flush=True)
    model = Model(seed=42, num_fits=5, parallel=False)
    model.fit(data=train, defaults=DEFAULTS, skills=SKILLS)
    fitted = extract_parameters(model)
    print("Training complete. Reading the held-out test file now.", flush=True)

    test = load_rows(TEST)
    if set(train.user_id) & set(test.user_id):
        raise ValueError("Train/test student overlap")
    if set(test.skill_code) != set(SKILLS):
        raise ValueError("Test data must include all seven skills")

    mapping = pd.read_csv(MAP, keep_default_na=False)
    if mapping.columns.tolist() != ["exercise", "problem_type", "skill_code"]:
        raise ValueError("Unexpected skill_map.csv columns")
    exercises = defaultdict(list)
    for row in mapping.itertuples(index=False):
        exercises[row.skill_code].append({"exercise": row.exercise, "problem_type": row.problem_type})

    rows = []
    skill_payload = {}
    for skill in SKILLS:
        train_skill = train[train.skill_code == skill]
        test_skill = test[test.skill_code == skill]
        trained = score(test_skill, fitted[skill])
        baseline = score(test_skill, BORROWED)
        rows.append({
            "skill_code": skill,
            "train_students": int(train_skill.user_id.nunique()),
            "train_answers": int(len(train_skill)),
            "test_students": int(test_skill.user_id.nunique()),
            "test_answers": int(len(test_skill)),
            "test_auc_fitted": trained["auc"],
            "test_auc_borrowed": baseline["auc"],
            "test_brier_fitted": trained["brier"],
            "test_brier_borrowed": baseline["brier"],
            "test_log_loss_fitted": trained["log_loss"],
            "test_log_loss_borrowed": baseline["log_loss"],
        })
        skill_payload[skill] = {
            **fitted[skill],
            "junyi_mapping": exercises[skill],
            "n_train_students": int(train_skill.user_id.nunique()),
            "n_train_answers": int(len(train_skill)),
            "n_test_students": int(test_skill.user_id.nunique()),
            "n_test_answers": int(len(test_skill)),
            "test_auc": trained["auc"],
            "test_auc_borrowed": baseline["auc"],
            "test_brier": trained["brier"],
            "test_brier_borrowed": baseline["brier"],
        }
        print(
            f"{skill}: AUC {trained['auc']:.3f} vs borrowed {baseline['auc']:.3f}; "
            f"Brier {trained['brier']:.3f} vs borrowed {baseline['brier']:.3f}",
            flush=True,
        )

    payload = {
        "dataset": "Junyi Academy Math Practicing Log, January 2015 release",
        "method": "standard BKT; pyBKT 1.4.3 EM; num_fits=5; seed=42; no forgetting",
        "label_rule": "y=1 iff first attempt correct without a hint (Junyi correct field)",
        "split": "student-disjoint; train <= 80000 rows; test <= 20000 rows; seed=42",
        "train_sha256": sha256(TRAIN),
        "test_sha256": sha256(TEST),
        "skill_map_sha256": sha256(MAP),
        "borrowed_parameters": BORROWED,
        "skills": skill_payload,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(METRICS_OUT, index=False, float_format="%.6f")
    print(f"Saved {JSON_OUT.name} and {METRICS_OUT.name}", flush=True)


if __name__ == "__main__":
    main()
