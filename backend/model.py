import json
import os

import joblib
import numpy as np
import xgboost as xgb
from scipy.ndimage import gaussian_filter1d

MODEL_PATH = "attention_model.joblib"

FEATURE_COLS = [
    "motion", "face_area", "face_count", "audio_rms", "audio_flux",
    "audio_zcr", "entropy", "saturation", "brightness",
    "is_cut", "seconds_since_cut", "position",
]

_model: xgb.XGBRegressor | None = None


def _load() -> xgb.XGBRegressor | None:
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
    return _model


def train(sessions: list, features_by_vid: dict) -> dict:
    X, y = [], []

    for session in sessions:
        vid = session["video_id"]
        if vid not in features_by_vid:
            continue
        feat_map = {f["t"]: f for f in features_by_vid[vid]}
        curve = json.loads(session["attention_json"])

        for point in curve:
            t = point["t"]
            if t not in feat_map:
                continue
            row = [feat_map[t].get(c, 0.0) for c in FEATURE_COLS]
            X.append(row)
            y.append(float(point["score"]))

    if len(X) < 5:
        return {"status": "not_enough_data", "samples": len(X), "needed": 5}

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        verbosity=0,
    )
    model.fit(np.array(X), np.array(y))
    joblib.dump(model, MODEL_PATH)

    global _model
    _model = model

    importance = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    return {
        "status": "trained",
        "samples": len(X),
        "feature_importance": importance,
    }


def predict_curve(features: list[dict]) -> list[dict]:
    model = _load()
    rows = [[f.get(c, 0.0) for c in FEATURE_COLS] for f in features]

    if model and rows:
        raw = model.predict(np.array(rows)).tolist()
    else:
        raw = [_rule_based(f) for f in features]

    smoothed = gaussian_filter1d(raw, sigma=2.5).tolist()
    return [
        {"t": f["t"], "score": round(min(100, max(0, s)))}
        for f, s in zip(features, smoothed)
    ]


def _rule_based(f: dict) -> float:
    """Fallback before first training — uses known attention correlates."""
    score = (
        f.get("motion", 0)      * 22 +
        f.get("face_area", 0)   * 28 +
        f.get("audio_rms", 0)   * 22 +
        f.get("audio_flux", 0)  * 18 +
        f.get("is_cut", 0)      * 10 +
        (1 - f.get("position", 0)) * 15
    ) * 1.35
    return min(100.0, max(0.0, score))
