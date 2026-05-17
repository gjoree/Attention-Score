import json
import os

import joblib
import numpy as np
import xgboost as xgb
from scipy.ndimage import gaussian_filter1d

MODEL_PATH = "attention_model.joblib"

FEATURE_COLS = [
    # Motion
    "motion", "motion_delta",
    # Face
    "face_area", "face_count", "face_centrality",
    # Visual quality
    "entropy", "contrast", "edge_density", "saturation", "brightness", "color_variety",
    # Excitement spikes
    "brightness_delta", "saturation_delta",
    # Scene pacing
    "is_cut", "seconds_since_cut", "cut_rate_10s",
    # Audio
    "audio_rms", "audio_flux", "audio_zcr", "spectral_centroid", "audio_delta",
    # Context
    "position",
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


def _is_blank_frame(f: dict) -> bool:
    """True when a frame has essentially no visual content (black screen, solid colour, etc.)"""
    return (
        f.get("brightness", 0) < 0.06 and
        f.get("contrast",   0) < 0.03 and
        f.get("motion",     0) < 0.05 and
        f.get("face_area",  0) < 0.01
    )


def predict_curve(features: list[dict]) -> list[dict]:
    model = _load()
    rows = [[f.get(c, 0.0) for c in FEATURE_COLS] for f in features]

    rule_scores = [_rule_based(f) for f in features]

    if model and rows:
        ml_scores = model.predict(np.array(rows)).tolist()
        raw = []
        for ml, rule, f in zip(ml_scores, rule_scores, features):
            if _is_blank_frame(f):
                # XGBoost extrapolates poorly to all-zero out-of-distribution inputs
                score = min(float(ml), 15.0)
            elif f.get("face_count", 0) == 0:
                # No faces — model was trained mostly on face-heavy content so it's
                # unreliable here. Let rule-based carry more weight so excitement
                # signals (fireworks, motion, audio spikes) aren't buried.
                score = 0.45 * float(ml) + 0.55 * rule
            else:
                score = 0.75 * float(ml) + 0.25 * rule
            raw.append(score)
    else:
        raw = rule_scores

    smoothed = gaussian_filter1d(raw, sigma=2.5).tolist()
    return [
        {"t": f["t"], "score": round(min(100, max(0, s)))}
        for f, s in zip(features, smoothed)
    ]


def _rule_based(f: dict) -> float:
    """Fallback before first training — uses known attention correlates.
    Note: kids/animals not detectable from raw video features — use Gemini engine for those signals."""
    # Excitement spike: sudden brightness/colour burst (explosion, flash) gets a bonus
    # regardless of whether faces are present
    excitement = min(1.0, f.get("brightness_delta", 0) * 4 + f.get("saturation_delta", 0) * 3)

    score = (
        f.get("motion", 0)           * 16 +
        f.get("motion_delta", 0)     * 10 +
        f.get("face_area", 0)        * 20 +
        f.get("face_centrality", 0)  * 8  +
        f.get("audio_rms", 0)        * 16 +
        f.get("audio_flux", 0)       * 12 +
        f.get("audio_delta", 0)      * 8  +
        f.get("contrast", 0)         * 8  +
        f.get("color_variety", 0)    * 6  +
        f.get("cut_rate_10s", 0)     * 2  +
        f.get("is_cut", 0)           * 5  +
        excitement                   * 35 +  # fireworks/flash bonus
        (1 - f.get("position", 0))   * 8
    ) * 1.05
    return min(100.0, max(0.0, score))
