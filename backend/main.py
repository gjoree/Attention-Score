import json
import shutil
import sqlite3
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from features import extract_features
from gemini_predict import predict_with_gemini
from model import predict_curve, train

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Attention Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("uploads").mkdir(exist_ok=True)

# ── Database ──────────────────────────────────────────────────────────────────
# SQLite — single file (sessions.db) created automatically in ./backend/
# Schema:
#   sessions       — one row per Mode A viewing session (panelist + attention curve)
#   video_features — one row per uploaded video (pre-extracted ML features)

DB_PATH = "sessions.db"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")  # set once — persists in the DB file
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            video_id    TEXT NOT NULL,
            age_group   TEXT NOT NULL DEFAULT 'unknown',
            gender      TEXT NOT NULL DEFAULT 'unknown',
            attention_json TEXT NOT NULL,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS video_features (
            video_id       TEXT PRIMARY KEY,
            filename       TEXT,
            features_json  TEXT NOT NULL,
            ready          INTEGER NOT NULL DEFAULT 0,
            created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


init_db()


# ── Videos ───────────────────────────────────────────────────────────────────

def _process_features_bg(video_id: str, path: str, filename: str) -> None:
    """Runs in background thread — extracts features and marks video ready."""
    try:
        features = extract_features(path)
        conn = get_db()
        conn.execute(
            "UPDATE video_features SET features_json=?, filename=?, ready=1 WHERE video_id=?",
            (json.dumps(features), filename, video_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[features] Error processing {video_id}: {e}")


@app.post("/videos/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a video. Returns immediately with video_id.
    Feature extraction runs in background (~10-20s for 60s video).
    Poll GET /videos/{video_id}/status to know when it's ready.
    """
    video_id = str(uuid.uuid4())
    # Keep original extension so OpenCV/librosa can detect format
    suffix = Path(file.filename).suffix or ".mp4"
    save_path = f"uploads/{video_id}{suffix}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Insert placeholder row immediately
    conn = get_db()
    conn.execute(
        "INSERT INTO video_features (video_id, filename, features_json, ready) VALUES (?,?,?,0)",
        (video_id, file.filename, "[]"),
    )
    conn.commit()
    conn.close()

    # Extract features in background so response is instant
    background_tasks.add_task(_process_features_bg, video_id, save_path, file.filename)

    return {"video_id": video_id, "status": "processing"}


@app.get("/videos/{video_id}/status")
async def video_status(video_id: str):
    conn = get_db()
    row = conn.execute(
        "SELECT ready, filename FROM video_features WHERE video_id=?", (video_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Video not found")
    return {"video_id": video_id, "ready": bool(row["ready"]), "filename": row["filename"]}


@app.get("/videos/{video_id}/predict")
async def predict(video_id: str):
    """Mode B: return predicted attention curve for a video."""
    conn = get_db()
    row = conn.execute(
        "SELECT features_json, ready, filename FROM video_features WHERE video_id=?", (video_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Video not found")
    if not row["ready"]:
        raise HTTPException(202, "Features still processing, try again in a few seconds")

    features = json.loads(row["features_json"])

    # Try Gemini first — it understands video content directly
    video_file = next(Path("uploads").glob(f"{video_id}.*"), None)
    curve = None
    model_used = "rule-based"

    if video_file:
        duration = features[-1]["t"] + 1 if features else 60
        curve = predict_with_gemini(str(video_file), duration)
        if curve:
            model_used = "gemini"

    # Fall back to XGBoost / rule-based if Gemini unavailable or errored
    if curve is None:
        curve = predict_curve(features)

    return {"video_id": video_id, "curve": curve, "model": model_used}


# ── Sessions ──────────────────────────────────────────────────────────────────

@app.post("/sessions")
async def save_session(data: dict):
    """
    Save a completed Mode A session.
    Body: { video_id, age_group, gender, attention_curve: [{t, score}, ...] }
    """
    required = {"video_id", "attention_curve"}
    if not required.issubset(data):
        raise HTTPException(400, f"Missing fields: {required - data.keys()}")

    session_id = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        "INSERT INTO sessions (id, video_id, age_group, gender, attention_json) VALUES (?,?,?,?,?)",
        (
            session_id,
            data["video_id"],
            data.get("age_group", "unknown"),
            data.get("gender", "unknown"),
            json.dumps(data["attention_curve"]),
        ),
    )
    conn.commit()
    conn.close()
    return {"session_id": session_id}


@app.get("/sessions")
async def list_sessions(video_id: str | None = None):
    conn = get_db()
    if video_id:
        rows = conn.execute(
            "SELECT id, video_id, age_group, gender, created_at FROM sessions WHERE video_id=?",
            (video_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, video_id, age_group, gender, created_at FROM sessions"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Panel dashboard data ───────────────────────────────────────────────────────

# US Census 2023 adult age distribution (simplified)
US_CENSUS = {
    "13-17": 0.07, "18-24": 0.13, "25-34": 0.18, "35-44": 0.17,
    "45-54": 0.16, "55-64": 0.15, "65+": 0.14,
}


@app.get("/videos/{video_id}/panel")
async def panel(video_id: str, reweight: bool = False):
    """Return panel attention curves, optionally reweighted to US Census."""
    conn = get_db()
    rows = conn.execute(
        "SELECT age_group, gender, attention_json FROM sessions WHERE video_id=?",
        (video_id,),
    ).fetchall()
    conn.close()

    if not rows:
        raise HTTPException(404, "No sessions for this video yet")

    # Group curves by age
    groups: dict[str, list[list]] = {}
    for row in rows:
        g = row["age_group"]
        groups.setdefault(g, []).append(json.loads(row["attention_json"]))

    total = sum(len(v) for v in groups.values())
    panel_dist = {g: len(v) / total for g, v in groups.items()}

    def avg_curve(curves):
        max_t = max(len(c) for c in curves)
        result = []
        for t in range(max_t):
            vals = [c[t]["score"] for c in curves if t < len(c)]
            result.append({"t": t, "score": round(sum(vals) / len(vals))})
        return result

    group_curves = {g: avg_curve(curves) for g, curves in groups.items()}

    def blend(weights):
        all_t = max((max(p["t"] for p in c) for c in group_curves.values()), default=0)
        result = []
        for t in range(all_t + 1):
            score, w_sum = 0.0, 0.0
            for g, curve in group_curves.items():
                pt = next((p for p in curve if p["t"] == t), None)
                if pt and g in weights:
                    score += pt["score"] * weights[g]
                    w_sum += weights[g]
            result.append({"t": t, "score": round(score / w_sum) if w_sum else 0})
        return result

    return {
        "panel_curve":      blend(panel_dist),
        "reweighted_curve": blend(US_CENSUS) if reweight else None,
        "breakdown":        {g: len(v) for g, v in groups.items()},
        "panel_dist":       panel_dist,
        "session_count":    total,
    }


# ── Model training ────────────────────────────────────────────────────────────

@app.post("/model/train")
async def train_model():
    """Retrain XGBoost on all collected session data."""
    conn = get_db()
    sessions = conn.execute("SELECT * FROM sessions").fetchall()
    feat_rows = conn.execute(
        "SELECT video_id, features_json FROM video_features WHERE ready=1"
    ).fetchall()
    conn.close()

    features_by_vid = {r["video_id"]: json.loads(r["features_json"]) for r in feat_rows}
    result = train([dict(s) for s in sessions], features_by_vid)
    return result


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    conn = get_db()
    session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    video_count = conn.execute(
        "SELECT COUNT(*) FROM video_features WHERE ready=1"
    ).fetchone()[0]
    conn.close()
    return {
        "status": "ok",
        "sessions": session_count,
        "videos_ready": video_count,
    }
