import json
import re
import time
from pathlib import Path

import cv2
import google.generativeai as genai
from scipy.ndimage import gaussian_filter1d

ENV_PATH = Path(__file__).resolve().parent / ".env"


def _read_env_value(key: str) -> str | None:
    if not ENV_PATH.exists():
        return None

    for raw_line in ENV_PATH.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, value = line.split("=", 1)
        name = name.strip()
        if name.startswith("export "):
            name = name[len("export "):].strip()

        if name == key:
            return value.strip().strip("\"'")

    return None


def _local_blank_ratio(video_path: str, sample_count: int = 12) -> float:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return 0.0
        step    = max(1, total // sample_count)
        blank   = 0
        checked = 0
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            if frame.mean() / 255.0 < 0.05:
                blank += 1
            checked += 1
            if checked >= sample_count:
                break
        cap.release()
        return blank / checked if checked else 0.0
    except Exception:
        return 0.0


def predict_with_gemini(video_path: str, duration: int) -> dict | None:
    """
    Upload video to Gemini, return:
      {"curve": [...], "status": "ok"|"issue", "issue": "...", "problems": [...]}
    or None on failure.
    """
    api_key = _read_env_value("GEMINI_API_KEY")
    if not api_key:
        return None

    blank_ratio = _local_blank_ratio(video_path)
    if blank_ratio >= 0.85:
        print(f"[gemini] Skipping API — {blank_ratio:.0%} of frames are blank")
        flat_curve = [{"t": t, "score": 1} for t in range(duration)]
        return {
            "curve":    flat_curve,
            "status":   "issue",
            "issue":    f"Video is blank or black ({blank_ratio:.0%} of sampled frames have no content).",
            "problems": [{
                "start":  0,
                "end":    duration,
                "type":   "blank screen",
                "detail": "The entire video is black with no visual content — viewers will immediately disengage.",
            }],
        }

    try:
        genai.configure(api_key=api_key)

        print(f"[gemini] Uploading {video_path} …")
        video_file = genai.upload_file(path=video_path)

        for _ in range(30):
            video_file = genai.get_file(video_file.name)
            if video_file.state.name != "PROCESSING":
                break
            time.sleep(2)

        if video_file.state.name == "FAILED":
            print("[gemini] Video processing failed")
            return None

        model = genai.GenerativeModel("gemini-3.1-flash-lite")

        last_t = duration - (duration % 5 or 5)
        prompt = f"""You are analysing a video to help its creator improve viewer retention.
Respond in EXACTLY three lines — STATUS, CURVE, PROBLEMS. No markdown, no code blocks, nothing else.

━━━ LINE 1 — STATUS ━━━
Check for technical problems (entirely black screen, corrupted frames, unreadable content).
  STATUS: OK
  (or if genuinely broken: STATUS: ISSUE — <short description>)

━━━ LINE 2 — CURVE ━━━
Score viewer engagement (0–100) every 5 seconds — how likely a real person is to keep watching.

SCALE:
0–4   Blank/black. Viewer has already looked away.
5–20  Nearly unwatchable. No movement, near-silence, near-black.
21–45 Low. Slow, repetitive, monotonous.
46–65 Moderate. Acceptable, viewer stays but is not engaged.
66–82 Engaging. Dynamic visuals, good energy, interesting content.
83–100 Cannot look away. Explosion, fireworks, dramatic reveal, baby/toddler, funny animal.

Boosters (always score higher):
- Baby or toddler visible → minimum 70
- Active cat, dog, or wildlife → minimum 65
- Fireworks / explosion → 88–98

CURVE: [{{"t":0,"score":N}},{{"t":5,"score":N}},...] covering t=0 to t={last_t}

━━━ LINE 3 — PROBLEMS ━━━
Identify every segment where viewer attention drops significantly (score falls below 45 OR drops 15+ points from a recent high and stays there). Each segment must be at least 3 seconds long.

For each problem use one of these TYPE labels — choose the most specific one that fits:
  boring content      — generic, uninteresting, nothing stands out
  slow-paced speech   — presenter speaks too slowly, long pauses between words
  static shot         — camera locked, nothing moves on screen
  dead air            — silence or near-silence with no compensating visual
  lack of visuals     — audio or text-heavy, no engaging imagery
  low energy          — flat/monotone delivery, no enthusiasm or variation
  monotonous footage  — same scene or action repeating without change
  poor lighting       — too dark, washed out, or visually unclear
  abrupt transition   — jarring cut or scene change that breaks immersion
  information overload — too much text or data on screen, viewers give up reading

DETAIL must be one concrete sentence describing EXACTLY what is happening at that moment (what is on screen, what the speaker is doing, what the audio sounds like).

If there are no problematic segments at all: PROBLEMS: []

PROBLEMS: [{{"start":N,"end":N,"type":"label","detail":"one sentence"}},...] or []

━━━ ABSOLUTE SCORING RULES ━━━
1. Completely/nearly black screen → 0–4. No exceptions.
2. Fireworks/explosion flash → 88–98 at that moment.
3. Baby or toddler visible → minimum 70.
4. Active cat or dog → minimum 65.

━━━ WORKED EXAMPLES ━━━

Example A — 60-second nature documentary with two dull sections:
STATUS: OK
CURVE: [{{"t":0,"score":74}},{{"t":5,"score":79}},{{"t":10,"score":28}},{{"t":15,"score":25}},{{"t":20,"score":71}},{{"t":25,"score":76}},{{"t":30,"score":73}},{{"t":35,"score":24}},{{"t":40,"score":21}},{{"t":45,"score":72}},{{"t":50,"score":77}},{{"t":55,"score":74}}]
PROBLEMS: [{{"start":10,"end":20,"type":"static shot","detail":"Slow horizontal pan across an empty sky with no animals, movement, or audio narration to hold attention."}},{{"start":35,"end":45,"type":"dead air","detail":"Wide shot of a forest floor in complete silence — no subjects visible, no movement, no sound."}}]

Example B — 30-second fireworks video:
STATUS: OK
CURVE: [{{"t":0,"score":55}},{{"t":5,"score":91}},{{"t":10,"score":88}},{{"t":15,"score":76}},{{"t":20,"score":93}},{{"t":25,"score":84}}]
PROBLEMS: []

Example C — 60-second product ad with a slow midsection:
STATUS: OK
CURVE: [{{"t":0,"score":60}},{{"t":5,"score":58}},{{"t":10,"score":27}},{{"t":15,"score":24}},{{"t":20,"score":26}},{{"t":25,"score":61}},{{"t":30,"score":65}},{{"t":35,"score":28}},{{"t":40,"score":25}},{{"t":45,"score":62}},{{"t":50,"score":66}},{{"t":55,"score":63}}]
PROBLEMS: [{{"start":10,"end":25,"type":"slow-paced speech","detail":"Presenter reads product specs in a flat monotone with 3-second pauses between each point, no visuals change on screen."}},{{"start":35,"end":45,"type":"lack of visuals","detail":"Audio continues but screen shows a static logo for 10 seconds with no animation or supporting imagery."}}]

Example D — 30-second kitten video:
STATUS: OK
CURVE: [{{"t":0,"score":78}},{{"t":5,"score":85}},{{"t":10,"score":82}},{{"t":15,"score":90}},{{"t":20,"score":88}},{{"t":25,"score":84}}]
PROBLEMS: []

Example E — all-black screen:
STATUS: ISSUE — The video is entirely black with no audio or visual content.
CURVE: [{{"t":0,"score":2}},{{"t":5,"score":1}},{{"t":10,"score":2}},{{"t":15,"score":1}},{{"t":20,"score":2}},{{"t":25,"score":1}}]
PROBLEMS: [{{"start":0,"end":{last_t},"type":"blank screen","detail":"The entire video is black with no visual content — viewers will immediately disengage and not return."}}]

━━━ NOW ANALYSE THIS VIDEO ({duration} seconds) ━━━
STATUS: OK  (or ISSUE — description)
CURVE: [{{"t":0,"score":N}},{{"t":5,"score":N}},...] covering t=0 to t={last_t}
PROBLEMS: [{{"start":N,"end":N,"type":"label","detail":"one sentence"}},...] or []"""

        response = model.generate_content([video_file, prompt])

        try:
            genai.delete_file(video_file.name)
        except Exception:
            pass

        text = response.text.strip()
        text = _strip_markdown(text)

        status, issue = _parse_status(text)
        sparse        = _parse_curve(text)
        problems      = _parse_problems(text)

        if sparse is None:
            print(f"[gemini] Could not parse curve from response:\n{text[:400]}")
            return None

        print(f"[gemini] status={status} | keyframes={len(sparse)} | problems={len(problems)}")

        return {
            "curve":    _interpolate(sparse, duration),
            "status":   status,
            "issue":    issue,
            "problems": problems,
        }

    except Exception as e:
        print(f"[gemini] Error: {e}")
        return None


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith(("json", "text", "plain")):
            text = text.split("\n", 1)[1] if "\n" in text else text[4:]
    return text.strip()


def _parse_status(text: str) -> tuple[str, str]:
    for line in text.splitlines():
        if line.upper().startswith("STATUS:"):
            value = line.split(":", 1)[1].strip()
            if value.upper().startswith("ISSUE"):
                desc = re.split(r"[—:\-]", value, maxsplit=1)
                return "issue", desc[1].strip() if len(desc) > 1 else value
            return "ok", ""
    return "ok", ""


def _parse_curve(text: str) -> list[dict] | None:
    idx = text.upper().find("CURVE:")
    search = text[idx:] if idx != -1 else text
    arr_start = search.find("[")
    # Don't grab the PROBLEMS array — stop before it
    problems_idx = search.upper().find("PROBLEMS:")
    search_end   = search[:problems_idx] if problems_idx != -1 else search
    arr_end = search_end.rfind("]") + 1
    if arr_start == -1 or arr_end == 0:
        return None
    raw = search_end[arr_start:arr_end].replace("'", '"')
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        pairs = re.findall(r'"t"\s*:\s*(\d+)\s*,\s*"score"\s*:\s*(\d+)', raw)
        if not pairs:
            return None
        data = [{"t": int(t), "score": int(s)} for t, s in pairs]
    if not isinstance(data, list) or not data:
        return None
    clean = []
    for pt in data:
        if not isinstance(pt, dict):
            continue
        t = pt.get("t")
        s = pt.get("score")
        if t is None or s is None:
            continue
        clean.append({"t": int(t), "score": max(0, min(100, int(s)))})
    return clean if clean else None


def _parse_problems(text: str) -> list[dict]:
    idx = text.upper().find("PROBLEMS:")
    if idx == -1:
        return []
    search    = text[idx:]
    arr_start = search.find("[")
    arr_end   = search.rfind("]") + 1
    if arr_start == -1 or arr_end == 0:
        return []
    raw = search[arr_start:arr_end].replace("'", '"')
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    clean = []
    for p in data:
        if not isinstance(p, dict):
            continue
        start  = p.get("start")
        end    = p.get("end")
        ptype  = p.get("type", "")
        detail = p.get("detail", "")
        if start is None or end is None:
            continue
        clean.append({
            "start":  int(start),
            "end":    int(end),
            "type":   str(ptype),
            "detail": str(detail),
        })
    return clean


# ── Interpolation ─────────────────────────────────────────────────────────────

def _interpolate(sparse: list[dict], duration: int) -> list[dict]:
    if not sparse:
        return []
    sparse = sorted(sparse, key=lambda p: p["t"])
    result = []
    for t in range(duration):
        lows  = [p for p in sparse if p["t"] <= t]
        highs = [p for p in sparse if p["t"] >= t]
        lo = lows[-1]  if lows  else sparse[0]
        hi = highs[0]  if highs else sparse[-1]
        if lo["t"] == hi["t"]:
            score = float(lo["score"])
        else:
            frac  = (t - lo["t"]) / (hi["t"] - lo["t"])
            score = lo["score"] + (hi["score"] - lo["score"]) * frac
        result.append({"t": t, "score": score})
    raw      = [r["score"] for r in result]
    smoothed = gaussian_filter1d(raw, sigma=1.5).tolist()
    return [
        {"t": r["t"], "score": round(min(100, max(0, s)))}
        for r, s in zip(result, smoothed)
    ]
