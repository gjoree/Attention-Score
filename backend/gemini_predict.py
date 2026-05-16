import json
import os
import time

import google.generativeai as genai
from scipy.ndimage import gaussian_filter1d


def predict_with_gemini(video_path: str, duration: int) -> list[dict] | None:
    """
    Upload video to Gemini, ask it to score attention per 5-second segment,
    interpolate to per-second curve. Returns None if no API key or on error.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        genai.configure(api_key=api_key)

        # Upload video — Gemini analyses the full video (motion, audio, transitions)
        print(f"[gemini] Uploading {video_path} …")
        video_file = genai.upload_file(path=video_path)

        # Wait for Gemini to finish processing the video
        for _ in range(30):
            video_file = genai.get_file(video_file.name)
            if video_file.state.name != "PROCESSING":
                break
            time.sleep(2)

        if video_file.state.name == "FAILED":
            print("[gemini] Video processing failed")
            return None

        model = genai.GenerativeModel("gemini-3.1-flash-lite")

        prompt = f"""You are analyzing a video to predict how glued viewers will be to the screen at each moment.

The score represents VIEWER ENGAGEMENT — would a real person look away or lean forward?

Score 0–100 (higher = more captivating, viewer cannot look away):
- 95–100: jaw-dropping moments — fireworks exploding, dramatic reveals, jump scares, fast action climax
- 75–94:  very engaging — people talking directly to camera, exciting music, vibrant fast-paced visuals
- 50–74:  moderate — decent content, viewer watches but not captivated
- 20–49:  low — slow, repetitive, monotonous narration, static shots with nothing happening
- 0–19:   viewer looks away — black screen, silence, blank frame, filler, no content whatsoever

KEY RULES:
- Fireworks bursting, explosions, bright flashes = ALWAYS 90+
- Black or near-black screen = ALWAYS below 15
- Sudden dramatic change (cut, flash, boom) = spike UP immediately
- Long static shot of nothing = drop DOWN progressively

The video is {duration} seconds long.

Return ONLY a valid JSON array, no markdown, no explanation:
[{{"t": 0, "score": 75}}, {{"t": 5, "score": 82}}, {{"t": 10, "score": 60}}, ...]

Cover from t=0 to t={duration} in 5-second steps."""

        response = model.generate_content([video_file, prompt])

        # Clean up the uploaded file from Gemini storage
        try:
            genai.delete_file(video_file.name)
        except Exception:
            pass

        text = response.text.strip()
        # Strip markdown code fences if Gemini wraps the JSON
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        start = text.find("[")
        end   = text.rfind("]") + 1
        if start == -1 or end == 0:
            print(f"[gemini] Could not parse JSON from response: {text[:200]}")
            return None

        sparse = json.loads(text[start:end])
        print(f"[gemini] Got {len(sparse)} keyframes from Gemini")

        return _interpolate(sparse, duration)

    except Exception as e:
        print(f"[gemini] Error: {e}")
        return None


def _interpolate(sparse: list[dict], duration: int) -> list[dict]:
    """Interpolate sparse keyframes to per-second scores with smoothing."""
    if not sparse:
        return []

    # Sort by t
    sparse = sorted(sparse, key=lambda p: p["t"])

    result = []
    for t in range(duration):
        # Find surrounding keyframes
        lo = sparse[0]
        hi = sparse[-1]
        for p in sparse:
            if p["t"] <= t:
                lo = p
            if p["t"] >= t and p == next((x for x in sparse if x["t"] >= t), sparse[-1]):
                hi = p
                break

        # Re-find cleanly
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

    # Smooth slightly
    raw      = [r["score"] for r in result]
    smoothed = gaussian_filter1d(raw, sigma=1.5).tolist()

    return [
        {"t": r["t"], "score": round(min(100, max(0, s)))}
        for r, s in zip(result, smoothed)
    ]
