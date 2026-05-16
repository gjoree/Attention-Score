import cv2
import librosa
import numpy as np

try:
    from scenedetect import detect, ContentDetector
    SCENEDETECT_OK = True
except Exception:
    SCENEDETECT_OK = False


def extract_features(video_path: str) -> list[dict]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = max(1, int(total_frames / fps))

    # ── Audio (whole file at once) ─────────────────────────────────────────────
    try:
        y, sr = librosa.load(video_path, sr=22050, mono=True)
        hop = sr  # 1-second hops
        rms      = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
        flux     = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        zcr      = librosa.feature.zero_crossing_rate(y=y, frame_length=hop * 2, hop_length=hop)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        has_audio = True
    except Exception:
        has_audio = False
        rms = flux = zcr = centroid = np.zeros(duration)

    # ── Scene cuts ────────────────────────────────────────────────────────────
    cut_seconds = set()
    if SCENEDETECT_OK:
        try:
            scenes = detect(video_path, ContentDetector(threshold=27))
            for scene in scenes:
                cut_seconds.add(int(scene[0].get_seconds()))
        except Exception:
            pass

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    results = []
    prev_gray   = None
    prev_motion = 0.0
    prev_rms    = 0.0
    last_cut    = 0

    for t in range(duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # ── Motion energy ─────────────────────────────────────────────────────
        motion = 0.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            motion = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
        prev_gray = gray

        # Rate of motion change — sudden bursts grab attention
        motion_delta = abs(motion - prev_motion)
        prev_motion = motion

        # ── Face presence & centrality ────────────────────────────────────────
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        face_area  = float(sum(fw * fh for (_, _, fw, fh) in faces)) / (w * h)
        face_count = int(len(faces))

        # How centred is the dominant face (0 = off-centre / no face, 1 = dead-centre)
        face_centrality = 0.0
        if len(faces) > 0:
            # Largest face by area
            fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            cx, cy = fx + fw / 2, fy + fh / 2
            # Normalised distance from frame centre (0 = centre, 1 = corner)
            dist = np.sqrt(((cx - w / 2) / (w / 2)) ** 2 + ((cy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
            face_centrality = float(1.0 - dist)  # flip: 1 = centred

        # ── Visual quality signals ────────────────────────────────────────────
        # Entropy (image complexity)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / (hist.sum() + 1e-9)
        entropy = float(-np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0])))

        # RMS contrast (flat/washed-out vs punchy)
        contrast = float(np.std(gray) / 255.0)

        # Edge density — visual detail independent of brightness
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges > 0))

        # Color saturation, brightness, and hue variety
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation   = float(hsv[:, :, 1].mean()) / 255.0
        brightness   = float(hsv[:, :, 2].mean()) / 255.0
        # Spread of hue values — colourful scenes score higher
        color_variety = float(np.std(hsv[:, :, 0]) / 90.0)  # hue std, max ~90 for full spectrum

        # ── Scene cut / staleness ─────────────────────────────────────────────
        is_cut = 1 if t in cut_seconds else 0
        if is_cut:
            last_cut = t
        seconds_since_cut = t - last_cut

        # Edit pace — how many cuts in the past 10 seconds
        cut_rate_10s = sum(1 for cs in cut_seconds if t - 10 <= cs <= t)

        # ── Audio features ────────────────────────────────────────────────────
        idx = min(t, len(rms) - 1)
        audio_rms_raw = float(np.sqrt(max(0, rms[idx]))) if has_audio else 0.0
        audio_flux    = float(flux[idx]) if idx < len(flux) else 0.0
        audio_zcr     = float(zcr[idx]) if idx < len(zcr) else 0.0

        # Spectral centroid: speech sits ~2-4 kHz, music varies widely; normalise to 0-1
        sp_centroid = float(centroid[idx]) if has_audio and idx < len(centroid) else 0.0
        spectral_centroid = min(1.0, sp_centroid / 8000.0)

        # Rate of audio energy change — sudden sound change = attention trigger
        audio_delta = abs(audio_rms_raw - prev_rms)
        prev_rms = audio_rms_raw

        # ── Position ─────────────────────────────────────────────────────────
        position = t / max(duration - 1, 1)

        results.append({
            "t":                 t,
            # Motion
            "motion":            round(motion, 4),
            "motion_delta":      round(motion_delta, 4),
            # Face
            "face_area":         round(face_area, 4),
            "face_count":        face_count,
            "face_centrality":   round(face_centrality, 4),
            # Visual quality
            "entropy":           round(entropy, 4),
            "contrast":          round(contrast, 4),
            "edge_density":      round(edge_density, 4),
            "saturation":        round(saturation, 4),
            "brightness":        round(brightness, 4),
            "color_variety":     round(color_variety, 4),
            # Scene pacing
            "is_cut":            is_cut,
            "seconds_since_cut": seconds_since_cut,
            "cut_rate_10s":      cut_rate_10s,
            # Audio
            "audio_rms":         round(audio_rms_raw, 4),
            "audio_flux":        round(audio_flux, 4),
            "audio_zcr":         round(audio_zcr, 4),
            "spectral_centroid": round(spectral_centroid, 4),
            "audio_delta":       round(audio_delta, 4),
            # Context
            "position":          round(position, 4),
        })

    cap.release()
    return results
