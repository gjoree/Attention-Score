import cv2
import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

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
        rms  = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
        flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        zcr  = librosa.feature.zero_crossing_rate(y=y, frame_length=hop * 2, hop_length=hop)[0]
        has_audio = True
    except Exception:
        has_audio = False
        rms = flux = zcr = np.zeros(duration)

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
    prev_gray = None
    last_cut = 0

    for t in range(duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Motion energy
        motion = 0.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            motion = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
        prev_gray = gray

        # Face presence
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        face_area = float(sum(fw * fh for (_, _, fw, fh) in faces)) / (w * h)
        face_count = int(len(faces))

        # Visual complexity (image entropy)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / (hist.sum() + 1e-9)
        entropy = float(-np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0])))

        # Color saturation and brightness
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = float(hsv[:, :, 1].mean()) / 255.0
        brightness = float(hsv[:, :, 2].mean()) / 255.0

        # Scene cut / staleness
        is_cut = 1 if t in cut_seconds else 0
        if is_cut:
            last_cut = t
        seconds_since_cut = t - last_cut

        # Safe audio index
        idx = min(t, len(rms) - 1)
        audio_rms  = float(np.sqrt(max(0, rms[idx]))) if has_audio else 0.0
        audio_flux = float(flux[idx]) if idx < len(flux) else 0.0
        audio_zcr  = float(zcr[idx]) if idx < len(zcr) else 0.0

        # Normalized position (attention decays over time — known effect)
        position = t / max(duration - 1, 1)

        results.append({
            "t":                 t,
            "motion":            round(motion, 4),
            "face_area":         round(face_area, 4),
            "face_count":        face_count,
            "audio_rms":         round(audio_rms, 4),
            "audio_flux":        round(audio_flux, 4),
            "audio_zcr":         round(audio_zcr, 4),
            "entropy":           round(entropy, 4),
            "saturation":        round(saturation, 4),
            "brightness":        round(brightness, 4),
            "is_cut":            is_cut,
            "seconds_since_cut": seconds_since_cut,
            "position":          round(position, 4),
        })

    cap.release()
    return results
