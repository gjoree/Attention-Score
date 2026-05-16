import { useEffect, useRef, useState, useCallback } from 'react';
import { FaceLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';

const WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

const SMOOTHING_WINDOW = 12; // frames (~400ms at 30fps)
const SCORE_HISTORY_MS = 90_000; // keep 90 seconds in chart
const PEAK_FLOOR = 80; // minimum denominator for normalization — avoids inflating scores before peak builds up

export function useAttentionTracker(videoRef, canvasRef) {
  const [status, setStatus] = useState('initializing'); // initializing | ready | tracking | error
  const [scores, setScores] = useState([]);
  const [currentScore, setCurrentScore] = useState(0);
  const [indicators, setIndicators] = useState({
    faceDetected: false,
    eyesOpen: 0,
    headPose: 0,
  });

  const landmarkerRef = useRef(null);
  const rafRef = useRef(null);
  const activeRef = useRef(false);
  const rawScoreBuffer = useRef([]);
  const peakRef = useRef(PEAK_FLOOR); // tracks personal session max for normalization

  // ── Init MediaPipe ──────────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    async function init() {
      try {
        const resolver = await FilesetResolver.forVisionTasks(WASM_URL);
        const landmarker = await FaceLandmarker.createFromOptions(resolver, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
          outputFaceBlendshapes: true,
          runningMode: 'VIDEO',
          numFaces: 1,
        });
        if (!cancelled) {
          landmarkerRef.current = landmarker;
          setStatus('ready');
        }
      } catch {
        if (!cancelled) setStatus('error');
      }
    }
    init();
    return () => {
      cancelled = true;
    };
  }, []);

  // ── Detection loop ──────────────────────────────────────────────────────────
  const startTracking = useCallback(() => {
    if (!landmarkerRef.current || !videoRef.current) return;
    if (activeRef.current) return;
    activeRef.current = true;
    peakRef.current = PEAK_FLOOR; // reset calibration for each new session
    rawScoreBuffer.current = [];
    setStatus('tracking');

    function loop(ts) {
      if (!activeRef.current) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (video && canvas && video.readyState >= 2) {
        // Sync canvas logical size to video intrinsic size
        if (
          canvas.width !== video.videoWidth ||
          canvas.height !== video.videoHeight
        ) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        const result = landmarkerRef.current.detectForVideo(video, ts);

        // ── Draw face mesh overlay ──────────────────────────────────────────
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (result.faceLandmarks.length > 0) {
          const du = new DrawingUtils(ctx);
          du.drawConnectors(
            result.faceLandmarks[0],
            FaceLandmarker.FACE_LANDMARKS_TESSELATION,
            { color: '#ffffff18', lineWidth: 0.8 }
          );
          du.drawConnectors(
            result.faceLandmarks[0],
            FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
            { color: '#30D158', lineWidth: 1.5 }
          );
          du.drawConnectors(
            result.faceLandmarks[0],
            FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
            { color: '#30D158', lineWidth: 1.5 }
          );
          du.drawConnectors(
            result.faceLandmarks[0],
            FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
            { color: '#FF6B6B', lineWidth: 2 }
          );
          du.drawConnectors(
            result.faceLandmarks[0],
            FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
            { color: '#FF6B6B', lineWidth: 2 }
          );
        }

        // ── Compute attention ───────────────────────────────────────────────
        const { rawScore, indic } = computeAttention(result);

        rawScoreBuffer.current.push(rawScore);
        if (rawScoreBuffer.current.length > SMOOTHING_WINDOW)
          rawScoreBuffer.current.shift();

        const smooth = Math.round(
          rawScoreBuffer.current.reduce((a, b) => a + b, 0) /
            rawScoreBuffer.current.length
        );

        // Update personal peak, then normalize so each person's ceiling = 100%
        if (smooth > peakRef.current) peakRef.current = smooth;
        const normalized = Math.min(100, Math.round(smooth / peakRef.current * 100));

        setCurrentScore(normalized);
        setIndicators(indic);

        const now = Date.now();
        setScores((prev) => {
          const entry = { time: now, score: normalized };
          const cutoff = now - SCORE_HISTORY_MS;
          return [...prev.filter((s) => s.time > cutoff), entry];
        });
      }

      rafRef.current = requestAnimationFrame(loop);
    }

    rafRef.current = requestAnimationFrame(loop);
  }, [videoRef, canvasRef]);

  const stopTracking = useCallback(() => {
    activeRef.current = false;
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setStatus('ready');
  }, []);

  useEffect(() => {
    return () => {
      activeRef.current = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return { status, scores, currentScore, indicators, startTracking, stopTracking };
}

// ── Attention computation ─────────────────────────────────────────────────────

function computeAttention(result) {
  if (!result || result.faceLandmarks.length === 0) {
    return {
      rawScore: 0,
      indic: { faceDetected: false, eyesOpen: 0, headPose: 0 },
    };
  }

  const blendshapes = result.faceBlendshapes?.[0]?.categories ?? [];
  const bs = {};
  blendshapes.forEach((b) => {
    bs[b.categoryName] = b.score;
  });

  // Eyes: MediaPipe blink score is ~0.05 even when eyes are wide open — remap
  // so that the natural open-eye baseline (0.05) maps to 1.0, and ~0.65 = closed.
  const BLINK_OPEN   = 0.05;
  const BLINK_CLOSED = 0.65;
  const remapBlink = (v) => Math.max(0, Math.min(1, 1 - (v - BLINK_OPEN) / (BLINK_CLOSED - BLINK_OPEN)));
  const blinkL = bs['eyeBlinkLeft'] ?? 0;
  const blinkR = bs['eyeBlinkRight'] ?? 0;
  const eyesOpen = (remapBlink(blinkL) + remapBlink(blinkR)) / 2;

  // Head pose from landmark geometry
  const headScore = computeHeadScore(result.faceLandmarks[0]);

  const rawScore = Math.round(eyesOpen * headScore * 100);

  return {
    rawScore,
    indic: {
      faceDetected: true,
      eyesOpen: Math.round(eyesOpen * 100),
      headPose: Math.round(headScore * 100),
    },
  };
}

function computeHeadScore(landmarks) {
  // Landmark indices (MediaPipe 478-point model):
  //   1 = nose tip, 33 = left eye outer, 263 = right eye outer
  const nose = landmarks[1];
  const lEye = landmarks[33];
  const rEye = landmarks[263];

  const faceWidth = Math.abs(rEye.x - lEye.x);
  if (faceWidth < 0.01) return 0.5;

  // Yaw: nose should be centered between outer eye corners
  const eyeMidX = (lEye.x + rEye.x) / 2;
  const noseOffX = Math.abs(nose.x - eyeMidX) / faceWidth;
  const yawScore = Math.max(0, 1 - noseOffX * 3.5);

  // Pitch: nose tip should sit ~18% below eye midpoint (normalized coords)
  const eyeMidY = (lEye.y + rEye.y) / 2;
  const nosePitch = nose.y - eyeMidY;
  const pitchDiff = Math.abs(nosePitch - 0.18);
  const pitchScore = Math.max(0, 1 - pitchDiff * 6);

  return yawScore * 0.65 + pitchScore * 0.35;
}
