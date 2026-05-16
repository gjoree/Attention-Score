import { useRef, useState, useEffect, useCallback } from 'react';
import { useAttentionTracker } from './hooks/useAttentionTracker';
import { AttentionChart } from './components/AttentionChart';
import { PredictionChart } from './components/PredictionChart';
import { VideoHeatmap } from './components/VideoHeatmap';
import { DemographicsModal } from './components/DemographicsModal';
import { uploadVideo, pollVideoReady, saveSession, predictCurve } from './api';

const STATUS_LABEL = {
  initializing: 'Loading model…',
  ready: 'Ready',
  tracking: 'Tracking',
  error: 'Error',
};

const STATUS_COLOR = {
  initializing: '#f5a623',
  ready: '#888',
  tracking: '#30D158',
  error: '#ff453a',
};

function scoreColor(score) {
  if (score >= 67) return '#30D158';
  if (score >= 34) return '#f5a623';
  return '#ff453a';
}

export default function App() {
  const webcamVideoRef = useRef(null);
  const canvasRef      = useRef(null);
  const videoPlayerRef = useRef(null);

  // Video state
  const [videoName,      setVideoName]      = useState(null);
  const [videoId,        setVideoId]        = useState(null);   // backend video_id
  const [videoReady,     setVideoReady]     = useState(false);  // features processed
  const [uploading,      setUploading]      = useState(false);
  const [predictedCurve, setPredictedCurve] = useState(null);
  const [isPredicting,   setIsPredicting]   = useState(false);
  const [predictError,   setPredictError]   = useState(null);
  const [modelUsed,      setModelUsed]      = useState(null);

  // Webcam / session state
  const [webcamActive,   setWebcamActive]   = useState(false);
  const [showModal,      setShowModal]      = useState(false);
  const [demographics,   setDemographics]   = useState(null);  // { ageGroup, gender }
  const [sessionSaved,   setSessionSaved]   = useState(false);
  const [saveError,      setSaveError]      = useState(null);
  const [isSaving,       setIsSaving]       = useState(false);
  const [avgScore,       setAvgScore]       = useState(null);

  // Attention data synced to video time (key = Math.floor(video.currentTime))
  const sessionDataRef = useRef({});  // { [videoSecond]: score }
  const scoreRef       = useRef(0);   // mirrors currentScore without closure staleness

  const { status, scores, currentScore, indicators, startTracking, stopTracking } =
    useAttentionTracker(webcamVideoRef, canvasRef);

  // Keep scoreRef in sync
  useEffect(() => { scoreRef.current = currentScore; }, [currentScore]);

  // Auto-start tracking once model + webcam are both ready
  useEffect(() => {
    if (status === 'ready' && webcamActive) startTracking();
  }, [status, webcamActive, startTracking]);

  // Rolling session average
  useEffect(() => {
    if (scores.length < 2) return;
    const avg = scores.reduce((s, e) => s + e.score, 0) / scores.length;
    setAvgScore(Math.round(avg));
  }, [scores]);

  // Collect one data point per second, keyed to video time (not wall clock).
  // Only runs when a video is loaded AND it's actually playing (not paused/at t=0).
  useEffect(() => {
    if (status !== 'tracking' || !videoName) return;
    const id = setInterval(() => {
      const video = videoPlayerRef.current;
      if (!video || video.paused || video.currentTime === 0) return;
      const vt = Math.floor(video.currentTime);
      sessionDataRef.current[vt] = scoreRef.current;
    }, 1000);
    return () => clearInterval(id);
  }, [status, videoName]);

  // ── Video upload ──────────────────────────────────────────────────────────

  async function loadVideoFile(file) {
    if (!file || !file.type.startsWith('video/')) return;

    // Play locally via object URL immediately (no waiting for upload)
    const localUrl = URL.createObjectURL(file);
    videoPlayerRef.current.src = localUrl;
    videoPlayerRef.current.play();
    setVideoName(file.name);
    setVideoId(null);
    setVideoReady(false);
    setSessionSaved(false);
    setSaveError(null);
    setPredictedCurve(null);
    setPredictError(null);
    sessionDataRef.current = {};

    // Upload to backend in parallel
    setUploading(true);
    try {
      const { video_id } = await uploadVideo(file);
      setVideoId(video_id);

      // Poll until feature extraction is done (~10-20s)
      const cleanup = pollVideoReady(video_id, () => {
        setVideoReady(true);
        setUploading(false);
      });

      // Safety cleanup if component unmounts during polling
      return cleanup;
    } catch (err) {
      console.error('Upload failed:', err);
      setUploading(false);
    }
  }

  function handleFileInput(e) {
    loadVideoFile(e.target.files[0]);
  }

  function handleDrop(e) {
    e.preventDefault();
    loadVideoFile(e.dataTransfer.files[0]);
  }

  // ── Camera / demographics ─────────────────────────────────────────────────

  function handleActivateCameraClick() {
    setShowModal(true); // show demographics form first
  }

  async function handleDemographicsConfirm(demo) {
    setShowModal(false);
    setDemographics(demo);
    setSessionSaved(false);
    setSaveError(null);
    sessionDataRef.current = {};

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
      });
      const vid = webcamVideoRef.current;
      vid.srcObject = stream;
      vid.onloadedmetadata = () => {
        vid.play();
        setWebcamActive(true);
      };
    } catch {
      alert('Camera permission denied. Please allow camera access and try again.');
    }
  }

  function handleStopCamera() {
    webcamVideoRef.current?.srcObject?.getTracks().forEach((t) => t.stop());
    if (webcamVideoRef.current) webcamVideoRef.current.srcObject = null;
    setWebcamActive(false);
    stopTracking();
  }

  // ── Mode B: predict attention curve ──────────────────────────────────────

  async function handlePredict() {
    if (!videoId || !videoReady) return;
    setIsPredicting(true);
    setPredictError(null);
    setPredictedCurve(null);
    try {
      const data = await predictCurve(videoId);
      setPredictedCurve(data.curve);
      setModelUsed(data.model ?? 'rule-based');
    } catch (err) {
      setPredictError('Prediction failed — is the backend running?');
      console.error(err);
    } finally {
      setIsPredicting(false);
    }
  }

  // ── Save session ──────────────────────────────────────────────────────────

  const handleSaveSession = useCallback(async () => {
    if (!videoId || !demographics) return;

    const curve = Object.entries(sessionDataRef.current)
      .map(([t, score]) => ({ t: parseInt(t, 10), score }))
      .sort((a, b) => a.t - b.t);

    if (curve.length < 3) {
      setSaveError('Not enough data — watch at least a few seconds first.');
      return;
    }

    setIsSaving(true);
    setSaveError(null);
    try {
      await saveSession({
        videoId,
        ageGroup: demographics.ageGroup,
        gender:   demographics.gender,
        attentionCurve: curve,
      });
      setSessionSaved(true);
    } catch (err) {
      setSaveError('Save failed — is the backend running?');
      console.error(err);
    } finally {
      setIsSaving(false);
    }
  }, [videoId, demographics]);

  // Auto-save when video ends
  useEffect(() => {
    const vid = videoPlayerRef.current;
    if (!vid) return;
    const onEnded = () => { if (videoId && demographics) handleSaveSession(); };
    vid.addEventListener('ended', onEnded);
    return () => vid.removeEventListener('ended', onEnded);
  }, [videoId, demographics, handleSaveSession]);

  // ── Render ────────────────────────────────────────────────────────────────

  const canSave = videoId && demographics && status === 'tracking' && !sessionSaved;

  return (
    <div className="app">
      {showModal && <DemographicsModal onConfirm={handleDemographicsConfirm} />}

      {/* ── Header ── */}
      <header className="header">
        <div className="header-logo">
          <span className="header-icon">👁</span>
          <span className="header-title">Attention Tracker</span>
        </div>

        <div className="header-right">
          {/* Backend status indicators */}
          {uploading && (
            <div className="upload-status">
              <span className="spinner" /> Extracting video features…
            </div>
          )}
          {videoReady && !uploading && (
            <div className="upload-status ready">Features ready</div>
          )}

          <div className="status-pill" style={{ '--color': STATUS_COLOR[status] }}>
            <span className="status-dot" />
            {STATUS_LABEL[status]}
          </div>
        </div>
      </header>

      {/* ── Main layout ── */}
      <div className="content">

        {/* Left: video player */}
        <div className="panel video-panel">
          <div className="panel-label">VIDEO</div>
          <div
            className="video-drop-zone"
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => document.getElementById('video-input').click()}
          >
            <video
              ref={videoPlayerRef}
              className="video-player"
              controls
              playsInline
              style={{ display: videoName ? 'block' : 'none' }}
            />
            {!videoName && (
              <div className="drop-prompt">
                <div className="drop-icon">▶</div>
                <div>Drop a video file here</div>
                <div className="drop-sub">or click to browse</div>
              </div>
            )}
            <input
              id="video-input"
              type="file"
              accept="video/*"
              style={{ display: 'none' }}
              onChange={handleFileInput}
            />
          </div>

          {/* Attention heatmap — shown directly below video after prediction */}
          <VideoHeatmap curve={predictedCurve} videoRef={videoPlayerRef} />

          {videoName && (
            <div className="file-badge">
              <span className="file-icon">🎬</span>
              {videoName}
              {uploading && <span className="badge-status"> · uploading…</span>}
              {videoReady && !uploading && <span className="badge-status ready"> · ready</span>}
            </div>
          )}

          {/* Mode B predict button */}
          {videoReady && (
            <div className="predict-row">
              <button
                className="btn btn-predict"
                onClick={handlePredict}
                disabled={isPredicting}
              >
                {isPredicting ? (
                  <><span className="spinner" /> Predicting…</>
                ) : predictedCurve ? (
                  'Re-run Prediction'
                ) : (
                  'Predict Attention'
                )}
              </button>
              {predictedCurve && (
                <span className="predict-badge">
                  {modelUsed === 'gemini' ? '✦ Gemini AI' : 'Rule-based'}
                </span>
              )}
            </div>
          )}
          {predictError && <div className="save-error">{predictError}</div>}
        </div>

        {/* Right: webcam + scores */}
        <div className="panel right-panel">
          <div className="panel-label">WEBCAM</div>

          <div className="webcam-wrapper">
            <video
              ref={webcamVideoRef}
              autoPlay
              playsInline
              muted
              className="webcam-video"
              style={{ transform: 'scaleX(-1)' }}
            />
            <canvas
              ref={canvasRef}
              className="webcam-canvas"
              style={{ transform: 'scaleX(-1)' }}
            />
            {!webcamActive && (
              <div className="webcam-placeholder">
                <div className="cam-icon">📷</div>
                <div>Camera inactive</div>
              </div>
            )}
          </div>

          <div className="cam-controls">
            {!webcamActive ? (
              <button className="btn btn-primary" onClick={handleActivateCameraClick}>
                Activate Camera
              </button>
            ) : (
              <>
                <button className="btn btn-danger" onClick={handleStopCamera}>
                  Stop Camera
                </button>
                <button
                  className="btn btn-save"
                  onClick={handleSaveSession}
                  disabled={!canSave || isSaving}
                >
                  {isSaving ? 'Saving…' : sessionSaved ? 'Saved ✓' : 'Save Session'}
                </button>
              </>
            )}
          </div>

          {demographics && (
            <div className="demo-badge">
              {demographics.ageGroup} · {demographics.gender}
            </div>
          )}

          {saveError && <div className="save-error">{saveError}</div>}
          {sessionSaved && (
            <div className="save-success">
              Session saved — contributing to panel data
            </div>
          )}

          {/* Score display */}
          <div className="score-section">
            <div className="score-big" style={{ color: scoreColor(currentScore) }}>
              {status === 'tracking' ? `${currentScore}%` : '—'}
            </div>
            <div className="score-label">Attention Score</div>

            <div className="indicators">
              <Indicator
                label="Face"
                value={indicators.faceDetected ? 'Detected' : 'Not found'}
                ok={indicators.faceDetected}
              />
              <Indicator
                label="Eyes"
                value={`${indicators.eyesOpen}%`}
                ok={indicators.eyesOpen > 50}
              />
              <Indicator
                label="Head"
                value={`${indicators.headPose}%`}
                ok={indicators.headPose > 60}
              />
            </div>

            {avgScore !== null && status === 'tracking' && (
              <div className="avg-score">
                Session avg:{' '}
                <strong style={{ color: scoreColor(avgScore) }}>{avgScore}%</strong>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Chart ── */}
      <div className={`chart-panel${predictedCurve ? ' chart-panel--b' : ''}`}>
        {predictedCurve ? (
          <>
            <div className="panel-label">
              PREDICTED ATTENTION CURVE
              <span className="mode-tag mode-b">MODE B</span>
              <button
                className="clear-prediction"
                onClick={() => { setPredictedCurve(null); setModelUsed(null); }}
                title="Clear prediction"
              >
                ✕
              </button>
            </div>
            <PredictionChart curve={predictedCurve} />
          </>
        ) : (
          <>
            <div className="panel-label">
              LIVE ATTENTION TIMELINE
              <span className="mode-tag mode-a">MODE A</span>
            </div>
            <div className="chart-wrapper">
              {scores.length > 1 ? (
                <AttentionChart scores={scores} />
              ) : (
                <div className="chart-empty">
                  {status === 'tracking' ? 'Collecting data…' : 'Start tracking to see the timeline'}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Indicator({ label, value, ok }) {
  return (
    <div className="indicator">
      <span className={`indicator-dot ${ok ? 'ok' : 'off'}`} />
      <div className="indicator-text">
        <div className="indicator-label">{label}</div>
        <div className="indicator-value">{value}</div>
      </div>
    </div>
  );
}
