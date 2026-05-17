import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { PredictionChart } from '../components/PredictionChart';
import { VideoHeatmap }    from '../components/VideoHeatmap';
import { uploadVideo, pollVideoReady, predictCurve } from '../api';

export default function ModeB() {
  const navigate       = useNavigate();
  const videoPlayerRef = useRef(null);

  const [videoName,  setVideoName]  = useState(null);
  const [videoId,    setVideoId]    = useState(null);
  const [videoReady, setVideoReady] = useState(false);
  const [uploading,  setUploading]  = useState(false);

  // Gemini state
  const [geminiCurve,    setGeminiCurve]    = useState(null);
  const [geminiProblems, setGeminiProblems] = useState([]);
  const [geminiIssue,    setGeminiIssue]    = useState(null);
  const [geminiLoading,  setGeminiLoading]  = useState(false);
  const [geminiError,    setGeminiError]    = useState(null);

  // Local model state
  const [localCurve,   setLocalCurve]   = useState(null);
  const [localLoading, setLocalLoading] = useState(false);
  const [localError,   setLocalError]   = useState(null);

  async function loadVideoFile(file) {
    if (!file || !file.type.startsWith('video/')) return;
    const localUrl = URL.createObjectURL(file);
    videoPlayerRef.current.src = localUrl;
    videoPlayerRef.current.play();
    setVideoName(file.name);
    setVideoId(null);
    setVideoReady(false);
    setGeminiCurve(null); setGeminiProblems([]); setGeminiIssue(null); setGeminiError(null);
    setLocalCurve(null);  setLocalError(null);
    setUploading(true);
    try {
      const { video_id } = await uploadVideo(file);
      setVideoId(video_id);
      const cleanup = pollVideoReady(video_id, () => { setVideoReady(true); setUploading(false); });
      return cleanup;
    } catch {
      setUploading(false);
    }
  }

  async function handleGemini() {
    if (!videoId || !videoReady || geminiLoading) return;
    setGeminiLoading(true);
    setGeminiError(null);
    setGeminiCurve(null);
    setGeminiProblems([]);
    setGeminiIssue(null);
    try {
      const data = await predictCurve(videoId, 'gemini');
      setGeminiCurve(data.curve);
      setGeminiProblems(data.problems ?? []);
      setGeminiIssue(data.issue ?? null);
    } catch {
      setGeminiError('Gemini prediction failed — check your API key and backend.');
    } finally {
      setGeminiLoading(false);
    }
  }

  async function handleLocal() {
    if (!videoId || !videoReady || localLoading) return;
    setLocalLoading(true);
    setLocalError(null);
    setLocalCurve(null);
    try {
      const data = await predictCurve(videoId, 'local');
      setLocalCurve(data.curve);
    } catch {
      setLocalError('Prediction failed — is the backend running?');
    } finally {
      setLocalLoading(false);
    }
  }

  // Heatmap shows whichever curve is available (Gemini preferred)
  const heatmapCurve = geminiCurve ?? localCurve;

  return (
    <div className="app mode-b-page">
      <header className="header">
        <div className="header-logo">
          <button className="back-btn" onClick={() => navigate('/')}>←</button>
          <span className="header-icon">✦</span>
          <span className="header-title">AI Prediction</span>
          <span className="mode-tag mode-b">MODE B</span>
        </div>
        <div className="header-right">
          {uploading && <div className="upload-status"><span className="spinner" /> Extracting features…</div>}
          {videoReady && !uploading && <div className="upload-status ready">Features ready</div>}
        </div>
      </header>

      <div className="mode-b-layout">
        {/* Left: video + heatmap + buttons */}
        <div className="mode-b-left">
          <div className="panel-label">VIDEO</div>
          <div
            className="video-drop-zone"
            onDrop={(e) => { e.preventDefault(); loadVideoFile(e.dataTransfer.files[0]); }}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => document.getElementById('video-input-b').click()}
          >
            <video
              ref={videoPlayerRef}
              className="video-player"
              controls
              playsInline
              style={{ display: videoName ? 'block' : 'none' }}
              onClick={(e) => e.stopPropagation()}
            />
            {!videoName && (
              <div className="drop-prompt">
                <div className="drop-icon">▶</div>
                <div>Drop a video file here</div>
                <div className="drop-sub">or click to browse</div>
              </div>
            )}
            <input id="video-input-b" type="file" accept="video/*" style={{ display: 'none' }} onChange={(e) => loadVideoFile(e.target.files[0])} />
          </div>

          <VideoHeatmap curve={heatmapCurve} videoRef={videoPlayerRef} />

          {videoName && (
            <div className="file-badge">
              <span className="file-icon">🎬</span>
              {videoName}
              {uploading && <span className="badge-status"> · uploading…</span>}
              {videoReady && !uploading && <span className="badge-status ready"> · ready</span>}
            </div>
          )}

          {videoReady && (
            <div className="predict-engines">
              <button className="btn btn-engine btn-engine-gemini" onClick={handleGemini} disabled={geminiLoading || localLoading}>
                {geminiLoading ? <><span className="spinner" /> Analysing…</> : geminiCurve ? '✦ Re-run Gemini' : '✦ Gemini AI'}
              </button>
              <button className="btn btn-engine btn-engine-local" onClick={handleLocal} disabled={localLoading || geminiLoading}>
                {localLoading ? <><span className="spinner" /> Running…</> : localCurve ? '⚙ Re-run Model' : '⚙ Our Model'}
              </button>
            </div>
          )}
        </div>

        {/* Right: split — Gemini top, Our Model bottom */}
        <div className="mode-b-right">

          {/* Top half — Gemini */}
          <div className="mode-b-half">
            <div className="panel-label mode-b-half-label">
              <span>✦ GEMINI AI</span>
              {geminiCurve && (
                <button className="clear-prediction" onClick={() => { setGeminiCurve(null); setGeminiProblems([]); setGeminiIssue(null); }}>✕</button>
              )}
            </div>
            {geminiError && <div className="save-error">{geminiError}</div>}
            {geminiCurve ? (
              <div className="mode-b-half-content">
                <div className="mode-b-chart-area">
                  <PredictionChart curve={geminiCurve} problems={geminiProblems} issue={geminiIssue} />
                </div>
              </div>
            ) : (
              <div className="mode-b-empty">
                {geminiLoading ? 'Gemini is analysing your video…' : 'Click "✦ Gemini AI" to analyse'}
              </div>
            )}
          </div>

          <div className="mode-b-divider" />

          {/* Bottom half — Our Model */}
          <div className="mode-b-half">
            <div className="panel-label mode-b-half-label">
              <span>⚙ OUR MODEL</span>
              {localCurve && (
                <button className="clear-prediction" onClick={() => setLocalCurve(null)}>✕</button>
              )}
            </div>
            {localError && <div className="save-error">{localError}</div>}
            {localCurve ? (
              <div className="mode-b-half-content">
                <div className="mode-b-chart-area">
                  <PredictionChart curve={localCurve} />
                </div>
              </div>
            ) : (
              <div className="mode-b-empty">
                {localLoading ? 'Running model…' : 'Click "⚙ Our Model" to analyse'}
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}
