import { useNavigate } from 'react-router-dom';

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="home">
      <div className="home-brand">
        <span className="home-icon">👁</span>
        <h1 className="home-title">Attention Tracker</h1>
        <p className="home-sub">Real-time viewer attention analysis powered by AI</p>
      </div>

      <div className="home-cards">
        <button className="mode-card mode-card-a" onClick={() => navigate('/mode-a')}>
          <div className="mc-badge">MODE A</div>
          <div className="mc-icon">📷</div>
          <div className="mc-title">Live Tracking</div>
          <div className="mc-desc">
            Webcam tracks your eyes and face in real time while you watch a video.
            Plots a live attention curve second by second.
          </div>
          <div className="mc-cta">Start Session →</div>
        </button>

        <button className="mode-card mode-card-b" onClick={() => navigate('/mode-b')}>
          <div className="mc-badge">MODE B</div>
          <div className="mc-icon">✦</div>
          <div className="mc-title">AI Prediction</div>
          <div className="mc-desc">
            Upload any video. Gemini AI analyses the full content and predicts
            exactly where viewers will zone out — before anyone watches.
          </div>
          <div className="mc-cta">Predict Attention →</div>
        </button>
      </div>
    </div>
  );
}
