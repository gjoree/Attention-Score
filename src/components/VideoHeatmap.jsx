import { useEffect, useRef, useState } from 'react';

// Interpolate between two hex colors
function lerp(a, b, t) {
  const ah = parseInt(a.slice(1), 16);
  const bh = parseInt(b.slice(1), 16);
  const ar = (ah >> 16) & 0xff, ag = (ah >> 8) & 0xff, ab = ah & 0xff;
  const br = (bh >> 16) & 0xff, bg = (bh >> 8) & 0xff, bb = bh & 0xff;
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const b2 = Math.round(ab + (bb - ab) * t);
  return `rgb(${r},${g},${b2})`;
}

function scoreToColor(score) {
  if (score >= 67) return lerp('#f5a623', '#30D158', (score - 67) / 33);
  if (score >= 40) return lerp('#ff453a', '#f5a623', (score - 40) / 27);
  return lerp('#110000', '#ff453a', score / 40);
}

function formatTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

export function VideoHeatmap({ curve, videoRef }) {
  const canvasRef   = useRef(null);
  const playheadRef = useRef(null);
  const [tooltip, setTooltip] = useState(null); // { x, t, score }

  const duration = curve?.at(-1)?.t ?? 0;

  // ── Draw heatmap ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!curve?.length || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const W = canvas.clientWidth || 600;
    canvas.width = W;

    for (let px = 0; px < W; px++) {
      const t = (px / W) * duration;
      const lo = Math.floor(t);
      const hi = Math.min(lo + 1, curve.length - 1);
      const frac = t - lo;
      const scoreA = curve[lo]?.score ?? 0;
      const scoreB = curve[hi]?.score ?? scoreA;
      const score = scoreA + (scoreB - scoreA) * frac;

      ctx.fillStyle = scoreToColor(score);
      ctx.fillRect(px, 0, 1, canvas.height);
    }

    // Subtle top/bottom fade to blend with bg
    const fadeH = 4;
    const fadeTop = ctx.createLinearGradient(0, 0, 0, fadeH);
    fadeTop.addColorStop(0, '#0d0d0d');
    fadeTop.addColorStop(1, 'transparent');
    ctx.fillStyle = fadeTop;
    ctx.fillRect(0, 0, W, fadeH);

    const fadeBot = ctx.createLinearGradient(0, canvas.height - fadeH, 0, canvas.height);
    fadeBot.addColorStop(0, 'transparent');
    fadeBot.addColorStop(1, '#0d0d0d');
    ctx.fillStyle = fadeBot;
    ctx.fillRect(0, canvas.height - fadeH, W, fadeH);
  }, [curve, duration]);

  // ── Sync playhead to video ────────────────────────────────────────────────
  useEffect(() => {
    const video = videoRef?.current;
    const playhead = playheadRef.current;
    if (!video || !playhead || !duration) return;

    function update() {
      const pct = Math.min(video.currentTime / duration, 1);
      playhead.style.left = `${pct * 100}%`;
    }

    video.addEventListener('timeupdate', update);
    return () => video.removeEventListener('timeupdate', update);
  }, [videoRef, duration]);

  // ── Seek on click ─────────────────────────────────────────────────────────
  function handleClick(e) {
    const video = videoRef?.current;
    if (!video || !duration) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    video.currentTime = Math.max(0, Math.min(pct * duration, duration));
  }

  // ── Tooltip on hover ──────────────────────────────────────────────────────
  function handleMouseMove(e) {
    if (!curve?.length) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    const t = Math.floor(pct * duration);
    const entry = curve.find((c) => c.t === t);
    if (entry) {
      setTooltip({ x: e.clientX - rect.left, t, score: entry.score });
    }
  }

  if (!curve?.length) return null;

  return (
    <div className="heatmap-root">
      <div
        className="heatmap-track"
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setTooltip(null)}
      >
        <canvas ref={canvasRef} className="heatmap-canvas" height={28} />

        {/* Playhead */}
        <div ref={playheadRef} className="heatmap-playhead" />

        {/* Hover tooltip */}
        {tooltip && (
          <div
            className="heatmap-tooltip"
            style={{ left: Math.min(tooltip.x, canvasRef.current?.clientWidth - 80) }}
          >
            <span className="ht-time">{formatTime(tooltip.t)}</span>
            <span
              className="ht-score"
              style={{ color: scoreToColor(tooltip.score) }}
            >
              {tooltip.score}%
            </span>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="heatmap-legend">
        <span className="hl-low">Low attention</span>
        <span className="hl-bar" />
        <span className="hl-high">High attention</span>
        <span className="hl-hint">Click to seek</span>
      </div>
    </div>
  );
}
