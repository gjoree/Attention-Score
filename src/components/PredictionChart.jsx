import { useEffect, useRef } from 'react';
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  Filler,
  Tooltip,
} from 'chart.js';

Chart.register(LineController, LineElement, PointElement, LinearScale, Filler, Tooltip);

const DIP_THRESHOLD = 50;
const MIN_DIP_SECS  = 2;

function formatTime(s) {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

function findDips(curve) {
  const dips = [];
  let start = null;
  for (let i = 0; i < curve.length; i++) {
    const low = curve[i].score < DIP_THRESHOLD;
    if (low && start === null) start = curve[i].t;
    if (!low && start !== null) {
      if (curve[i].t - start >= MIN_DIP_SECS) dips.push({ start, end: curve[i].t });
      start = null;
    }
  }
  if (start !== null && curve.at(-1).t - start >= MIN_DIP_SECS)
    dips.push({ start, end: curve.at(-1).t });
  return dips;
}

export function PredictionChart({ curve, problems, issue }) {
  const canvasRef = useRef(null);
  const chartRef  = useRef(null);

  useEffect(() => {
    if (!curve?.length) return;

    const pointColors = curve.map(({ score }) =>
      score >= 67 ? '#30D158' : score >= 40 ? '#f5a623' : '#ff453a'
    );

    const ctx = canvasRef.current.getContext('2d');
    chartRef.current?.destroy();

    chartRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'Predicted attention',
            data: curve.map(({ t, score }) => ({ x: t, y: score })),
            borderColor: '#30D158',
            backgroundColor: 'rgba(48,209,88,0.07)',
            fill: true,
            tension: 0.45,
            pointRadius: 3,
            pointBackgroundColor: pointColors,
            borderWidth: 2,
          },
          {
            label: 'Threshold',
            data: curve.map(({ t }) => ({ x: t, y: DIP_THRESHOLD })),
            borderColor: '#ff453a44',
            borderDash: [5, 4],
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: 'linear',
            title: { display: false },
            ticks: {
              color: '#bbb',
              font: { size: 12 },
              callback: (v) => formatTime(v),
              maxTicksLimit: 10,
            },
            grid: { color: '#1e1e1e' },
            border: { color: '#333' },
          },
          y: {
            min: 0,
            max: 100,
            ticks: {
              color: '#bbb',
              font: { size: 12 },
              callback: (v) => v + '%',
              stepSize: 25,
            },
            grid: { color: '#1e1e1e' },
            border: { color: '#333' },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1a1a1a',
            borderColor: '#333',
            borderWidth: 1,
            titleColor: '#ccc',
            bodyColor: '#fff',
            callbacks: {
              title: ([ctx]) => formatTime(ctx.parsed.x),
              label: (ctx) =>
                ctx.datasetIndex === 0
                  ? ` ${ctx.parsed.y}% predicted attention`
                  : null,
            },
          },
        },
      },
    });

    return () => chartRef.current?.destroy();
  }, [curve]);

  if (!curve?.length) return null;

  const useProblems = Array.isArray(problems) && problems.length > 0;
  const dips        = useProblems ? [] : findDips(curve);

  return (
    <div className="prediction-chart-root">
      <div className="prediction-canvas-wrapper">
        <canvas ref={canvasRef} />
      </div>

      {issue && (
        <div className="problem-card problem-card--issue">
          <span className="problem-type">⚠ video issue</span>
          <span className="problem-detail">{issue}</span>
        </div>
      )}

      {useProblems ? (
        <div className="problem-list">
          <div className="problem-list-title">Problematic segments</div>
          {problems.map((p, i) => (
            <div key={i} className="problem-card">
              <div className="problem-card-header">
                <span className="problem-type">{p.type}</span>
                <span className="problem-range">{formatTime(p.start)} – {formatTime(p.end)}</span>
              </div>
              <span className="problem-detail">{p.detail}</span>
            </div>
          ))}
        </div>
      ) : Array.isArray(problems) ? (
        <div className="dip-row">
          <span className="dip-label no-dips">No problematic segments detected</span>
        </div>
      ) : dips.length > 0 ? (
        <div className="dip-row">
          <span className="dip-label">Predicted dips</span>
          {dips.map((d, i) => (
            <span key={i} className="dip-chip">
              {formatTime(d.start)}–{formatTime(d.end)}
            </span>
          ))}
        </div>
      ) : (
        <div className="dip-row">
          <span className="dip-label no-dips">No significant dips predicted</span>
        </div>
      )}
    </div>
  );
}
