import { useEffect, useRef } from 'react';
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  TimeScale,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';
import 'chartjs-adapter-date-fns';

Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  TimeScale,
  Filler,
  Tooltip,
  Legend
);

export function AttentionChart({ scores }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    const ctx = canvasRef.current.getContext('2d');

    chartRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'Attention %',
            data: [],
            borderColor: '#30D158',
            backgroundColor: 'rgba(48, 209, 88, 0.08)',
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            borderWidth: 2,
          },
        ],
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: {
            type: 'time',
            time: { unit: 'second', displayFormats: { second: 'HH:mm:ss' } },
            ticks: { color: '#555', maxTicksLimit: 8, font: { size: 11 } },
            grid: { color: '#1e1e1e' },
            border: { color: '#333' },
          },
          y: {
            min: 0,
            max: 100,
            ticks: {
              color: '#555',
              font: { size: 11 },
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
            titleColor: '#888',
            bodyColor: '#fff',
            callbacks: {
              label: (ctx) => ` ${ctx.parsed.y.toFixed(0)}% attention`,
            },
          },
        },
      },
    });

    return () => chartRef.current?.destroy();
  }, []);

  useEffect(() => {
    if (!chartRef.current || scores.length === 0) return;
    chartRef.current.data.datasets[0].data = scores.map((s) => ({
      x: s.time,
      y: s.score,
    }));
    chartRef.current.update('none');
  }, [scores]);

  return <canvas ref={canvasRef} />;
}
