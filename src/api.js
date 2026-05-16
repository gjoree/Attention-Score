const BASE = 'http://localhost:8000';

async function request(method, path, body, isFile = false) {
  const opts = { method, headers: {} };
  if (body && !isFile) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  } else if (body && isFile) {
    opts.body = body; // FormData — browser sets content-type with boundary
  }
  const res = await fetch(`${BASE}${path}`, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${method} ${path} → ${res.status}: ${text}`);
  }
  return res.json();
}

// ── Videos ───────────────────────────────────────────────────────────────────

export async function uploadVideo(file) {
  const form = new FormData();
  form.append('file', file);
  return request('POST', '/videos/upload', form, true);
  // returns { video_id, status: 'processing' }
}

export async function pollVideoReady(videoId, onReady, intervalMs = 2000) {
  const timer = setInterval(async () => {
    try {
      const data = await request('GET', `/videos/${videoId}/status`);
      if (data.ready) {
        clearInterval(timer);
        onReady(data);
      }
    } catch {
      // ignore transient errors during polling
    }
  }, intervalMs);
  return () => clearInterval(timer); // returns cleanup fn
}

export async function predictCurve(videoId) {
  return request('GET', `/videos/${videoId}/predict`);
  // returns { video_id, curve: [{t, score}, ...] }
}

// ── Sessions ─────────────────────────────────────────────────────────────────

export async function saveSession({ videoId, ageGroup, gender, attentionCurve }) {
  return request('POST', '/sessions', {
    video_id:        videoId,
    age_group:       ageGroup,
    gender:          gender,
    attention_curve: attentionCurve,
  });
  // returns { session_id }
}

// ── Panel ────────────────────────────────────────────────────────────────────

export async function fetchPanel(videoId, reweight = false) {
  return request('GET', `/videos/${videoId}/panel?reweight=${reweight}`);
}

// ── Model ────────────────────────────────────────────────────────────────────

export async function trainModel() {
  return request('POST', '/model/train', {});
}

export async function fetchHealth() {
  return request('GET', '/health');
}
