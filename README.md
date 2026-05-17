# Attention Tracker

A real-time viewer attention analysis platform.
Watch a video while the webcam tracks your eyes and plots a live attention curve.
Upload any video and get an AI-predicted attention curve showing exactly where viewers will zone out, before a single person watches it.

---

## Problem Definiton

Attention measurement is valuable but usually locked behind expensive panel-based systems.
That makes Nielsen-style insight hard to access for creators, streamers, and small advertisers.

We solve this by estimating attention directly from the video itself, without the need of a panel.
The result is fast, affordable pre-campaign audience insight from any laptop, with the goal of making attention measurement free.

---

## General Overview

The system uses a combination of real-time webcam tracking and AI prediction to analyze viewer attention in videos.

It has two modes:

- **Live Tracking** - Realtime webcam tracking of viewer attention using `MediaPipe`.
- **AI Prediction** - AI-based prediction of viewer attention using `Google Gemini`.

### Live Tracking

The user uploads a video and turns on the live tracking feature.
While the user is watching the video, the webcam tracks their eyes and plots a live attention curve.
The webcam uses `MediaPipe` to track face landmarks, eye blink blendshapes, and head pose in real time.
An attention score (0–100%) is computed per frame and plotted as a live timeline.
Sessions are saved to a local `SQLite` database with demographic info (age, group, gender).
For each session, 22 video features are extracted per second (motion, faces, audio energy, scene cuts, brightness spikes, colour variety, etc.), and in the database.

### AI Prediction

The user uploads a video.
The system reads the video.
The user has the option to choose one of our pre-trained models or use Google Gemini to predict the viewer attention in respect of time for the uploaded video.
Our custom model uses XGBREgressor to predict viewer attention in respect of time.
It is pre-trained on the feature data taken from the Live Tracking.

In the case of Google Gemini, the full video is sent and then the model directly predicts viewer attention in respect of time.

## Tech Stack

### Frontend

- **React 18 + Vite 5**: UI
- **Chart.js**: renders live and predicted attention curves.
- **MediaPipe Tasks Vision**: supports on-device attention signal tracking in the browser.

### Backend

- **FastAPI + Uvicorn**: REST API for video upload, feature extraction, prediction, session saving, and analytics endpoints.
- **SQLite**: lightweight local persistence for session data and extracted video features.
- **Python ML/Video stack**:
  - **OpenCV** for frame-level visual analysis.
  - **librosa** for audio feature extraction.
  - **PySceneDetect** for scene-cut detection.
  - **XGBoost + NumPy/SciPy/Pandas + joblib** for training and inference pipeline.
- **Gemini API (optional)**: Big Model AI user attention prediction.

### Database

- **SQLite**: lightweight local persistence for session data and extracted video features.

## How to Run

### Run Online (Cloud Deployment)

- **Frontend URL**: `https://<frontend-domain>`
- **Backend API URL**: `https://<backend-domain>`

### Run Locally

#### 1. Prerequisites

- **Node.js** 18+ and **npm**
- **Python** 3.10+
- **ffmpeg** (recommended for robust video/audio handling)
- **Gemini API key** (optional, if you want to use the Big Model AI attention prediction)

#### 2. Clone the repository

```bash
# clone the repository
git clone https://github.com/gjoree/Attention-Score.git

# enter the project
cd Attention-Score
```

#### 3. Start the backend

```bash
# enter the backend directory
cd backend

# create a python virtual environment
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run the backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend will be available at `http://localhost:8000`.

> IMPORTANT: For your Gemini integration to work, you have to set up an environment variable with your Gemini API key.

```bash
export GEMINI_API_KEY="your_api_key_here"
```

#### 4. Start frontend

Open a new terminal:

```bash
# enter the frontend directory
cd Attention-Score

# install dependencies
npm install

# start the development server
npm run dev
```

Frontend will be available at `http://localhost:5173`.

## Contribution Guide

### Rules for Contribution

1. **Fork and branch** from `develop` using a descriptive branch name (example: `feature/panel-weighting-fix`).
2. **Keep pull requests focused** on one logical change (feature, fix, or refactor).
3. **Follow existing code style and structure** in both frontend and backend.
4. **Test your changes locally** before opening a pull request.
5. **Write clear commit messages** that explain what changed and why.
6. **Document behavior changes** in this README when setup, API behavior, or architecture is affected.
7. **Open a pull request** with:
   - a concise summary,
   - testing notes,
   - screenshots/video for UI changes (if applicable).

### Reporting Issues

- Use **GitHub Issues** for bug reports and feature requests.
- Include reproduction steps, expected behavior, and actual behavior.

---

## Troubleshooting

**Camera permission denied**
Allow camera access in your browser when prompted.
In Chrome: click the camera icon in the address bar.

**"Features still processing"**
Feature extraction takes 10–20 seconds per minute of video.
Wait for the **"Features ready"** badge in the header.

**Gemini 429 quota error**
The free tier has limited requests.
Either wait a minute and retry, or enable billing at [aistudio.google.com](https://aistudio.google.com) (the whole hackathon demo costs under $1).

**`database is locked` error on startup**
Close any SQLite viewer open in VS Code (bottom status bar).
Then delete leftover WAL files:

```bash
Remove-Item backend\sessions.db-wal -ErrorAction SilentlyContinue
Remove-Item backend\sessions.db-shm -ErrorAction SilentlyContinue
```
