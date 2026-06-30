"""
FastAPI server for the Pacman RL Visualization Platform.
Exposes endpoints for training, testing, and model management.
Serves the frontend static files.
"""
import os
import sys
import shutil
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure the backend package can find the engine module
sys.path.insert(0, os.path.dirname(__file__))
from engine import GameEngine, TrainingRunner, get_available_layouts, get_layout_preview

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pacman RL Visualization API",
    description="Backend API for the Interactive RL Visualization Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singletons
game_engine = GameEngine()
training_runner = TrainingRunner()

PACMAN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PACMAN_DIR, "saved_models")
FRONTEND_DIR = os.path.join(PACMAN_DIR, "frontend")
os.makedirs(MODELS_DIR, exist_ok=True)

# Mount static directories (CSS, JS) at /static
app.mount("/static/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
app.mount("/static/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    episodes: int = 100
    alpha: float = 0.5
    gamma: float = 0.8
    epsilon: float = 0.3
    layout: str = "mediumClassic"
    numGhosts: int = 4
    ghostType: str = "random"

class PlayStartRequest(BaseModel):
    layout: str = "mediumClassic"
    ghostType: str = "random"
    numGhosts: int = 4
    modelPath: Optional[str] = None


# ---------------------------------------------------------------------------
# Layout Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/layouts")
def list_layouts():
    """Return list of available layout names."""
    layouts = get_available_layouts()
    return {"layouts": layouts}

@app.get("/api/layouts/{layout_name}")
def get_layout(layout_name: str):
    """Return the raw text preview of a layout."""
    preview = get_layout_preview(layout_name)
    if preview is None:
        raise HTTPException(status_code=404, detail=f"Layout '{layout_name}' not found")
    return {"name": layout_name, "text": preview}


# ---------------------------------------------------------------------------
# Training Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/train")
def start_training(req: TrainRequest):
    """Start training the RL agent in the background."""
    result = training_runner.start(
        episodes=req.episodes,
        alpha=req.alpha,
        gamma=req.gamma,
        epsilon=req.epsilon,
        layout_name=req.layout,
        num_ghosts=req.numGhosts,
        ghost_type=req.ghostType,
    )
    return result

@app.get("/api/train/status")
def training_status():
    """Get current training progress and metrics."""
    metrics = training_runner.get_metrics()
    metrics["is_training"] = training_runner.is_training
    return metrics

@app.post("/api/train/stop")
def stop_training():
    """Stop training (sets flag — thread finishes current episode)."""
    training_runner.is_training = False
    return {"status": "stop_requested"}


# ---------------------------------------------------------------------------
# Play / Test Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/play/start")
def play_start(req: PlayStartRequest):
    """Start a new test game for step-by-step playback."""
    try:
        model_path = None
        if req.modelPath:
            if not os.path.isabs(req.modelPath) and os.path.dirname(req.modelPath) == "":
                model_path = os.path.join(MODELS_DIR, req.modelPath)
            else:
                model_path = req.modelPath
        snapshot = game_engine.start_game(
            layout_name=req.layout,
            ghost_type=req.ghostType,
            num_ghosts=req.numGhosts,
            model_path=model_path,
        )
        return snapshot
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/play/step")
def play_step():
    """Advance the game by one step and return the new state."""
    snapshot = game_engine.step()
    return snapshot

@app.get("/api/play/state")
def play_state():
    """Return the current game state without advancing."""
    return game_engine.get_snapshot()

@app.post("/api/play/reset")
def play_reset():
    """Reset the current game to the beginning."""
    return game_engine.reset()


# ---------------------------------------------------------------------------
# Model Management
# ---------------------------------------------------------------------------

@app.get("/api/models")
def list_models():
    """List all saved model files."""
    if not os.path.isdir(MODELS_DIR):
        return {"models": []}

    models = []
    for f in sorted(os.listdir(MODELS_DIR)):
        if f.endswith(".txt") or f.endswith(".json"):
            path = os.path.join(MODELS_DIR, f)
            models.append({
                "name": f,
                "size": os.path.getsize(path),
                "modified": os.path.getmtime(path),
            })
    return {"models": models}

@app.post("/api/model/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a Q-table model file."""
    dest = os.path.join(MODELS_DIR, file.filename)
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"status": "uploaded", "filename": file.filename}

@app.get("/api/model/download/{name}")
def download_model(name: str):
    """Download a saved model file."""
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return FileResponse(path, filename=name)

@app.post("/api/model/save-current")
def save_current_model():
    """Copy the current weights.json into saved_models/."""
    src = os.path.join(PACMAN_DIR, "weights.json")
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail="No weights.json found")
    import time as _time
    ts = int(_time.time())
    dest_name = f"weights_{ts}.json"
    shutil.copy2(src, os.path.join(MODELS_DIR, dest_name))
    return {"status": "saved", "filename": dest_name}


# ---------------------------------------------------------------------------
# Frontend — serve index.html for all non-API routes (SPA catch-all)
# ---------------------------------------------------------------------------

@app.get("/{full_path:path}")
def serve_frontend(full_path: str):
    """Serve the frontend SPA."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(index_path, media_type="text/html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

