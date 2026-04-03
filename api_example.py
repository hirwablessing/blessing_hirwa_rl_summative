"""
FastAPI REST API Example - African Literacy AI Tutor

Demonstrates how the RL environment can be serialized to JSON
and served as an API endpoint for frontend/mobile integration.

Usage:
    pip install fastapi uvicorn
    uvicorn api_example:app --reload

Endpoints:
    POST /reset        - Start a new tutoring session
    POST /step         - Take a tutoring action
    GET  /actions      - List available actions
    GET  /state        - Get current environment state as JSON
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from environment.custom_env import AfricanLiteracyTutorEnv

app = FastAPI(
    title="African Literacy AI Tutor API",
    description="REST API for the RL-based literacy tutoring environment. "
                "Enables integration with web/mobile frontends.",
    version="1.0.0",
)

# Global environment instance
env = AfricanLiteracyTutorEnv()
session_active = False


class ActionRequest(BaseModel):
    action: str  # Action name, e.g., "PHONEME_DRILL"


class StepResponse(BaseModel):
    state: dict
    reward: float
    terminated: bool
    truncated: bool
    info: dict


@app.post("/reset", response_model=dict)
def reset_session():
    """Start a new tutoring session. Returns initial environment state as JSON."""
    global session_active
    env.reset()
    session_active = True
    return env.to_json()


@app.post("/step", response_model=StepResponse)
def take_action(request: ActionRequest):
    """Take a tutoring action and return the updated state."""
    global session_active
    if not session_active:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")

    try:
        action = env.action_from_json({"action": request.action})
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action '{request.action}'. Use GET /actions for valid actions."
        ) from exc

    _obs, reward, terminated, truncated, info = env.step(action)
    state = env.to_json()

    if terminated or truncated:
        session_active = False

    # Convert numpy types in info
    clean_info = {k: float(v) if hasattr(v, 'item') else v for k, v in info.items()}

    return StepResponse(
        state=state,
        reward=float(reward),
        terminated=terminated,
        truncated=truncated,
        info=clean_info,
    )


@app.get("/actions")
def list_actions():
    """List all available tutoring actions with descriptions."""
    return env.get_action_descriptions()


@app.get("/state")
def get_state():
    """Get the current environment state as JSON."""
    if not session_active:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    return env.to_json()


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "environment": "AfricanLiteracyTutor-v0"}
