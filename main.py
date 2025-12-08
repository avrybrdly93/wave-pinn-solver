from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from solvers.wave2d_fd import solver
import numpy as np

app = FastAPI(title="Wave Simulation Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://avrybrdly93.github.io"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------
# Models
# -------------------------------------
class DropRequest(BaseModel):
    x: float
    y: float
    A: float = 0.02
    sigma: float = 0.05

# -------------------------------------
# Routes
# -------------------------------------
@app.get("/")
def root():
    return {"message": "Wave backend running"}


@app.post("/reset")
def reset():
    solver.reset()
    return {"status": "ok"}


@app.post("/step")
def step(steps: int = 1):
    for _ in range(steps):
        solver.step()
    return {"status": "ok"}


@app.post("/drop")
def drop(body: DropRequest):
    solver.drop_pebble(body.x, body.y, body.A)
    return {"status": "ok"}


@app.get("/frame")
def frame():
    u = solver.u_curr  # 2D numpy array
    nx, ny = u.shape

    # Use the solver's own spatial grids
    x = solver.x       # length nx
    y = solver.y       # length ny

    return {
        "u": u.tolist(),
        "x": x.tolist(),
        "y": y.tolist(),
        "t": solver.time,
    }



@app.post("/step_and_frame")
def step_and_frame(steps: int = 1):
    for _ in range(steps):
        solver.step()

    u = solver.u_curr
    nx, ny = u.shape
    x = solver.x
    y = solver.y

    return {
        "u": u.tolist(),
        "x": x.tolist(),
        "y": y.tolist(),
        "t": solver.time,
    }

