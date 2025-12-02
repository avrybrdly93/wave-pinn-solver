from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.solver import solver

app = FastAPI(title="Wave Simulation Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DropRequest(BaseModel):
    x: float
    y: float
    A: float = 0.02
    sigma: float = 0.05


@app.get("/")
def root():
    return {"message": "Wave backend running"}

@app.post("/reset")
def reset():
    solver.reset()
    return {"status": "ok"}

@app.post("/step")
def step(steps: int = 1):
    steps = max(1, int(steps))
    for _ in range(steps):
        solver.step()
    return {"status": "ok", "time": solver.time}

@app.post("/drop")
def drop(req: DropRequest):
    solver.drop_pebble(req.x, req.y, A=req.A, sigma=req.sigma)
    return {"status": "ok"}

@app.get("/frame")
def frame():
    return solver.frame()
