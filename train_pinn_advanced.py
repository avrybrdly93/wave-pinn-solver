"""
train_pinn_advanced.py

Advanced training loop for a Physics-Informed Neural Network (PINN)
solving the 2D wave equation:

    u_tt = c^2 (u_xx + u_yy)

Key features:
- Advanced network with Fourier features + residual blocks
- Two-stage optimization: Adam -> L-BFGS
- Adaptive loss balancing between PDE / IC / BC
"""

import os
import time
from typing import Dict

import torch
import torch.nn as nn

from models.pinn_wave2d_advanced import PINNWave2DAdvanced


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C_WAVE = 1.0
T_FINAL = 3.0
DOMAIN_SIZE = 10.0  # meters

# Sampling sizes
# N_INT = 12000   # interior collocation points
# N_IC  = 4000    # initial-condition points
# N_BC  = 4000    # boundary-condition points

# # Optimization
# ADAM_EPOCHS  = 4000
# LBFGS_STEPS  = 120   # number of L-BFGS "epochs" (step calls)
# LR_ADAM      = 1e-3

# PRINT_EVERY  = 200
MODEL_SAVE_PATH = "pinn_wave2d_advanced.pt"

# Adaptive loss weighting
EMA_BETA = 0.99
EPS_ADAPT = 1e-8




# Lighter settings for a weaker CPU

N_INT = 2000     # interior points
N_IC  = 800      # initial condition points
N_BC  = 800      # boundary points

ADAM_EPOCHS = 300
LBFGS_STEPS = 20
LR_ADAM = 1e-3
PRINT_EVERY = 10



# -------------------------------------------------------------------
# Sampling functions (same as before)
# -------------------------------------------------------------------

def sample_interior(n_points, T=T_FINAL):
  t = T * torch.rand(n_points, 1)
  x = DOMAIN_SIZE * torch.rand(n_points, 1)
  y = DOMAIN_SIZE * torch.rand(n_points, 1)
  return t, x, y


def sample_initial(n_points):
  x = DOMAIN_SIZE * torch.rand(n_points, 1)
  y = DOMAIN_SIZE * torch.rand(n_points, 1)
  t = torch.zeros_like(x)

  # xc, yc = DOMAIN_SIZE / 2, DOMAIN_SIZE / 2
  # sigma = 0.1 * DOMAIN_SIZE
  # A = 1.0
  # u0 = A * torch.exp(-((x - xc) ** 2 + (y - yc) ** 2) / sigma ** 2)
  x = torch.rand(n_points, 1)
  y = torch.rand(n_points, 1)
  t = torch.zeros_like(x)

  xc, yc = 0.5, 0.5
  sigma = 0.15
  A = 0.5
  k = 20.0

  r2 = (x - xc) ** 2 + (y - yc) ** 2
  r = torch.sqrt(r2 + 1e-12)  # small epsilon to avoid sqrt(0) issues

  u0 = A * torch.cos(k * r) * torch.exp(-r2 / sigma**2)


  return t, x, y, u0


def sample_boundary(n_points, T=T_FINAL):
  t = T * torch.rand(n_points, 1)

  side = torch.randint(0, 4, (n_points, 1))

  x = DOMAIN_SIZE * torch.rand(n_points, 1)
  y = DOMAIN_SIZE * torch.rand(n_points, 1)

  x = torch.where(side == 0, torch.zeros_like(x), x)
  x = torch.where(side == 1, torch.full_like(x, DOMAIN_SIZE), x)
  y = torch.where(side == 2, torch.zeros_like(y), y)
  y = torch.where(side == 3, torch.full_like(y, DOMAIN_SIZE), y)

  return t, x, y


# -------------------------------------------------------------------
# PDE residual: u_tt - c^2 (u_xx + u_yy)
# -------------------------------------------------------------------

def wave_residual(model, t, x, y, c):
  t.requires_grad_(True)
  x.requires_grad_(True)
  y.requires_grad_(True)

  u = model(t, x, y)  # (N, 1)

  u_t = torch.autograd.grad(
    u, t,
    grad_outputs=torch.ones_like(u),
    create_graph=True
  )[0]

  u_x = torch.autograd.grad(
    u, x,
    grad_outputs=torch.ones_like(u),
    create_graph=True
  )[0]

  u_y = torch.autograd.grad(
    u, y,
    grad_outputs=torch.ones_like(u),
    create_graph=True
  )[0]

  u_tt = torch.autograd.grad(
    u_t, t,
    grad_outputs=torch.ones_like(u_t),
    create_graph=True
  )[0]

  u_xx = torch.autograd.grad(
    u_x, x,
    grad_outputs=torch.ones_like(u_x),
    create_graph=True
  )[0]

  u_yy = torch.autograd.grad(
    u_y, y,
    grad_outputs=torch.ones_like(u_y),
    create_graph=True
  )[0]

  residual = u_tt - (c**2) * (u_xx + u_yy)
  return residual


# -------------------------------------------------------------------
# Adaptive loss weighting utils
# -------------------------------------------------------------------

def init_ema_dict() -> Dict[str, float]:
  # Start each term with 1.0 so we don't divide by tiny values at the beginning
  return {
    "pde": 1.0,
    "ic_disp": 1.0,
    "ic_vel": 1.0,
    "bc": 1.0,
  }


def update_ema(ema: Dict[str, float], key: str, value: float, beta: float = EMA_BETA):
  ema[key] = beta * ema[key] + (1.0 - beta) * value


def balanced_loss(
  loss_pde: torch.Tensor,
  loss_ic_disp: torch.Tensor,
  loss_ic_vel: torch.Tensor,
  loss_bc: torch.Tensor,
  ema: Dict[str, float],
  eps: float = EPS_ADAPT,
) -> torch.Tensor:
  """
  Simple adaptive balancing:
  Each term is normalized by its EMA magnitude, so that
  no single term dominates just because it's numerically larger.
  """
  lp = loss_pde   / (ema["pde"]     + eps)
  lid = loss_ic_disp / (ema["ic_disp"] + eps)
  liv = loss_ic_vel  / (ema["ic_vel"]  + eps)
  lbc = loss_bc   / (ema["bc"]      + eps)

  # Average of normalized terms
  return 0.25 * (lp + lid + liv + lbc)


# -------------------------------------------------------------------
# Single batch of sampled PINN points (used afresh each pass)
# -------------------------------------------------------------------

def sample_all_points(n_int, n_ic, n_bc):
  t_int, x_int, y_int       = sample_interior(n_int)
  t_ic, x_ic, y_ic, u0      = sample_initial(n_ic)
  t_bc, x_bc, y_bc          = sample_boundary(n_bc)

  t_int, x_int, y_int = t_int.to(device), x_int.to(device), y_int.to(device)
  t_ic, x_ic, y_ic, u0 = (
    t_ic.to(device),
    x_ic.to(device),
    y_ic.to(device),
    u0.to(device),
  )
  t_bc, x_bc, y_bc = t_bc.to(device), x_bc.to(device), y_bc.to(device)

  return (t_int, x_int, y_int,
          t_ic, x_ic, y_ic, u0,
          t_bc, x_bc, y_bc)


# -------------------------------------------------------------------
# Loss computation given a set of points
# -------------------------------------------------------------------

def compute_losses(model, c, points):
  (
    t_int, x_int, y_int,
    t_ic, x_ic, y_ic, u0,
    t_bc, x_bc, y_bc,
  ) = points

  # PDE loss (interior)
  r_int = wave_residual(model, t_int, x_int, y_int, c)
  loss_pde = torch.mean(r_int**2)

  # Initial displacement loss: u(0,x,y) ~ u0(x,y)
  u_ic_pred = model(t_ic, x_ic, y_ic)
  loss_ic_disp = torch.mean((u_ic_pred - u0) ** 2)

  # Initial velocity loss: u_t(0,x,y) ~ 0
  t_ic.requires_grad_(True)
  x_ic.requires_grad_(True)
  y_ic.requires_grad_(True)

  u_ic = model(t_ic, x_ic, y_ic)
  u_t_ic = torch.autograd.grad(
    u_ic, t_ic,
    grad_outputs=torch.ones_like(u_ic),
    create_graph=True
  )[0]
  loss_ic_vel = torch.mean(u_t_ic**2)

  # Boundary loss: u(t,x,y) ~ 0 on boundary
  u_bc_pred = model(t_bc, x_bc, y_bc)
  loss_bc = torch.mean(u_bc_pred**2)

  return loss_pde, loss_ic_disp, loss_ic_vel, loss_bc


# -------------------------------------------------------------------
# Training with Adam, then L-BFGS
# -------------------------------------------------------------------

def train_pinn_advanced(
  adam_epochs: int = ADAM_EPOCHS,
  lbfgs_steps: int = LBFGS_STEPS,
  n_int: int = N_INT,
  n_ic: int = N_IC,
  n_bc: int = N_BC,
  lr_adam: float = LR_ADAM,
  c_wave: float = C_WAVE,
  model_save_path: str = MODEL_SAVE_PATH,
):
  model = PINNWave2DAdvanced(
    num_frequencies=6,
    hidden_width=128,
    num_res_blocks=5,
  ).to(device)

  c = torch.tensor(c_wave, dtype=torch.float32, device=device)

  print("Using device:", device)
  print("Training advanced PINN for 2D wave equation")
  print(f"Interior points: {n_int}, IC points: {n_ic}, BC points: {n_bc}")
  print(f"Adam epochs: {adam_epochs}, LR: {lr_adam}, L-BFGS steps: {lbfgs_steps}")

  # Adaptive loss EMA state
  ema = init_ema_dict()

  # -------------------------------
  # Stage 1: Adam optimization
  # -------------------------------
  optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr_adam)

  start_time = time.time()

  for epoch in range(1, adam_epochs + 1):
    # Resample points every epoch
    points = sample_all_points(n_int, n_ic, n_bc)

    optimizer_adam.zero_grad()

    loss_pde, loss_ic_disp, loss_ic_vel, loss_bc = compute_losses(model, c, points)

    # Update EMA magnitudes (detach to avoid autograd history)
    update_ema(ema, "pde",     loss_pde.detach().item())
    update_ema(ema, "ic_disp", loss_ic_disp.detach().item())
    update_ema(ema, "ic_vel",  loss_ic_vel.detach().item())
    update_ema(ema, "bc",      loss_bc.detach().item())

    loss = balanced_loss(loss_pde, loss_ic_disp, loss_ic_vel, loss_bc, ema)

    loss.backward()
    optimizer_adam.step()

    if epoch % PRINT_EVERY == 0 or epoch == 1:
      elapsed = time.time() - start_time
      print(
        f"[Adam] Epoch {epoch:5d}/{adam_epochs} "
        f"| Total(bal): {loss.item():.4e} "
        f"| PDE: {loss_pde.item():.4e} "
        f"| IC_disp: {loss_ic_disp.item():.4e} "
        f"| IC_vel: {loss_ic_vel.item():.4e} "
        f"| BC: {loss_bc.item():.4e} "
        f"| Elapsed: {elapsed:.1f}s"
      )

  # -------------------------------
  # Stage 2: L-BFGS optimization
  # -------------------------------
  # L-BFGS works best on a fixed batch, but we can still resample occasionally.
  # Here we fix one batch for simplicity.
  points_lbfgs = sample_all_points(n_int, n_ic, n_bc)

  # PyTorch LBFGS requires closure
  optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    max_iter=20,         # per .step() call
    history_size=50,
    line_search_fn="strong_wolfe",
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
  )

  print("\nSwitching to L-BFGS refinement...")

  def lbfgs_closure():
    optimizer_lbfgs.zero_grad()

    loss_pde, loss_ic_disp, loss_ic_vel, loss_bc = compute_losses(
      model, c, points_lbfgs
    )

    # Update EMA for consistency in weighting
    update_ema(ema, "pde",     loss_pde.detach().item())
    update_ema(ema, "ic_disp", loss_ic_disp.detach().item())
    update_ema(ema, "ic_vel",  loss_ic_vel.detach().item())
    update_ema(ema, "bc",      loss_bc.detach().item())

    loss = balanced_loss(loss_pde, loss_ic_disp, loss_ic_vel, loss_bc, ema)
    loss.backward()

    # Attach scalar losses to closure for logging convenience
    lbfgs_closure.loss_val = loss.item()
    lbfgs_closure.loss_pde_val = loss_pde.item()
    lbfgs_closure.loss_ic_disp_val = loss_ic_disp.item()
    lbfgs_closure.loss_ic_vel_val = loss_ic_vel.item()
    lbfgs_closure.loss_bc_val = loss_bc.item()

    return loss

  for step in range(1, lbfgs_steps + 1):
    start_step = time.time()
    loss_val = optimizer_lbfgs.step(lbfgs_closure)
    elapsed_step = time.time() - start_step

    if step % max(1, lbfgs_steps // 10) == 0 or step == 1:
      print(
        f"[L-BFGS] Step {step:4d}/{lbfgs_steps} "
        f"| Total(bal): {lbfgs_closure.loss_val:.4e} "
        f"| PDE: {lbfgs_closure.loss_pde_val:.4e} "
        f"| IC_disp: {lbfgs_closure.loss_ic_disp_val:.4e} "
        f"| IC_vel: {lbfgs_closure.loss_ic_vel_val:.4e} "
        f"| BC: {lbfgs_closure.loss_bc_val:.4e} "
        f"| Step time: {elapsed_step:.2f}s"
      )

  total_elapsed = time.time() - start_time
  print(f"\nTotal training time: {total_elapsed/60:.1f} minutes")

  # Save model
  os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
  torch.save(model.state_dict(), model_save_path)
  print(f"Model saved to {model_save_path}")

  return model


if __name__ == "__main__":
  train_pinn_advanced()
