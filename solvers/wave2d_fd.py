# solvers/wave2d_fd.py
# Basic 2D wave equation finite difference solver
import numpy as np


def solve_wave_2d(nx=101, ny=101, nt=600, c=1.0, Lx=10, Ly=10, T=3.0):
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dt = T / (nt - 1)

    # CFL condition (simple form)
    assert c * dt / dx < 1 / np.sqrt(2), "CFL condition violated"

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    t = np.linspace(0, T, nt)

    # Allocate arrays
    u_prev = np.zeros((nx, ny))
    u_curr = np.zeros((nx, ny))
    u_next = np.zeros((nx, ny))
    U = np.zeros((nt, nx, ny))

    # Pebble drop: slight bump plus a short downward impulse at center
    X, Y = np.meshgrid(x, y, indexing="ij")
    xc, yc = Lx / 2, Ly / 2
    r2 = (X - xc) ** 2 + (Y - yc) ** 2

    sigma_disp = 0.15 * min(Lx, Ly)
    A_disp = 0.05
    u_curr = A_disp * np.exp(-r2 / sigma_disp**2)

    sigma_vel = 0.10 * min(Lx, Ly)
    A_vel = -1.5  # negative = downward push
    u_t0 = A_vel * np.exp(-r2 / sigma_vel**2)

    # Backward step: u(t - dt) ≈ u(0) - dt * u_t(0)
    u_prev = u_curr - dt * u_t0

    U[0] = u_curr.copy()

    # Time stepping
    cx = (c * dt / dx) ** 2
    cy = (c * dt / dy) ** 2

    for n in range(1, nt):
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u_next[i, j] = (
                    2 * u_curr[i, j]
                    - u_prev[i, j]
                    + cx * (u_curr[i + 1, j] - 2 * u_curr[i, j] + u_curr[i - 1, j])
                    + cy * (u_curr[i, j + 1] - 2 * u_curr[i, j] + u_curr[i, j - 1])
                )

        # Enforce boundary conditions u=0
        u_next[0, :] = 0
        u_next[-1, :] = 0
        u_next[:, 0] = 0
        u_next[:, -1] = 0

        U[n] = u_next.copy()

        # advance time levels
        u_prev, u_curr, u_next = u_curr, u_next, u_prev

    return t, x, y, U


if __name__ == "__main__":
    t, x, y, U = solve_wave_2d()
    print("FD solver complete", U.shape)
