# solvers/wave2d_fd.py
import numpy as np

class Wave2DSolver:
    def __init__(self, nx=101, ny=101, c=1.0, Lx=10, Ly=10, dt=0.01):
        self.nx = nx
        self.ny = ny
        self.c = c
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)

        # CFL check
        assert c * dt / self.dx < 1 / np.sqrt(2), "CFL condition violated"

        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        self.reset()

    # def reset(self):
    #     """Reset wave to flat surface."""
    #     self.u_prev = np.zeros((self.nx, self.ny))
    #     self.u_curr = np.zeros((self.nx, self.ny))
    #     self.u_next = np.zeros((self.nx, self.ny))
    #     self.time = 0.0

    #     self.cx = (self.c * self.dt / self.dx) ** 2
    #     self.cy = (self.c * self.dt / self.dy) ** 2

    def reset(self):
        """Reset wave to flat surface and drop an initial pebble."""
        self.u_prev = np.zeros((self.nx, self.ny))
        self.u_curr = np.zeros((self.nx, self.ny))
        self.u_next = np.zeros((self.nx, self.ny))
        self.time = 0.0

        self.cx = (self.c * self.dt / self.dx) ** 2
        self.cy = (self.c * self.dt / self.dy) ** 2

        # 🔹 Add an initial Gaussian bump in the center
        x0 = self.Lx / 2
        y0 = self.Ly / 2
        self.drop_pebble(x0, y0, A=0.02, sigma=0.5)


    def drop_pebble(self, x0, y0, A=0.02, sigma=0.05):
        """Add a Gaussian displacement bump."""
        r2 = (self.X - x0) ** 2 + (self.Y - y0) ** 2
        self.u_curr += A * np.exp(-r2 / sigma**2)

    def step(self):
        """Advance simulation by one time step using finite differences."""
        u = self.u_curr
        u_prev = self.u_prev
        u_next = self.u_next

        # Vectorized interior update
        u_next[1:-1, 1:-1] = (
            2 * u[1:-1, 1:-1]
            - u_prev[1:-1, 1:-1]
            + self.cx * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1])
            + self.cy * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])
        )

        # Boundary conditions (fixed edges)
        u_next[0, :] = 0
        u_next[-1, :] = 0
        u_next[:, 0] = 0
        u_next[:, -1] = 0

        # Rotate buffers
        self.u_prev, self.u_curr, self.u_next = (
            self.u_curr,
            self.u_next,
            self.u_prev,
        )

        self.time += self.dt

    def frame(self):
        """Return current height field as a Python list (JSON serializable)."""
        return {
            "time": self.time,
            "u": self.u_curr.tolist(),
        }
        
    def laplacian(u, dx, dy):
        return (
            (np.roll(u, +1, axis=0) - 2*u + np.roll(u, -1, axis=0)) / dx**2 +
            (np.roll(u, +1, axis=1) - 2*u + np.roll(u, -1, axis=1)) / dy**2
        )



# Create global solver instance for FastAPI
solver = Wave2DSolver()
