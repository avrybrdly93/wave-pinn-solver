# export_wave_data.py
# Converts FD + PINN results into the JSON format the front end expects


import json
import numpy as np


def save_wave_data_json(t, x, y, u_fd, u_pinn, path="data/wave_data.json"):
    data = {
        "t": t.tolist(),
        "x": x.tolist(),
        "y": y.tolist(),
        "fd": u_fd.tolist(),
        "pinn": u_pinn.tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Saved to {path}")


if __name__ == "__main__":
    # Dummy example; replace with real FD + PINN outputs
    Nt, Nx, Ny = 10, 21, 21
    t = np.linspace(0, 1, Nt)
    x = np.linspace(0, 10, Nx)
    y = np.linspace(0, 10, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")


# Fake dummy data
    u_fd = np.zeros((Nt, Nx, Ny))
    u_pinn = np.zeros((Nt, Nx, Ny))
    for n in range(Nt):
        u_fd[n] = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.cos(np.pi * t[n])
        u_pinn[n] = 0.98 * u_fd[n]


    save_wave_data_json(t, x, y, u_fd, u_pinn)
