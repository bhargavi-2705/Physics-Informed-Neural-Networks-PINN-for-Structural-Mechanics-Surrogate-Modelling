import torch
import numpy as np
import pandas as pd

a = 0.5
R = 0.1 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#COLLOCATION POINT GENERATION
 
def sample_interior(N):
    pts = []
    while len(pts) < N:
        x = np.random.uniform(-a, a, size=(N * 3,))
        y = np.random.uniform(-a, a, size=(N * 3,))

        r = np.sqrt(x**2 + y**2)

        mask = r >= R          # condition to keep only points outside the hole

        x, y = x[mask], y[mask]

        for xi, yi in zip(x, y):
            pts.append([xi, yi])
            if len(pts) == N:
                break
    
    arr = np.array(pts[:N], dtype=np.float32)
    t = torch.tensor(arr, requires_grad=True).to(device)

    return t


def sample_left_right(N):
    y = np.random.uniform(-a, a, size=(N,1)).astype(np.float32)
    xl = np.full(shape=(N, 1), fill_value = -a).astype(np.float32)
    xr = np.full(shape=(N, 1), fill_value = a).astype(np.float32)
    left = torch.tensor(np.hstack([xl, y]), requires_grad=True).to(device)
    right = torch.tensor(np.hstack([xr, y]), requires_grad=True).to(device)

    return left, right

def sample_top_bottom(N):
    x  = np.random.uniform(-a, a, (N, 1)).astype(np.float32)
    yt = np.full((N, 1),  a, dtype=np.float32)
    yb = np.full((N, 1), -a, dtype=np.float32)
    top = torch.tensor(np.hstack([x, yt]), requires_grad=True).to(device)
    bot = torch.tensor(np.hstack([x, yb]), requires_grad=True).to(device)

    return top, bot
 
 
def sample_hole(N):
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False).astype(np.float32)
    x = (R * np.cos(theta)).reshape(-1, 1)
    y = (R * np.sin(theta)).reshape(-1, 1)

    nx = (-np.cos(theta)).reshape(-1, 1)
    ny = (-np.sin(theta)).reshape(-1, 1)

    pts = torch.tensor(np.hstack([x, y]), requires_grad=True).to(device)
    normals = torch.tensor(np.hstack([nx, ny]), dtype=torch.float32).to(device)

    return pts, normals
 


N_int   = 3000   # interior collocation points
N_edge  = 200    # points per boundary segment
N_hole  = 200    # points around hole


x_int_grid = sample_interior(N_int)
x_left_grid , x_right_grid = sample_left_right(N_edge)
x_top_grid, x_bot_grid = sample_top_bottom(N_edge)
x_hole_grid, n_hole_grid = sample_hole(N_hole)
 

