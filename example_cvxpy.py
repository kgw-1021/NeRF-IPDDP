import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Trajectory horizon
T = 20

# Grid settings
x_min, x_max = 0, 50
y_min, y_max = 0, 50
n_grid = 100
grid_x = np.linspace(x_min, x_max, n_grid)
grid_y = np.linspace(y_min, y_max, n_grid)
dx = grid_x[1] - grid_x[0]
dy = grid_y[1] - grid_y[0]

# Sample NeRF-like density points
density_points = np.array([
    [25, 25, 10],
    [35, 15, 5],
    [10, 40, 8]
])

def generate_density_cost_map(grid_x, grid_y, density_points, sigma=2.0):
    xx, yy = np.meshgrid(grid_x, grid_y)
    cost_map = np.zeros_like(xx)
    for x, y, d in density_points:
        dist_sq = (xx - x)**2 + (yy - y)**2
        cost_map += d * np.exp(-dist_sq / (2 * sigma**2))
    return cost_map

def get_cost_at_points(xs, ys, cost_map):
    x_idx = np.clip(((xs - x_min) / dx).astype(int), 0, n_grid - 1)
    y_idx = np.clip(((ys - y_min) / dy).astype(int), 0, n_grid - 1)
    return cost_map[y_idx, x_idx]

# QP variable
traj = cp.Variable(2 * T)

# Smoothness term
Q_smooth = np.zeros((2 * T, 2 * T))
for t in range(1, T - 1):
    for i in range(2):
        idx = 2 * t + i
        Q_smooth[idx - 2, idx - 2] += 1
        Q_smooth[idx, idx] += 4
        Q_smooth[idx + 2, idx + 2] += 1
        Q_smooth[idx - 2, idx] -= 2
        Q_smooth[idx + 2, idx] -= 2
        Q_smooth[idx, idx - 2] -= 2
        Q_smooth[idx, idx + 2] -= 2
Q_smooth = Q_smooth / 4

# Goal term
goal = np.array([45.0, 45.0])
Q_goal = np.zeros((2 * T, 2 * T))
Q_goal[-2, -2] = 1
Q_goal[-1, -1] = 1
c_goal = np.zeros(2 * T)
c_goal[-2:] = -goal * 2

# Density term from cost map
cost_map = generate_density_cost_map(grid_x, grid_y, density_points)
x_init = np.linspace(5, goal[0], T)
y_init = np.linspace(5, goal[1], T)
density_weights = np.zeros(2 * T)
density_weights[::2] = get_cost_at_points(x_init, y_init, cost_map)
density_weights[1::2] = get_cost_at_points(x_init, y_init, cost_map)
Q_density = np.diag(density_weights)

print(f"Q smooth:\n {Q_smooth}")
print(f"Q goal:\n {Q_goal}")
print(f"Q density:\n {Q_density}")

# Total cost
Q_total = (1.0 * Q_smooth + 0.1 * Q_goal + 5.0 * Q_density)
c_total = c_goal

# Solve
objective = cp.Minimize(0.5 * cp.quad_form(traj, Q_total) + c_total @ traj)
constraints = [traj[0] == 5.0, traj[1] == 5.0]
prob = cp.Problem(objective, constraints)
prob.solve()

# Retrieve result
traj_opt = traj.value.reshape(-1, 2)

# Visualization
plt.figure(figsize=(8, 8))
plt.imshow(cost_map, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='hot', alpha=0.5)
plt.plot(traj_opt[:, 0], traj_opt[:, 1], 'cyan', lw=2, label='Optimized Trajectory')
plt.scatter(*goal, c='blue', label='Goal')
plt.scatter(*density_points[:, :2].T, c='red', label='Density Peaks')
plt.scatter(5, 5, c='green', label='Start')
plt.title("QP-optimized Trajectory with Density-based Cost Field")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()