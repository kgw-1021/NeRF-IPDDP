
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

def generate_sample_trajectory(start, v, w_traj, dt=0.5):
    x, y, theta = start
    traj = []
    for w in w_traj:
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt
        traj.append([x, y, theta])
    return np.array(traj)

def generate_hard_bounded_density_field(grid_x, grid_y, obstacles, threshold=0.5, low_value=1e-3):
    density = np.zeros((len(grid_y), len(grid_x)))
    for (cx, cy, r, intensity) in obstacles:
        xx, yy = np.meshgrid(grid_x, grid_y)
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        density += intensity * np.exp(-dist_sq / (2 * (r**2)))
    density = density / np.max(density)
    density[density < threshold] = low_value
    return density

def get_density_interpolator(grid_x, grid_y, density_map):
    return RegularGridInterpolator((grid_y, grid_x), density_map, bounds_error=False, fill_value=0)

def build_cost_field_from_trajectories(trajectories, density_fn, grid_x, grid_y, sigma=2.0):
    cost_field = np.zeros((len(grid_y), len(grid_x)))
    x_res = grid_x[1] - grid_x[0]
    y_res = grid_y[1] - grid_y[0]

    all_points = np.vstack([traj[:, :2] for traj in trajectories])
    densities = density_fn(all_points)

    for i, point in enumerate(all_points):
        xi = int((point[0] - grid_x[0]) / x_res)
        yi = int((point[1] - grid_y[0]) / y_res)
        if 0 <= xi < len(grid_x) and 0 <= yi < len(grid_y):
            cost_field[yi, xi] += densities[i]

    return gaussian_filter(cost_field, sigma=sigma)

def get_cost_interpolator(cost_field, grid_x, grid_y):
    return RegularGridInterpolator((grid_y, grid_x), cost_field, bounds_error=False, fill_value=0)

def compute_cost_from_costfield(traj, cost_fn):
    traj_xy = traj[:, :2]
    return np.sum(cost_fn(traj_xy))

def compute_log_likelihood_cost(traj, goal, cost_fn, sigma_s=0.1, sigma_g=8.0, sigma_d=0.001):
    traj_xy = traj[:, :2]
    diffs = np.diff(traj_xy, axis=0)
    smoothness = np.sum(np.linalg.norm(np.diff(diffs, axis=0), axis=1) ** 2)
    goal_dist = np.linalg.norm(traj_xy[-1] - goal) ** 2
    cost = (smoothness / (2 * sigma_s**2)) + (goal_dist / (2 * sigma_g**2)) + (compute_cost_from_costfield(traj, cost_fn) / (2 * sigma_d**2))
    return cost

def optimize_trajectory_with_generated_costfield(start, goal, density_fn, grid_x, grid_y,
                                                 horizon=40, num_samples=200):
    all_trajs = []
    for _ in range(num_samples):
        v = np.random.uniform(0.0, 1.5)
        w_traj = np.random.uniform(-0.5, 0.5, size=horizon)
        traj = generate_sample_trajectory(start, v, w_traj)
        all_trajs.append(traj)

    cost_field = build_cost_field_from_trajectories(all_trajs, density_fn, grid_x, grid_y, sigma=5.0)
    cost_fn = get_cost_interpolator(cost_field, grid_x, grid_y)

    best_cost = np.inf
    best_traj = None
    for traj in all_trajs:
        cost = compute_log_likelihood_cost(traj, goal, cost_fn)
        if cost < best_cost:
            best_cost = cost
            best_traj = traj
    return best_traj, cost_field

def run_navigation_loop():
    grid_x = np.linspace(0, 50, 200)
    grid_y = np.linspace(0, 50, 200)
    obstacles = [(25, 25, 5, 50), (35, 15, 3, 30), (10, 40, 4, 40)]
    density_map = generate_hard_bounded_density_field(grid_x, grid_y, obstacles)
    density_fn = get_density_interpolator(grid_x, grid_y, density_map)

    start = (5, 5, 0)
    goal = np.array([45, 45])
    current_pose = np.array(start)
    history = [current_pose[:2]]
    max_iter = 500
    iter = 0
    while np.linalg.norm(current_pose[:2] - goal) > 1.0:
        best_traj, cost_field = optimize_trajectory_with_generated_costfield(
            current_pose, goal, density_fn, grid_x, grid_y, horizon=20, num_samples=200
        )
        current_pose = best_traj[0]  # restore to [x, y, theta] remains
        history.append(current_pose[:2])
        if iter == max_iter:
            break
        iter += 1

    # 시각화
    plt.figure(figsize=(8, 8))
    plt.imshow(density_map, extent=[0, 50, 0, 50], origin='lower', cmap='hot', alpha=0.3)
    plt.plot(*zip(*history), 'cyan', lw=2, label='Trajectory Path')
    plt.scatter(*start[:2], c='green', label='Start')
    plt.scatter(*goal, c='blue', label='Goal')
    plt.legend()
    plt.title("Navigation with Heading Preserved")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    run_navigation_loop()