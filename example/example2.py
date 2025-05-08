
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def generate_sample_trajectory(start, v, w_traj, dt=0.5):
    x, y, theta = start
    traj = []
    for w in w_traj:
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt
        traj.append([x, y, theta])
    return np.array(traj)

def generate_density_field(grid_x, grid_y, obstacles):
    density = np.zeros((len(grid_y), len(grid_x)))
    for (cx, cy, r, intensity) in obstacles:
        xx, yy = np.meshgrid(grid_x, grid_y)
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        density += intensity * np.exp(-dist_sq / (2 * (r**2)))
    return density

def get_density_interpolator(grid_x, grid_y, density_map):
    return RegularGridInterpolator((grid_y, grid_x), density_map, bounds_error=False, fill_value=0)

def compute_log_likelihood_cost(traj, goal, density_fn, sigma_s=1.0, sigma_g=5.0, sigma_d=5.0):
    traj_xy = traj[:, :2]
    diffs = np.diff(traj_xy, axis=0)
    smoothness = np.sum(np.linalg.norm(np.diff(diffs, axis=0), axis=1) ** 2)
    goal_dist = np.linalg.norm(traj_xy[-1] - goal) ** 2
    sigma_vals = density_fn(traj_xy)
    density_sum = np.sum(sigma_vals)
    cost = (smoothness / (2 * sigma_s**2)) + (goal_dist / (2 * sigma_g**2)) + (density_sum / (2 * sigma_d**2))
    return cost

def mle_with_density(start, goal, density_fn, horizon=20, num_samples=200):
    best_cost = np.inf
    best_traj = None
    for _ in range(num_samples):
        v = np.random.uniform(0.5, 1.5)
        w_traj = np.random.uniform(-0.5, 0.5, size=horizon)
        traj = generate_sample_trajectory(start, v, w_traj)
        cost = compute_log_likelihood_cost(traj, goal, density_fn)
        if cost < best_cost:
            best_cost = cost
            best_traj = traj
    return best_traj, best_cost

def run_mle_navigation_loop():
    grid_x = np.linspace(0, 50, 200)
    grid_y = np.linspace(0, 50, 200)
    obstacles = [(25, 25, 5, 50), (35, 15, 3, 30), (10, 40, 4, 40)]
    density_map = generate_density_field(grid_x, grid_y, obstacles)
    density_fn = get_density_interpolator(grid_x, grid_y, density_map)

    start = (5, 5, 0)
    goal = np.array([45, 45])
    current_pose = np.array(start)
    history = [current_pose[:2]]

    while np.linalg.norm(current_pose[:2] - goal) > 1.0:
        best_traj, cost = mle_with_density(current_pose, goal, density_fn)
        current_pose = best_traj[0]  # 한 단계만 진행
        history.append(current_pose[:2])

    plt.figure(figsize=(8, 8))
    plt.imshow(density_map, extent=[0, 50, 0, 50], origin='lower', cmap='hot', alpha=0.3)
    plt.plot(*zip(*history), 'cyan', lw=2, label='Trajectory Path')
    plt.scatter(*start[:2], c='green', label='Start')
    plt.scatter(*goal, c='blue', label='Goal')
    plt.legend()
    plt.title("Iterative MLE Navigation with Density Penalty")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    run_mle_navigation_loop()