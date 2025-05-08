# IPDDP with direct NeRF density field querying (no Gaussian kernel smoothing)

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from numpy.linalg import inv
import os
from PIL import Image
import glob

# ------------------------- 기본 유틸리티 -------------------------
def normalize_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def dynamics(x, u, dt):
    theta = x[2]
    v, omega = u
    return np.array([
        x[0] + dt * v * np.cos(theta),
        x[1] + dt * v * np.sin(theta),
        normalize_angle(x[2] + dt * omega)
    ])

def generate_density_field(grid_x, grid_y, obstacles):
    xx, yy = np.meshgrid(grid_x, grid_y)
    density = np.zeros_like(xx)
    for (cx, cy, r, intensity) in obstacles:
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        density += intensity * np.exp(-dist_sq / (2 * r**2))
    return density / np.max(density)

def build_uniform_sampled_costmap(x_seq, density_field, grid_x, grid_y, sigma=6.0, n_samples=50):
    cost_map = np.zeros((len(grid_y), len(grid_x)))
    x_res = grid_x[1] - grid_x[0]
    y_res = grid_y[1] - grid_y[0]
    density_fn = RegularGridInterpolator((grid_y, grid_x), density_field)

    # uniform distance sampling over trajectory
    traj = np.array(x_seq)
    distances = np.cumsum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1))
    total_dist = distances[-1]
    sample_distances = np.linspace(0, total_dist, n_samples)

    sampled_pts = [traj[0, :2]]
    idx = 0
    for d in sample_distances[1:]:
        while idx < len(distances) - 1 and distances[idx] < d:
            idx += 1
        if idx < len(traj):
            sampled_pts.append(traj[idx, :2])

    for pt in sampled_pts:
        pt = np.clip(pt, [grid_x[0], grid_y[0]], [grid_x[-1], grid_y[-1]])
        try:
            d = density_fn(pt.reshape(1, -1))[0]
        except:
            d = 0.0
        xx, yy = np.meshgrid(grid_x, grid_y)
        dist_sq = (xx - pt[0])**2 + (yy - pt[1])**2
        cost_map += d * np.exp(-dist_sq / (2 * sigma**2))

    return cost_map

def build_density_cost_fn(grid_x, grid_y, cost_map):
    interp = RegularGridInterpolator((grid_y, grid_x), cost_map)
    def cost_fn(x):
        pt = np.clip(x[:2], [grid_x[0], grid_y[0]], [grid_x[-1], grid_y[-1]])
        return interp(pt.reshape(1, -1))[0]
    return cost_fn

# ------------------- 실행 및 시각화 -------------------
def run_ipddp_direct_density(x0, goal, density_field, grid_x, grid_y, T=30, dt=0.2, max_iters=20,
                             alpha_list=[1.0, 0.8, 0.5, 0.3, 0.1], convergence_tol=1e-3,
                             w_control=0.1, w_density=10.0, w_goal=10.0, w_smooth=10.0,
                             goal_tolerance=1.0):
    output_dir = "ipddp_with_gaussian_kernel"
    os.makedirs(output_dir, exist_ok=True)
    dx = (goal[0] - x0[0]) / (T * dt)
    dy = (goal[1] - x0[1]) / (T * dt)
    initial_theta = np.arctan2(dy, dx)
    v = np.hypot(dx, dy)
    omega = normalize_angle(initial_theta - x0[2]) / (T * dt)

    u_seq = np.tile(np.array([v, omega]), (T, 1))
    prev_total_cost = float('inf')
    converged_iter = max_iters

    for i in range(max_iters):
        # rollout
        x_seq = [x0]
        for u in u_seq:
            x_seq.append(dynamics(x_seq[-1], u, dt))
        x_seq = np.array(x_seq)

        # uniform resampled costmap
        cost_map = build_uniform_sampled_costmap(x_seq, density_field, grid_x, grid_y)
        cost_fn = build_density_cost_fn(grid_x, grid_y, cost_map)

        best_cost = float('inf')
        best_u_seq = None
        best_x_seq = None

        for alpha in alpha_list:
            u_test = u_seq + alpha * (np.random.randn(*u_seq.shape) * 0.05)
            x_test = [x0]
            total_cost = 0.0

            for t in range(T):
                x = x_test[-1]
                u = u_test[t]
                total_cost += w_control * np.dot(u, u) + w_density * cost_fn(x)
                if t > 0:
                    du = u_test[t] - u_test[t - 1]
                    total_cost += w_smooth * np.dot(du, du)
                x_test.append(dynamics(x, u, dt))

            x_test = np.array(x_test)
            total_cost += w_goal * np.sum((x_test[-1][:2] - goal[:2])**2)

            if total_cost < best_cost:
                best_cost = total_cost
                best_u_seq = u_test
                best_x_seq = x_test

        u_seq = best_u_seq
        x_seq = best_x_seq

        cost_reduction = prev_total_cost - best_cost
        dist_to_goal = np.linalg.norm(x_seq[-1][:2] - goal[:2])

        if dist_to_goal < goal_tolerance and cost_reduction < convergence_tol:
            converged_iter = i + 1
            print(f"✅ Converged at iteration {converged_iter} (ΔJ = {cost_reduction:.6f}, dist = {dist_to_goal:.3f})")
            break

        prev_total_cost = best_cost

        # Save visualization
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.contourf(grid_x, grid_y, density_field, levels=50, cmap='hot')
        plt.plot(x_seq[:, 0], x_seq[:, 1], 'cyan', lw=2, label=f'Iter {i+1}')
        plt.scatter(*x0[:2], c='green', label='Start')
        plt.scatter(*goal[:2], c='red', label='Goal')
        plt.title(f"Global Density Field - Iter {i+1}")
        plt.axis('equal'); plt.grid(True); plt.legend()

        plt.subplot(1, 2, 2)
        plt.contourf(grid_x, grid_y, cost_map, levels=50, cmap='hot')
        plt.plot(x_seq[:, 0], x_seq[:, 1], 'cyan', lw=2)
        plt.title("Local Costmap Used")
        plt.axis('equal'); plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/iteration_{i+1:02d}.png")
        plt.close()

    # Make GIF
    image_files = sorted(glob.glob(f"{output_dir}/iteration_*.png"))
    images = []
    base_shape = None
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        if base_shape is None:
            base_shape = img.size
        else:
            img = img.resize(base_shape)
        images.append(img)

    gif_path = "ipddp_with_gaussian_kernel.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=50, loop=0)
    print(f"✅ Direct density 기반 IPDDP 종료 → GIF 저장됨: {gif_path}")
    return x_seq, u_seq, gif_path

# ---------------- 실행 ----------------
grid_x = np.linspace(0, 50, 100)
grid_y = np.linspace(0, 50, 100)
obstacles = [(15, 20, 2.5, 1.0), (25, 25, 3.0, 1.2), (35, 30, 2.0, 1.0),
             (40, 42, 2.5, 1.5), (20, 40, 2.0, 0.8), (30, 15, 1.5, 1.0)]
density_field = generate_density_field(grid_x, grid_y, obstacles)

start = np.array([5.0, 5.0, 0.0])
goal = np.array([45.0, 45.0, 0.0])

x_seq_direct, u_seq_direct, gif_path_direct = run_ipddp_direct_density(start, goal, density_field, grid_x, grid_y)
