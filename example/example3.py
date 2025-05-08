import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from numpy.linalg import inv
import os
import imageio
import glob

# ----------------------------------------------------
# 유틸 및 dynamics
# ----------------------------------------------------
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

def build_local_costmap_from_trajectory(x_seq, density_field, grid_x, grid_y, sigma=2.0):
    cost_map = np.zeros((len(grid_y), len(grid_x)))
    x_res = grid_x[1] - grid_x[0]
    y_res = grid_y[1] - grid_y[0]
    density_fn = RegularGridInterpolator((grid_y, grid_x), density_field)
    for x in x_seq:
        pt = np.clip(x[:2], [grid_x[0], grid_y[0]], [grid_x[-1], grid_y[-1]])
        try:
            d = density_fn(pt.reshape(1, -1))[0]
        except:
            d = 0.0
        xi = int((pt[0] - grid_x[0]) / x_res)
        yi = int((pt[1] - grid_y[0]) / y_res)
        if 0 <= xi < len(grid_x) and 0 <= yi < len(grid_y):
            cost_map[yi, xi] += d
    return gaussian_filter(cost_map, sigma=sigma)

def build_cost_fn_from_costmap(costmap, grid_x, grid_y):
    interp = RegularGridInterpolator((grid_y, grid_x), costmap)
    def cost_fn(x):
        pt = np.clip(x[:2], [grid_x[0], grid_y[0]], [grid_x[-1], grid_y[-1]])
        return interp(pt.reshape(1, -1))[0]
    return cost_fn

# ----------------------------------------------------
# 선형화 및 비용 근사
# ----------------------------------------------------
def finite_difference_jacobian(f, x, u, dt, eps=1e-4):
    nx, nu = len(x), len(u)
    fx = np.zeros((nx, nx))
    fu = np.zeros((nx, nu))
    f0 = f(x, u, dt)
    for i in range(nx):
        dx = np.zeros(nx)
        dx[i] = eps
        fx[:, i] = (f(x + dx, u, dt) - f0) / eps
    for i in range(nu):
        du = np.zeros(nu)
        du[i] = eps
        fu[:, i] = (f(x, u + du, dt) - f0) / eps
    return fx, fu

def cost_quadratic_approx(x, u, goal, cost_fn):
    Q = np.eye(3) * 0.1
    R = np.eye(2) * 0.1
    q = np.zeros(3)
    r = np.zeros(2)
    l = 0.1 * np.dot(u, u) + 10.0 * cost_fn(x)
    return Q, R, q, r, l

def terminal_cost_approx(x, goal):
    Qf = np.eye(3) * 10.0
    qf = 10.0 * (x - goal)
    lf = 10.0 * np.sum((x[:2] - goal[:2])**2)
    return Qf, qf, lf

# ----------------------------------------------------
# IPDDP core
# ----------------------------------------------------
def ipddp_step(x0, u_seq, goal, density_field, grid_x, grid_y, T, dt):
    nx, nu = 3, 2
    x_seq = [x0]
    for u in u_seq:
        x_seq.append(dynamics(x_seq[-1], u, dt))
    x_seq = np.array(x_seq)

    local_costmap = build_local_costmap_from_trajectory(x_seq, density_field, grid_x, grid_y)
    cost_fn = build_cost_fn_from_costmap(local_costmap, grid_x, grid_y)

    fx_seq, fu_seq = [], []
    Q_seq, R_seq, q_seq, r_seq, l_seq = [], [], [], [], []

    for t in range(T):
        fx, fu = finite_difference_jacobian(dynamics, x_seq[t], u_seq[t], dt)
        Q, R, q, r, l = cost_quadratic_approx(x_seq[t], u_seq[t], goal, cost_fn)
        fx_seq.append(fx)
        fu_seq.append(fu)
        Q_seq.append(Q)
        R_seq.append(R)
        q_seq.append(q)
        r_seq.append(r)
        l_seq.append(l)

    Qf, qf, lf = terminal_cost_approx(x_seq[-1], goal)

    V = Qf
    v = qf
    k_seq = []
    K_seq = []

    for t in reversed(range(T)):
        fx, fu = fx_seq[t], fu_seq[t]
        Q, R = Q_seq[t], R_seq[t]
        q, r = q_seq[t], r_seq[t]

        Q_x = Q + fx.T @ V @ fx
        Q_u = R + fu.T @ V @ fu
        Q_xu = fx.T @ V @ fu
        Q_ux = Q_xu.T
        q_x = q + fx.T @ v
        q_u = r + fu.T @ v

        inv_Q_u = inv(Q_u + 1e-5 * np.eye(nu))
        K = -inv_Q_u @ Q_ux
        k = -inv_Q_u @ q_u

        V = Q_x + K.T @ Q_u @ K + K.T @ Q_ux + Q_ux.T @ K
        v = q_x + K.T @ Q_u @ k + K.T @ q_u + Q_ux.T @ k

        k_seq.insert(0, k)
        K_seq.insert(0, K)

    # Forward rollout
    x = x0
    x_seq_new = [x0]
    u_seq_new = []
    for t in range(T):
        dx = x - x_seq[t]
        u = u_seq[t] + k_seq[t] + K_seq[t] @ dx
        x = dynamics(x, u, dt)
        x_seq_new.append(x)
        u_seq_new.append(u)

    return np.array(x_seq_new), np.array(u_seq_new), local_costmap

# ----------------------------------------------------
# 실행 + 시각화 + GIF 생성
# ----------------------------------------------------
def run_ipddp_with_visualization(x0, goal, density_field, grid_x, grid_y, T=30, dt=0.2, iters=20):
    output_dir = "ipddp_iterations"
    os.makedirs(output_dir, exist_ok=True)

    u_seq = np.zeros((T, 2))
    for i in range(iters):
        x_seq, u_seq, costmap = ipddp_step(x0, u_seq, goal, density_field, grid_x, grid_y, T, dt)

        # 시각화 저장
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.contourf(grid_x, grid_y, density_field, levels=50, cmap='hot')
        plt.plot(x_seq[:, 0], x_seq[:, 1], 'cyan', lw=2, label=f'Iter {i+1}')
        plt.scatter(*x0[:2], c='green', label='Start')
        plt.scatter(*goal[:2], c='red', label='Goal')
        plt.title(f"Global Density Field - Iter {i+1}")
        plt.axis('equal'); plt.grid(True); plt.legend()

        plt.subplot(1, 2, 2)
        plt.contourf(grid_x, grid_y, costmap, levels=50, cmap='hot')
        plt.plot(x_seq[:, 0], x_seq[:, 1], 'cyan', lw=2)
        plt.title("Local Costmap Used")
        plt.axis('equal'); plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/iteration_{i+1:02d}.png")
        plt.close()

    # GIF 생성
    image_files = sorted(glob.glob(f"{output_dir}/iteration_*.png"))
    images = [imageio.imread(img) for img in image_files]
    gif_path = "ipddp_trajectory_optimization.gif"
    imageio.mimsave(gif_path, images, duration=8)

    print(f"✅ 최적화 완료: {iters}번 iteration 수행 → GIF 저장됨: {gif_path}")
    return gif_path

# ----------------------------------------------------
# 실행 파라미터 설정
# ----------------------------------------------------
grid_x = np.linspace(0, 50, 100)
grid_y = np.linspace(0, 50, 100)
obstacles = [(25, 25, 5, 50), (35, 15, 3, 30), (10, 40, 4, 40)]
density_field = generate_density_field(grid_x, grid_y, obstacles)

start = np.array([5.0, 5.0, 0.0])
goal = np.array([45.0, 45.0, 0.0])

# 최적화 및 GIF 생성 실행
run_ipddp_with_visualization(start, goal, density_field, grid_x, grid_y)
