import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from numpy.linalg import inv
import os
import imageio.v2 as imageio  # deprecation warning 해결
import glob

# ---------------------- 유틸리티 ----------------------
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

def make_density_field_fn(density_field, grid_x, grid_y):
    return RegularGridInterpolator((grid_y, grid_x), density_field)

# ---------------------- 선형화 및 비용 근사 ----------------------
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

def cost_quadratic_approx(x, u, goal):
    Q = np.eye(3) * 0.1
    R = np.eye(2) * 0.1
    q = np.zeros(3)
    r = np.zeros(2)
    l = 0.1 * np.dot(u, u)
    return Q, R, q, r, l

def terminal_cost_approx(x, goal):
    Qf = np.eye(3) * 10.0
    qf = 10.0 * (x - goal)
    lf = 10.0 * np.sum((x[:2] - goal[:2])**2)
    return Qf, qf, lf

# ---------------------- IPDDP Step ----------------------
def ipddp_step(x0, u_seq, goal, density_field, grid_x, grid_y, T, dt):
    nx, nu = 3, 2
    x_seq = [x0]
    for u in u_seq:
        x_seq.append(dynamics(x_seq[-1], u, dt))
    x_seq = np.array(x_seq)

    fx_seq, fu_seq = [], []
    Q_seq, R_seq, q_seq, r_seq = [], [], [], []

    for t in range(T):
        fx, fu = finite_difference_jacobian(dynamics, x_seq[t], u_seq[t], dt)
        Q, R, q, r, _ = cost_quadratic_approx(x_seq[t], u_seq[t], goal)
        fx_seq.append(fx)
        fu_seq.append(fu)
        Q_seq.append(Q)
        R_seq.append(R)
        q_seq.append(q)
        r_seq.append(r)

    Qf, qf, _ = terminal_cost_approx(x_seq[-1], goal)
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

    x = x0
    x_seq_new = [x0]
    u_seq_new = []
    for t in range(T):
        dx = x - x_seq[t]
        u = u_seq[t] + k_seq[t] + K_seq[t] @ dx
        x = dynamics(x, u, dt)
        x_seq_new.append(x)
        u_seq_new.append(u)

    return np.array(x_seq_new), np.array(u_seq_new)

# ---------------------- 실행 ----------------------
def run_ipddp_full(x0, goal, density_field, grid_x, grid_y, T=30, dt=0.2, max_iters=50,
                   alpha_list=[1.0, 0.8, 0.5, 0.3, 0.1], convergence_tol=1e-9,
                   w_control=0.1, w_density=10.0, w_goal=10.0, w_smooth=10.0,
                   goal_tolerance=0.1):
    output_dir = "ipddp_without_gaussian_kernel"
    os.makedirs(output_dir, exist_ok=True)
    u_seq = np.zeros((T, 2))
    prev_total_cost = float('inf')
    converged_iter = max_iters

    density_fn = make_density_field_fn(density_field, grid_x, grid_y)

    for i in range(max_iters):
        x_seq, u_seq_candidate = ipddp_step(x0, u_seq, goal, density_field, grid_x, grid_y, T, dt)

        best_cost = float('inf')
        best_u_seq = None
        best_x_seq = None

        for alpha in alpha_list:
            u_test = u_seq + alpha * (u_seq_candidate - u_seq)
            x_test = [x0]
            total_cost = 0.0

            for t in range(T):
                x = x_test[-1]
                u = u_test[t]
                total_cost += w_control * np.dot(u, u) + w_density * density_fn(x[:2])[0]
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

        # 시각화 저장
        plt.figure(figsize=(6, 6), dpi=100)
        plt.contourf(grid_x, grid_y, density_field, levels=50, cmap='hot')
        plt.plot(x_seq[:, 0], x_seq[:, 1], 'cyan', lw=2, label=f'Iter {i+1}')
        plt.scatter(*x0[:2], c='green', label='Start')
        plt.scatter(*goal[:2], c='red', label='Goal')
        plt.title(f"Density Field - Iter {i+1}")
        plt.axis('equal'); plt.grid(True); plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/iteration_{i+1:02d}.png")
        plt.close()

    # GIF 생성
    image_files = sorted(glob.glob(f"{output_dir}/iteration_*.png"))
    images = []
    for img_path in image_files:
        img = imageio.imread(img_path)
        if img.shape == images[0].shape if images else True:
            images.append(img)
        else:
            print(f"⚠ 이미지 크기 불일치: {img_path}, shape = {img.shape}")

    gif_path = "ipddp_full_optimized.gif"
    imageio.mimsave(gif_path, images, duration=5)
    print(f"✅ 총 {converged_iter}번 iteration 후 종료 → GIF 저장됨: {gif_path}")
    return x_seq, u_seq, gif_path

# ---------------------- 실행 예시 ----------------------
if __name__ == "__main__":
    grid_x = np.linspace(0, 50, 100)
    grid_y = np.linspace(0, 50, 100)
    obstacles = [
        (15, 20, 2.5, 1.0),
        (25, 25, 3.0, 1.2),
        (35, 30, 2.0, 1.0),
        (40, 42, 2.5, 1.5),
        (20, 40, 2.0, 0.8),
        (30, 15, 1.5, 1.0)
    ]
    density_field = generate_density_field(grid_x, grid_y, obstacles)

    start = np.array([5.0, 5.0, 0.0])
    goal = np.array([45.0, 45.0, 0.0])

    x_seq_final, u_seq_final, gif_path_final = run_ipddp_full(
        start, goal, density_field, grid_x, grid_y,
        w_control=0.1, w_density=5.0, w_goal=50.0, w_smooth=20.0
    )
