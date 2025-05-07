import numpy as np
import casadi as ca
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# ----------------------------------------------
# Instant-NGP 쿼리를 대체할 density field 생성 (예시)
# ----------------------------------------------
def generate_density_field(grid_x, grid_y, obstacles):
    xx, yy = np.meshgrid(grid_x, grid_y)
    density = np.zeros_like(xx)
    for (cx, cy, r, intensity) in obstacles:
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        density += intensity * np.exp(-dist_sq / (2 * r**2))
    return density / np.max(density)

# ----------------------------------------------
# 샘플 trajectory 생성
# ----------------------------------------------
def sample_trajectories(x0, T, N, dt, noise_scale):
    trajs = []
    for _ in range(N):
        x = np.array(x0)
        traj = [x.copy()]
        for _ in range(T):
            v, omega = np.random.randn(2) * noise_scale
            theta = x[2]
            x_new = np.array([
                x[0] + dt * v * np.cos(theta),
                x[1] + dt * v * np.sin(theta),
                x[2] + dt * omega
            ])
            x = x_new
            traj.append(x.copy())
        trajs.append(np.array(traj))
    return trajs
# ----------------------------------------------
# 샘플 기반 local costmap 생성
# ----------------------------------------------
def generate_costmap_from_samples(trajs, density_fn, grid_x, grid_y, sigma=2.0):
    cost_map = np.zeros((len(grid_y), len(grid_x)))
    x_res = grid_x[1] - grid_x[0]
    y_res = grid_y[1] - grid_y[0]
    all_points = np.vstack(trajs)
    densities = density_fn(np.clip(all_points, [grid_x[0], grid_y[0]], [grid_x[-1], grid_y[-1]]))
    for pt, d in zip(all_points, densities):
        xi = int((pt[0] - grid_x[0]) / x_res)
        yi = int((pt[1] - grid_y[0]) / y_res)
        if 0 <= xi < len(grid_x) and 0 <= yi < len(grid_y):
            cost_map[yi, xi] += d
    return gaussian_filter(cost_map, sigma=sigma)

# ----------------------------------------------
# IPDDP (CasADi + IPOPT) local trajectory optimizer
# ----------------------------------------------
def run_ipddp_local_planner(x0, goal, cost_field, grid_x, grid_y, T=15, dt=0.2):
    nx, nu = 3, 2  # x, y, theta | v, omega
    X = ca.MX.sym('X', nx, T+1)
    U = ca.MX.sym('U', nu, T)
    v_min = -0.5
    v_max = 1.5
    omega_min = -0.8
    omega_max = 0.8

    lbx = []
    ubx = []
    for _ in range(T+1):  # state X: no bounds
        lbx += [-ca.inf]*nx
        ubx += [ ca.inf]*nx
    for _ in range(T):  # control U: bounded
        lbx += [v_min, omega_min]
        ubx += [v_max, omega_max]

    cost_lut = ca.interpolant('cost_lut', 'linear',
        [grid_x.tolist(), grid_y.tolist()],
        cost_field.flatten(order='F')
    )

    def cost_fn(x): return cost_lut(ca.vertcat(x[0], x[1]))

    cost = 0
    g = []
    for t in range(T):
        xt, ut = X[:, t], U[:, t]
        x_next = X[:, t+1]

        x, y, theta = xt[0], xt[1], xt[2]
        v, omega = ut[0], ut[1]

        # Dynamics constraint
        x_pred = ca.vertcat(
            x + dt * v * ca.cos(theta),
            y + dt * v * ca.sin(theta),
            theta + dt * omega
        )
        g.append(x_next - x_pred)

        # Cost
        cost += 0.1 * ca.dot(ut, ut) + 1000.0 * cost_fn(xt)

    # goal cost: only x, y
    goal_sym = ca.MX(goal[:2])
    cost += 1000 * ca.dot(X[0:2, -1] - goal_sym, X[0:2, -1] - goal_sym)

    vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    g_flat = ca.vertcat(*g)
    solver = ca.nlpsol("solver", "ipopt", {"x": vars, "f": cost, "g": g_flat})

    x_init = np.tile(np.array(x0).reshape(-1, 1), T+1)
    u_init = np.random.uniform(low=np.array([[0.1], [0.0]]),
                           high=np.array([[0.2], [0.1]]),
                           size=(nu, T))
    vars_init = np.concatenate([x_init.flatten(), u_init.flatten()])

    sol = solver(x0=vars_init, lbg=np.zeros(g_flat.numel()), ubg=np.zeros(g_flat.numel()), lbx=lbx, ubx=ubx)
    w_opt = sol['x'].full().flatten()
    X_opt = w_opt[:nx*(T+1)].reshape((nx, T+1))
    return X_opt

# ----------------------------------------------
# 반복적 local planning으로 글로벌 경로 생성
# ----------------------------------------------
def plan_path(start, goal, grid_x, grid_y, density_field, steps=20):
    path = [start.copy()]
    density_fn = RegularGridInterpolator((grid_y, grid_x), density_field)
    x = start.copy()
    for _ in range(steps):
        trajs = sample_trajectories(x, T=15, N=50, dt=0.2, noise_scale=1.0)
        cost_field = generate_costmap_from_samples([traj[:, :2] for traj in trajs], density_fn, grid_x, grid_y)
        X_opt = run_ipddp_local_planner(x, goal, cost_field, grid_x, grid_y)
        x_next = X_opt[:, 1]
        path.append(x_next)
        x = x_next
        if np.linalg.norm(x[:2] - goal[:2]) < 1.0:
            break
    return np.array(path)

# ===============================================
# 실행
# ===============================================
grid_x = np.linspace(0, 50, 100)
grid_y = np.linspace(0, 50, 100)
obstacles = [(25, 25, 3.0, 1.0), (35, 40, 3.0, 0.8)]
density_field = generate_density_field(grid_x, grid_y, obstacles)

start = np.array([5.0, 5.0, 0.0])  # x, y, theta
goal = np.array([45.0, 45.0, 0.0]) # x, y only used
path = plan_path(start, goal, grid_x, grid_y, density_field)

plt.figure(figsize=(6,6))
plt.contourf(grid_x, grid_y, density_field, levels=50, cmap='hot')
plt.plot(path[:, 0], path[:, 1], 'cyan', lw=2, label='Trajectory Path')
plt.scatter(*start[:2], c='green', label='Start')
plt.scatter(*goal[:2], c='red', label='Goal')
plt.title("Differential Drive Planning via IPDDP")
plt.xlabel("X"); plt.ylabel("Y")
plt.axis('equal'); plt.grid(True); plt.legend(); plt.show()
