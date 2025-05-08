
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

class RRTStarNode:
    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent
        self.cost = 0.0

def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def steer(from_node, to_position, max_dist=1.0):
    vec = np.array(to_position) - from_node.position
    dist = np.linalg.norm(vec)
    if dist <= max_dist:
        return to_position
    return from_node.position + (vec / dist) * max_dist

def is_collision(p1, p2, density_fn, threshold=5.0):
    line = np.linspace(p1, p2, 10)
    densities = density_fn(line)
    return np.any(densities > threshold)

def find_rrt_star_path(start, goal, density_fn, grid_bounds, max_iter=1000, step_size=1.0):
    nodes = [RRTStarNode(start)]
    goal_node = None

    for _ in range(max_iter):
        rand_point = np.random.uniform(grid_bounds[0], grid_bounds[1], size=2)
        nearest_node = min(nodes, key=lambda node: euclidean(node.position, rand_point))
        new_pos = steer(nearest_node, rand_point, max_dist=step_size)
        if is_collision(nearest_node.position, new_pos, density_fn):
            continue
        new_node = RRTStarNode(new_pos, parent=nearest_node)
        new_node.cost = nearest_node.cost + euclidean(nearest_node.position, new_pos)
        nodes.append(new_node)
        if euclidean(new_pos, goal) < 2.0:
            goal_node = new_node
            break

    if goal_node is None:
        return []

    path = []
    current = goal_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    path.reverse()
    return np.array(path)

def generate_density_field(grid_x, grid_y, obstacles):
    density = np.zeros((len(grid_y), len(grid_x)))
    for (cx, cy, r, intensity) in obstacles:
        xx, yy = np.meshgrid(grid_x, grid_y)
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        density += intensity * np.exp(-dist_sq / (2 * (r**2)))
    return density

def get_density_interpolator(grid_x, grid_y, density_map):
    return RegularGridInterpolator((grid_y, grid_x), density_map, bounds_error=False, fill_value=0)

def sample_trajectory(start, v, w_arr, dt=0.5):
    x, y, theta = start
    traj = []
    for w in w_arr:
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt
        traj.append([x, y, theta])
    return np.array(traj)

def compute_stage1_cost(traj, goal, weight_smooth=1.0, weight_goal=1.5):
    traj_xy = traj[:, :2]
    diffs = np.diff(traj_xy, axis=0)
    smoothness = np.sum(np.linalg.norm(np.diff(diffs, axis=0), axis=1))
    goal_dist = np.linalg.norm(traj_xy[-1] - np.array(goal))
    return weight_smooth * smoothness + weight_goal * goal_dist

def build_cost_field_vectorized(all_waypoints, density_fn, grid_x, grid_y, sigma=2.0):
    cost_field = np.zeros((len(grid_y), len(grid_x)))
    x_res = grid_x[1] - grid_x[0]
    y_res = grid_y[1] - grid_y[0]
    for wp in all_waypoints:
        xi = int((wp[0] - grid_x[0]) / x_res)
        yi = int((wp[1] - grid_y[0]) / y_res)
        if 0 <= xi < len(grid_x) and 0 <= yi < len(grid_y):
            sigma_val = density_fn([[wp[0], wp[1]]])[0]
            cost_field[yi, xi] += sigma_val
    return gaussian_filter(cost_field, sigma=sigma)

def get_cost_interpolator_from_field(cost_field, grid_x, grid_y):
    return RegularGridInterpolator((grid_y, grid_x), cost_field, bounds_error=False, fill_value=0)

def compute_stage2_cost_with_costfield(traj, cost_fn, density_weight = 100):
    traj_xy = traj[:, :2]
    return density_weight * np.sum(cost_fn(traj_xy))

def plan_with_mppi_rrt_vectorized_costfield(start, goal, density_fn, global_path,
                                            grid_x, grid_y, horizon=20, num_samples=100, waypoint_count=2000):
    trajs = []
    costs = []
    all_waypoints = []
    for _ in range(num_samples):
        v = np.random.uniform(0.5, 1.5)
        w_traj = np.random.uniform(-0.5, 0.5, size=horizon)
        traj = sample_trajectory(start, v, w_traj, dt=0.5)
        all_waypoints.extend(traj[:, :2])
        trajs.append(traj)
    cost_field = build_cost_field_vectorized(np.array(all_waypoints), density_fn, grid_x, grid_y, sigma=2.0)
    cost_fn = get_cost_interpolator_from_field(cost_field, grid_x, grid_y)
    for traj in trajs:
        cost1 = compute_stage1_cost(traj, goal)
        cost2 = compute_stage2_cost_with_costfield(traj, cost_fn)
        total_cost = cost1 + cost2
        costs.append(total_cost)
    best_idx = np.argmin(costs)
    return trajs[best_idx], costs[best_idx]

def navigate_v3():
    grid_x = np.linspace(0, 50, 200)
    grid_y = np.linspace(0, 50, 200)
    obstacles = [(25, 25, 5, 50), (35, 15, 3, 30), (10, 40, 4, 40)]
    density_map = generate_density_field(grid_x, grid_y, obstacles)
    density_fn = get_density_interpolator(grid_x, grid_y, density_map)
    start = (5, 5, 0)
    goal = (45, 45)
    current_pose = np.array(start)
    global_path = find_rrt_star_path(start[:2], goal, density_fn, grid_bounds=((0, 0), (50, 50)))
    history = [current_pose[:2]]
    while np.linalg.norm(current_pose[:2] - np.array(goal)) > 1.0:
        best_traj, cost = plan_with_mppi_rrt_vectorized_costfield(current_pose, goal, density_fn, global_path,
                                                                   grid_x, grid_y, horizon=20, num_samples=100)
        current_pose = best_traj[0]
        history.append(current_pose[:2])
    plt.figure(figsize=(8, 8))
    plt.imshow(density_map, extent=[0, 50, 0, 50], origin='lower', cmap='hot', alpha=0.3)
    plt.plot(*zip(*history), 'cyan', lw=2, label='Navigation Path')
    # plt.plot(global_path[:, 0], global_path[:, 1], 'white', lw=1, label='Global Path (RRT*)')
    plt.scatter(*start[:2], c='green', label='Start')
    plt.scatter(*goal, c='blue', label='Goal')
    plt.legend()
    plt.title("Optimized MPPI Navigation with Vectorized Cost Field")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    navigate_v3()