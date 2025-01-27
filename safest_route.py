import numpy as np
from scipy.ndimage import sobel
import heapq  # For priority queue in Dijkstra/A*
import rasterio
import matplotlib.pyplot as plt

def calculate_cost_map(dem_data, slope, safe_zone_map):
    """Calculate cost map based on slope and safe zones."""
    cost_map = np.full(dem_data.shape, np.inf)

    for i in range(dem_data.shape[0]):
        for j in range(dem_data.shape[1]):
            if safe_zone_map[i, j] == 1:  # Safe zone
                cost_map[i, j] = 1  # Lowest cost
            elif not np.isnan(slope[i, j]):  # Regular areas
                cost_map[i, j] = slope[i, j] * 10  # Cost proportional to slope
            else:  # Obstacles (e.g., cliffs, invalid areas)
                cost_map[i, j] = np.inf
    return cost_map

def find_path(cost_map, start, goal):
    """Find the shortest path using Dijkstra's algorithm."""
    rows, cols = cost_map.shape
    visited = np.zeros_like(cost_map, dtype=bool)
    distances = np.full_like(cost_map, np.inf)
    distances[start] = 0
    priority_queue = [(0, start)]

    parent_map = {}

    while priority_queue:
        current_distance, current_cell = heapq.heappop(priority_queue)

        if visited[current_cell]:
            continue

        visited[current_cell] = True

        if current_cell == goal:
            break

        # Explore neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current_cell[0] + dx, current_cell[1] + dy)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and not visited[neighbor]:
                new_distance = current_distance + cost_map[neighbor]
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    parent_map[neighbor] = current_cell
                    heapq.heappush(priority_queue, (new_distance, neighbor))

    # Reconstruct path
    path = []
    current = goal
    while current in parent_map:
        path.append(current)
        current = parent_map[current]
    path.reverse()
    return path

# Input setup
dem_file = r'C:\Users\vinod\OneDrive\文档\project_drone\Idukki_DEM.tif'

# Load DEM data
with rasterio.open(dem_file) as src:
    dem_data = src.read(1)
    dem_data = np.where(dem_data == src.nodata, np.nan, dem_data)

# Compute slope
dx = sobel(dem_data, axis=0)
dy = sobel(dem_data, axis=1)
slope = np.sqrt(dx**2 + dy**2)

# Define safe zone map (example logic, adjust as needed)
safe_zone_map = np.zeros_like(dem_data, dtype=int)
safe_zone_map[dem_data > 500] = 1  # Example safe zones where elevation > 500m

# Define start and goal points (adjust based on your DEM grid)
start = (10, 10)  # Starting cell (row, column)
goal = (100, 100)  # Goal cell (row, column)

# Calculate cost map
cost_map = calculate_cost_map(dem_data, slope, safe_zone_map)

# Find the shortest path
path = find_path(cost_map, start, goal)

# Define zoom range around the path
path_x, path_y = zip(*path)
min_x, max_x = min(path_x) - 10, max(path_x) + 10  # Zoom range in X direction
min_y, max_y = min(path_y) - 10, max(path_y) + 10  # Zoom range in Y direction

# Clip the zoomed area to stay within bounds of the DEM
min_x, max_x = max(min_x, 0), min(max_x, dem_data.shape[0] - 1)
min_y, max_y = max(min_y, 0), min(max_y, dem_data.shape[1] - 1)

# Visualization
plt.figure(figsize=(12, 6))

# Plot DEM (zoomed)
plt.subplot(1, 3, 1)
plt.imshow(dem_data[min_x:max_x, min_y:max_y], cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.title("Zoomed DEM")

# Plot Safe Zone Map (zoomed)
plt.subplot(1, 3, 2)
plt.imshow(safe_zone_map[min_x:max_x, min_y:max_y], cmap='Blues')
plt.title("Zoomed Safe Zones")

# Plot Cost Map with Path (zoomed)
plt.subplot(1, 3, 3)
plt.imshow(cost_map[min_x:max_x, min_y:max_y], cmap='hot', vmax=np.nanmax(cost_map))
plt.plot(np.array(path_y) - min_y, np.array(path_x) - min_x, color='cyan', linewidth=2, label='Path')
plt.title("Zoomed Cost Map and Path")
plt.legend()
plt.tight_layout()
plt.show()

print("Path:", path)
