import folium
import rasterio
import numpy as np
from folium import Marker
import random

# Step 1: Read the GeoTIFF using rasterio
tif_file = 'safe_risky_zones.tif'

with rasterio.open(tif_file) as src:
    bounds = src.bounds  # (left, bottom, right, top)
    data = src.read(1)
    transform = src.transform  # Affine transformation for geo-coordinates

# Step 2: Identify all safe and risky zone points
safe_points = np.argwhere(data == 1)  # Safe zones (value = 1)
risky_points = np.argwhere(data == 0)  # Risky zones (value = 0)

# Step 3: Select random safe zones (e.g., 5) and exactly 5 risky zones
safe_samples = safe_points[random.sample(range(len(safe_points)), min(5, len(safe_points)))]
risky_samples = risky_points[random.sample(range(len(risky_points)), min(5, len(risky_points)))]

# Step 4: Convert pixel coordinates to geographic coordinates
def pixel_to_geo(row, col, transform):
    """Convert row, col pixel indices to geographic coordinates."""
    lon, lat = rasterio.transform.xy(transform, row, col)
    return lat, lon

# Step 5: Create a folium map centered on the image
center_lat = (bounds[1] + bounds[3]) / 2
center_lon = (bounds[0] + bounds[2]) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Step 6: Add markers for **5 random safe zones**
for row, col in safe_samples:
    lat, lon = pixel_to_geo(row, col, transform)
    Marker([lat, lon], popup="Safe Zone", icon=folium.Icon(color="blue")).add_to(m)

# Step 7: Add markers for **exactly 5 random risky zones**
for row, col in risky_samples:
    lat, lon = pixel_to_geo(row, col, transform)
    Marker([lat, lon], popup="Risky Zone", icon=folium.Icon(color="red")).add_to(m)

# Step 8: Save and display the map
m.save('map_with_random_safe_and_risky_zones.html')
print("Map with random safe and risky zones saved as 'map_with_random_safe_and_risky_zones.html'")
