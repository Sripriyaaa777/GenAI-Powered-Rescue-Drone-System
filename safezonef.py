# Step 1: Import required libraries
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from rasterio.windows import Window

# File path to your DEM data
dem_file = r'C:\Users\vinod\OneDrive\文档\project_drone\Idukki_DEM.tif'

# Define the region of interest (ROI) in pixel coordinates (xmin, ymin, xmax, ymax)
roi_xmin, roi_ymin, roi_xmax, roi_ymax = 1000, 1000, 2000, 2000  # Example values

# Step 2: Read the DEM data and calculate slope
with rasterio.open(dem_file) as src:
    # Read the DEM data for the ROI
    window = Window(roi_xmin, roi_ymin, roi_xmax - roi_xmin, roi_ymax - roi_ymin)
    dem_data = src.read(1, window=window)
    transform = src.window_transform(window)
    crs = src.crs

# Calculate slope using Sobel operator
dx = sobel(dem_data, axis=0)
dy = sobel(dem_data, axis=1)
slope = np.sqrt(dx**2 + dy**2)

# Step 3: Identify safe zones
slope_threshold = 5  # degrees
elevation_threshold = 300  # meters
safe_zones = (slope < slope_threshold) & (dem_data > elevation_threshold)

# Get the pixel coordinates of the safe zones
safe_zone_pixels = np.column_stack(np.where(safe_zones))

# Convert pixel coordinates to geographic coordinates
safe_zone_coords = [
    rasterio.transform.xy(transform, row, col, offset='center')
    for row, col in safe_zone_pixels
]

# Extract a few coordinates for demonstration (e.g., first 10 safe zones)
sample_coords = safe_zone_coords[:10]

# Step 4: Visualize DEM, Slope Map, and Safe Zones
plt.figure(figsize=(18, 6))

# Plot the cropped DEM data
plt.subplot(1, 3, 1)
plt.imshow(dem_data, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.title("Zoomed-in DEM")

# Plot the slope data for the zoomed-in region
plt.subplot(1, 3, 2)
plt.imshow(slope, cmap='viridis')
plt.colorbar(label='Slope')
plt.title("Zoomed-in Slope Map")

# Plot the safe zones
plt.subplot(1, 3, 3)
plt.imshow(safe_zones, cmap='Blues')
plt.title("Zoomed-in Predicted Safe Zones")
plt.tight_layout()
plt.show()

# Step 5: Save the Safe Zones and Print Sample Coordinates
safe_zones_file = 'safe_zones_zoomed.tif'
with rasterio.open(dem_file) as src:
    profile = src.profile
    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(safe_zones_file, 'w', **profile) as dst:
        dst.write(safe_zones.astype(rasterio.uint8), 1)

print(f"Zoomed-in Safe Zones saved to {safe_zones_file}")

# Print sample safe zone coordinates
print("Sample coordinates of safe zones (lat, lon):")
for coord in sample_coords:
    print(coord)
