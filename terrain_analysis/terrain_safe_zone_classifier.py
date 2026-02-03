# Step 1: Import required libraries
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# File path to your DEM data
dem_file = r'C:\Users\vinod\OneDrive\文档\project_drone\Idukki_DEM.tif'

# Step 2: Read the DEM data and calculate slope
with rasterio.open(dem_file) as src:
    dem_data = src.read(1)
    transform = src.transform
    crs = src.crs

# Handle invalid values (e.g., nodata regions)
dem_data = np.where(dem_data == src.nodata, np.nan, dem_data)

# Calculate slope using Sobel operator
dx = sobel(dem_data, axis=0)
dy = sobel(dem_data, axis=1)
slope = np.sqrt(dx**2 + dy**2)

# Flatten arrays while preserving the original shape
rows, cols = dem_data.shape
elevation_flat = dem_data.ravel()
slope_flat = slope.ravel()

# Create a mask for valid data points
valid_mask = ~np.isnan(elevation_flat) & ~np.isnan(slope_flat)

# Extract valid data points
elevation_valid = elevation_flat[valid_mask]
slope_valid = slope_flat[valid_mask]

# Combine valid features into a dataset
data = {
    'elevation': elevation_valid,
    'slope': slope_valid,
}
data_df = pd.DataFrame(data)

# Define thresholds for safe zones
slope_threshold = 5  # degrees
elevation_threshold = 300  # meters

# Add target labels
data_df['label'] = ((data_df['slope'] < slope_threshold) & (data_df['elevation'] > elevation_threshold)).astype(int)

# Step 3: Train-test split
X = data_df[['elevation', 'slope']]
y = data_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: Predict safe zones for the entire DEM area
X_valid = np.column_stack((elevation_valid, slope_valid))
predictions_valid = model.predict(X_valid)

# Create a full array for predictions matching the DEM shape
predictions_full = np.full(elevation_flat.shape, np.nan)
predictions_full[valid_mask] = predictions_valid

# Reshape predictions back to the original DEM shape
safe_zone_map = predictions_full.reshape(rows, cols)

# Step 7: Visualize the results
plt.figure(figsize=(18, 6))

# Plot the original DEM
plt.subplot(1, 3, 1)
plt.imshow(dem_data, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.title("DEM")

# Plot the slope map
plt.subplot(1, 3, 2)
plt.imshow(slope, cmap='viridis')
plt.colorbar(label='Slope')
plt.title("Slope Map")

# Plot the predicted safe zones
plt.subplot(1, 3, 3)
plt.imshow(safe_zone_map, cmap='Blues')
plt.title("Predicted Safe Zones")
plt.tight_layout()
plt.show()

# Step 8: Save the Safe Zones map
output_file = 'predicted_safe_zones.tif'
with rasterio.open(dem_file) as src:
    profile = src.profile
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(safe_zone_map.astype(rasterio.uint8), 1)

print(f"Safe zone map saved to {output_file}")

import folium
import rasterio
import numpy as np
from folium import raster_layers

# Step 1: Read the GeoTIFF using rasterio
tif_file = 'safe_risky_zones.tif'

with rasterio.open(tif_file) as src:
    # Read the spatial bounds of the image
    bounds = src.bounds  # (left, bottom, right, top)
    
    # Read the data (assuming 0 = risky, 1 = safe)
    data = src.read(1)
    
    # Check the unique values in the raster data to verify it's binary
    print("Unique values in the raster data:", np.unique(data))

    # Get the CRS (Coordinate Reference System) of the image
    crs = src.crs

# Step 2: Create binary masks for safe and risky zones
# Safe zones: 1 for safe, 0 for risky (assuming data has 1 for safe and 0 for risky)
safe_zone_data = np.where(data == 1, 1, np.nan)  # Safe zones are marked as 1
risky_zone_data = np.where(data == 0, 1, np.nan)  # Risky zones are marked as 1

# Debugging: Check the shape of the data and masks
print("Shape of data array:", data.shape)
print("Safe zone data (first 10 elements):", safe_zone_data.flatten()[:10])
print("Risky zone data (first 10 elements):", risky_zone_data.flatten()[:10])

# Step 3: Set up the folium map
# Calculate the center of the image based on its bounds
center_lat = (bounds[1] + bounds[3]) / 2
center_lon = (bounds[0] + bounds[2]) / 2

# Create a folium map centered on the image
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Step 4: Add safe and risky zones to the map as overlays
# Safe zones - Adding as blue
folium.raster_layers.ImageOverlay(
    image=safe_zone_data,  # Safe zones (1 = safe)
    bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],  # Use the bounds of the image
    colormap=lambda x: (0, 0, 255, x * 255),  # Blue for safe zones
    opacity=0.6
).add_to(m)

# Risky zones - Adding as red
folium.raster_layers.ImageOverlay(
    image=risky_zone_data,  # Risky zones (1 = risky)
    bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],  # Use the bounds of the image
    colormap=lambda x: (255, 0, 0, x * 255),  # Red for risky zones
    opacity=0.6
).add_to(m)

# Step 5: Display the map
m.save('map_with_safe_and_risky_zones.html')
print("Map with safe and risky zones saved as 'map_with_safe_and_risky_zones.html'")
