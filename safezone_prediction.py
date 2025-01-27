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

# Step 9: Extract coordinates of safe zones (where value is 1)
safe_zone_coords = np.column_stack(np.where(safe_zone_map == 1))

# Display the first few safe zone coordinates
print("First few safe zone coordinates (row, col):")
print(safe_zone_coords[:5])

# Step 10: Convert row and column indices to geographic coordinates
geographic_coords = [~transform * (col, row) for row, col in safe_zone_coords[:5]]
print("Geographic coordinates of the first few safe zones (longitude, latitude):")
print(geographic_coords)
