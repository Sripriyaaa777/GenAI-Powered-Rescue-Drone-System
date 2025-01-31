import rasterio
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU
import folium
from folium import raster_layers

# File path to your DEM data
dem_file = r'C:\Users\vinod\OneDrive\文档\project_drone\Idukki_DEM.tif'

# Step 1: Read the DEM data and calculate slope
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

# Step 2: Collect weather data using OpenWeatherMap API (or another source)
api_key = "76325309e81602599360cef60fd9caa5"
lat, lon = 10.22, 76.78  # Example coordinates

# Weather API URL
url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"

# Get weather data
response = requests.get(url)
weather_data = response.json()

# Extract weather features (temperature, humidity, etc.)
temperature = weather_data['main']['temp']
humidity = weather_data['main']['humidity']
rainfall = weather_data.get('rain', {}).get('1h', 0)
wind_speed = weather_data['wind']['speed']

# Step 3: Combine DEM and weather features
data = {
    'elevation': elevation_valid,
    'slope': slope_valid,
    'temperature': np.full(elevation_valid.shape, temperature),
    'humidity': np.full(elevation_valid.shape, humidity),
    'rainfall': np.full(elevation_valid.shape, rainfall),
    'wind_speed': np.full(elevation_valid.shape, wind_speed),
}
data_df = pd.DataFrame(data)

# Define thresholds for safe zones
slope_threshold = 5  # degrees
elevation_threshold = 300  # meters

# Add target labels (safe zone = 1, risky zone = 0)
data_df['label'] = ((data_df['slope'] < slope_threshold) & (data_df['elevation'] > elevation_threshold)).astype(int)

# Step 4: Train-test split
X = data_df[['elevation', 'slope', 'temperature', 'humidity', 'rainfall', 'wind_speed']]
y = data_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Predict safe zones for the entire DEM area
X_valid = np.column_stack((elevation_valid, slope_valid, np.full(elevation_valid.shape, temperature),
                           np.full(elevation_valid.shape, humidity), np.full(elevation_valid.shape, rainfall),
                           np.full(elevation_valid.shape, wind_speed)))
predictions_valid = model.predict(X_valid)

# Create a full array for predictions matching the DEM shape
predictions_full = np.full(elevation_flat.shape, np.nan)
predictions_full[valid_mask] = predictions_valid

# Reshape predictions back to the original DEM shape
safe_zone_map = predictions_full.reshape(rows, cols)

# Step 8: Visualize the results
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

# Step 9: Save the Safe Zones map
output_file = 'predicted_safe_zones.tif'
with rasterio.open(dem_file) as src:
    profile = src.profile
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(safe_zone_map.astype(rasterio.uint8), 1)

print(f"Safe zone map saved to {output_file}")

# Step 10: Use GANs for data augmentation (optional)
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))  # Latent space
    model.add(LeakyReLU(0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(0.2))
    model.add(Dense(np.prod(dem_data.shape), activation='tanh'))  # Generate synthetic DEM
    model.add(Reshape(dem_data.shape))  # Reshape to match DEM shape
    return model

# Instantiate and train the GAN
generator = build_generator()

# Generate synthetic DEM using GAN
latent_vector = np.random.normal(0, 1, (1, 100))  # Random noise vector
synthetic_dem = generator.predict(latent_vector)

# Step 11: Map safe and risky zones on a folium map (Optional)
# Use the safe zone map saved above and visualize it on a Folium map
tif_file = 'predicted_safe_zones.tif'
with rasterio.open(tif_file) as src:
    bounds = src.bounds
    data = src.read(1)

# Create binary masks for safe and risky zones
safe_zone_data = np.where(data == 1, 1, np.nan)
risky_zone_data = np.where(data == 0, 1, np.nan)

# Create a folium map
center_lat = (bounds[1] + bounds[3]) / 2
center_lon = (bounds[0] + bounds[2]) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Add safe and risky zones to the map
folium.raster_layers.ImageOverlay(
    image=safe_zone_data,
    bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
    colormap=lambda x: (0, 0, 255, x * 255),  # Blue for safe zones
    opacity=0.6
).add_to(m)

folium.raster_layers.ImageOverlay(
    image=risky_zone_data,
    bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
    colormap=lambda x: (255, 0, 0, x * 255),  # Red for risky zones
    opacity=0.6
).add_to(m)

# Save the map as an HTML file
m.save('map_with_safe_and_risky_zones.html')
print("Map with safe and risky zones saved as 'map_with_safe_and_risky_zones.html'")
