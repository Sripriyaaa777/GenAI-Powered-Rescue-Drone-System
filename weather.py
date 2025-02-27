import requests

# Your WeatherStack API key
API_KEY = "49f6f22a6caa65f9562c6208b8d9c4ff"
city = "New York"

# WeatherStack API endpoint
url = f"http://api.weatherstack.com/current"

# Set parameters for the request
params = {
    "access_key": API_KEY,  # Your API key
    "query": city,           # City you want the weather data for
}

# Send Request
response = requests.get(url, params=params)

# Parse the response JSON
data = response.json()

# Check if the 'current' key exists in the response
if 'current' in data:
    # Extract relevant data from the response
    temperature = data['current']['temperature']
    precipitation = data['current']['precip']
    cloud_cover = data['current']['cloudcover']
    wind_speed = data['current']['wind_speed']
    wind_dir = data['current']['wind_dir']
    humidity = data['current']['humidity']
    weather_description = data['current']['weather_descriptions'][0]

    # Print the weather data
    print(f"Temperature: {temperature}°C")
    print(f"Precipitation: {precipitation} mm")
    print(f"Cloud Cover: {cloud_cover}%")
    print(f"Wind Speed: {wind_speed} km/h")
    print(f"Wind Direction: {wind_dir}")
    print(f"Humidity: {humidity}%")
    print(f"Weather Description: {weather_description}")
else:
    print("Error: 'current' data is missing. Here's the full response:")
    print(data)  # Print the full response to debug
