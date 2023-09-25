import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Number of data points
n = 1000

# Generate random flight data
flight_data = {
    "flight_id": range(1, n+1),
    "departure_time": [datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in range(n)],
    "arrival_time": [datetime.now() + timedelta(days=np.random.randint(1, 365)) for _ in range(n)],
    "origin": np.random.choice(["LAX", "JFK", "ATL", "SFO", "DFW"], n),
    "destination": np.random.choice(["LAX", "JFK", "ATL", "SFO", "DFW"], n),
    "delay": np.random.choice([True, False], n)
}

flight_df = pd.DataFrame(flight_data)

# Save raw flight data
flight_df.to_csv("data/raw/flight_data.csv", index=False)

# Generate random weather data
weather_data = {
    "date": pd.date_range(start="2022-01-01", periods=n).tolist(),
    "location": np.random.choice(["LAX", "JFK", "ATL", "SFO", "DFW"], n),
    "temperature": np.random.uniform(low=-10, high=35, size=n),
    "precipitation": np.random.uniform(low=0, high=50, size=n)
}

weather_df = pd.DataFrame(weather_data)

# Save raw weather data
weather_df.to_csv("data/raw/weather_data.csv", index=False)

print("Data collection is complete. Raw data saved to 'data/raw/'.")

