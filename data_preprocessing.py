import pandas as pd

# Load raw data
flight_df = pd.read_csv("data/raw/flight_data.csv", parse_dates=["departure_time", "arrival_time"])
weather_df = pd.read_csv("data/raw/weather_data.csv", parse_dates=["date"])

# Data Cleaning and Preprocessing

## Convert delay column to boolean
flight_df["delay"] = flight_df["delay"].astype(bool)

## Convert temperature to a more common unit, e.g., Fahrenheit to Celsius
weather_df["temperature"] = (weather_df["temperature"] - 32) * 5/9

## Merge flight data with weather data based on date and location
flight_df["departure_date"] = flight_df["departure_time"].dt.date
weather_df["date"] = weather_df["date"].dt.date

flight_weather_df = pd.merge(flight_df, weather_df, left_on=["departure_date", "origin"], right_on=["date", "location"], how="left")
flight_weather_df.drop(columns=["departure_date", "date", "location"], inplace=True)

# Handle missing values if any (for this example, we'll just drop them, but in a real-world scenario you might want to impute them)
flight_weather_df.dropna(inplace=True)

# Save processed data
flight_weather_df.to_csv("data/processed/processed_flight_weather_data.csv", index=False)

print("Data preprocessing is complete. Processed data saved to 'data/processed/'.")

