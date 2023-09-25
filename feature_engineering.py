import pandas as pd
import numpy as np

# Load processed data
flight_weather_df = pd.read_csv("data/processed/processed_flight_weather_data.csv", parse_dates=["departure_time", "arrival_time"])

# Feature Engineering

## Create 'flight_duration' feature
flight_weather_df['flight_duration'] = (flight_weather_df['arrival_time'] - flight_weather_df['departure_time']).dt.total_seconds() / 60

## Create 'is_weekend' feature
flight_weather_df['is_weekend'] = flight_weather_df['departure_time'].dt.weekday >= 5

## Create 'departure_month' feature
flight_weather_df['departure_month'] = flight_weather_df['departure_time'].dt.month

## Create 'departure_hour' feature
flight_weather_df['departure_hour'] = flight_weather_df['departure_time'].dt.hour

## One-hot encode categorical variables (e.g., origin, destination)
flight_weather_df = pd.get_dummies(flight_weather_df, columns=['origin', 'destination'], drop_first=True)

# Save feature-engineered dataset
flight_weather_df.to_csv("data/processed/feature_engineered_flight_weather_data.csv", index=False)

print("Feature engineering is complete. Feature-engineered data saved to 'data/processed/'.")

