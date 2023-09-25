import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
flight_weather_df = pd.read_csv("data/processed/processed_flight_weather_data.csv", parse_dates=["departure_time", "arrival_time"])

# Set the style of the visualization
sns.set(style="whitegrid")

# 1. Distribution of flight delays
plt.figure(figsize=(6, 4))
sns.countplot(x='delay', data=flight_weather_df)
plt.title('Distribution of Flight Delays')
plt.savefig("notebooks/distribution_of_flight_delays.png")

# 2. Distribution of temperatures
plt.figure(figsize=(6, 4))
sns.histplot(flight_weather_df['temperature'], bins=30, kde=True)
plt.title('Distribution of Temperature')
plt.savefig("notebooks/distribution_of_temperature.png")

# 3. Relationship between delay and temperature
plt.figure(figsize=(6, 4))
sns.boxplot(x='delay', y='temperature', data=flight_weather_df)
plt.title('Relationship between Flight Delay and Temperature')
plt.savefig("notebooks/relationship_between_delay_and_temperature.png")

# 4. Average delay per origin airport
avg_delay_origin = flight_weather_df.groupby("origin")["delay"].mean().reset_index()
plt.figure(figsize=(6, 4))
sns.barplot(x='origin', y='delay', data=avg_delay_origin)
plt.title('Average Delay per Origin Airport')
plt.savefig("notebooks/average_delay_per_origin_airport.png")

print("Exploratory Data Analysis is complete. Plots saved to 'notebooks/'.")

