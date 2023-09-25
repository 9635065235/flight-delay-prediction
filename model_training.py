import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load feature-engineered data
flight_weather_df = pd.read_csv("data/processed/feature_engineered_flight_weather_data.csv")

# Define features (X) and target (y)
X = flight_weather_df.drop(columns=['flight_id', 'departure_time', 'arrival_time', 'delay'])
y = flight_weather_df['delay']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict the target for the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)

# Save the trained model
joblib.dump(model, "models/logistic_regression_model.joblib")

print("Model training is complete. Trained model saved to 'models/'.")

