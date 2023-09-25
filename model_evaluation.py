import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load feature-engineered data
flight_weather_df = pd.read_csv("data/processed/feature_engineered_flight_weather_data.csv")

# Define features (X) and target (y)
X = flight_weather_df.drop(columns=['flight_id', 'departure_time', 'arrival_time', 'delay'])
y = flight_weather_df['delay']

# Load the trained model
model = joblib.load("models/logistic_regression_model.joblib")

# Make predictions on the entire dataset
y_pred = model.predict(X)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
classification_rep = classification_report(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)

print(f"Model Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)

# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig("notebooks/confusion_matrix.png")

print("Model evaluation is complete. Confusion matrix saved to 'notebooks/'.")

