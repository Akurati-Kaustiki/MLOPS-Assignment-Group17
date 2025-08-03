import joblib
import numpy as np

# Load model
model = joblib.load("models/model.pkl")

# Define 4 test samples: [sepal_length, sepal_width, petal_length, petal_width]
samples = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Setosa
    [7.0, 3.2, 4.7, 1.4],  # Versicolor
    [6.3, 3.3, 6.0, 2.5],  # Virginica
    [5.8, 2.7, 5.1, 1.9]   # Virginica
])

# Run predictions
predictions = model.predict(samples)

# Display predictions
print("Predictions for test samples:", predictions.tolist())
