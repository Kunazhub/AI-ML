import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Example dataset
data = {
    "Feature1": [1, 2, 3, 4, 5],
    "Feature2": [100, 200, 300, 400, 500],
    "Feature3": [1000, 900, 800, 700, 600],
    "Target": [0, 1, 0, 1, 0],
}

# Convert to DataFrame
df = pd.DataFrame(data)

print("df : ",df)

# Split dataset into features and target
X = df.drop("Target", axis=1)
y = df["Target"]

print("X : ",X)

print("y : ",y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling using StandardScaler (Z-score normalization)
scaler = StandardScaler()

# Fit the scaler on the training set and transform both training and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display the scaled data
print("Scaled Training Set:")
print(X_train_scaled)

print("\nScaled Test Set:")
print(X_test_scaled)

# Feature Scaling using MinMaxScaler (Normalization to [0, 1] range)
minmax_scaler = MinMaxScaler()

# Fit the scaler on the training set and transform both training and test sets
X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)

print("\nMinMax Scaled Training Set:")
print(X_train_minmax)

print("\nMinMax Scaled Test Set:")
print(X_test_minmax)
