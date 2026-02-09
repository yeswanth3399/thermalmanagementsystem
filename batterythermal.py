import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

# Load the dataset
# Replace 'battery_data.csv' with the path to your actual dataset file
data = pd.read_csv('battery_dataset.csv')

# Extract features
X = data[['discharge_rate', 'ambient_temperature']].values  # Feature matrix
temperature = data['battery_temperature'].values  # Measured temperature values

# Define a threshold for safe vs. unsafe temperature (e.g., 60°C)
threshold = 60
y = (temperature > threshold).astype(int)  # Label as 1 if temperature > threshold, else 0

# Print out unique labels to ensure both classes are present
unique_labels, counts = np.unique(y, return_counts=True)
print(f"Unique labels in the dataset: {dict(zip(unique_labels, counts))}")

# Check if we have more than one class
if len(unique_labels) > 1:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVM classifier
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # Plotting decision boundary
    plt.figure(figsize=(8, 6))

    # Create a mesh to plot decision boundary
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Decision function to determine the class for each point in the mesh
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Ensure levels are strictly increasing for contourf
    min_level = Z.min()
    max_level = Z.max()
    levels = np.linspace(min_level, max_level, num=3)

    # Plot the decision boundary and the margins
    plt.contourf(xx, yy, Z, levels=levels, alpha=0.5, colors=['#FFB6C1', '#ADD8E6'])
    plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.7)

    # Scatter plot of training data
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=50, edgecolors='k', label='Training Data')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', s=50, edgecolors='k', marker='x', label='Test Data')

    plt.title('SVM for Thermal Management of EV Batteries')
    plt.xlabel('Discharge Rate (C-rate)')
    plt.ylabel('Ambient Temperature (°C)')
    plt.legend()
    plt.colorbar()
    plt.show()

else:
    print("Error: Only one class is present in the data. Adjust the threshold or verify the data.")
