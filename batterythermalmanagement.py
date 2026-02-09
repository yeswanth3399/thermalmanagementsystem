import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

# Synthetic data creation: features (e.g., discharge rate, ambient temperature) and labels (e.g., safe or unsafe temperature levels)
np.random.seed(42)
n_samples = 100
discharge_rate = np.random.uniform(0.5, 2.0, n_samples)  # Discharge rate in C-rate
ambient_temperature = np.random.uniform(20, 40, n_samples)  # Ambient temperature in °C

# Assume a simple relationship where high discharge rate and high ambient temp lead to overheating
temperature = discharge_rate * ambient_temperature + np.random.normal(0, 5, n_samples)
labels = (temperature > 60).astype(int)  # Label 1 if temperature > 60°C, else 0

# Combine features into a matrix
X = np.column_stack((discharge_rate, ambient_temperature))
y = labels

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

# Plot the decision boundary and the margins
plt.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], alpha=0.5, colors=['#FFB6C1', '#ADD8E6'])
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
