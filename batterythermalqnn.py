import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pennylane as qml
from pennylane.optimize import NesterovMomentumOptimizer

# Load data from CSV
data = pd.read_csv("battery_dataset_with_labels_3.csv")  # Replace "battery_data.csv" with your file path

# Assuming the CSV file has columns 'discharge_rate', 'ambient_temperature', and 'label'
X = data[['discharge_rate', 'ambient_temperature']].values
y = data['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the quantum device and quantum circuit for QNN
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

def qnn(weights, x):
    # Encode classical data into quantum states
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)
    
    # Apply variational layers
    for i in range(len(weights)):
        qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=0)
        qml.Rot(weights[i, 3], weights[i, 4], weights[i, 5], wires=1)
        qml.CNOT(wires=[0, 1])
    
    return qml.expval(qml.PauliZ(0))

# QNode for quantum neural network classifier
@qml.qnode(dev)
def circuit(weights, x):
    return qnn(weights, x)

# Define a simple cost function
def cost(weights, X, y):
    predictions = [circuit(weights, x) for x in X]
    return np.mean((predictions - y) ** 2)

# Initialize weights and optimizer
weights = 0.01 * np.random.randn(2, 6)
opt = NesterovMomentumOptimizer(stepsize=0.01)

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    weights = opt.step(lambda w: cost(w, X_train, y_train), weights)
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Cost: {cost(weights, X_train, y_train)}")

# Prediction on test set
predictions = [1 if circuit(weights, x) > 0.5 else 0 for x in X_test]

# Accuracy calculation
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plotting decision boundary
plt.figure(figsize=(8, 6))

# Create mesh for boundary plot
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Evaluate the QNN for each point in the mesh
Z = np.array([circuit(weights, [i, j]) for i, j in np.c_[xx.ravel(), yy.ravel()]])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and margins
plt.contourf(xx, yy, Z, levels=[Z.min(), 0.5, Z.max()], alpha=0.5, colors=['#FFB6C1', '#ADD8E6'])
plt.contour(xx, yy, Z, colors='k', levels=[0.5], alpha=0.7)

# Scatter plot for training and test data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=50, edgecolors='k', label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', s=50, edgecolors='k', marker='x', label='Test Data')

plt.title('QNN for Thermal Management of EV Batteries')
plt.xlabel('Discharge Rate (C-rate)')
plt.ylabel('Ambient Temperature (°C)')
plt.legend()
plt.colorbar()
plt.show()
