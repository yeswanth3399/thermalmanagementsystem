import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pennylane as qml
from pennylane.optimize import NesterovMomentumOptimizer

# Load modified dataset
file_path = "battery_dataset_with_labels_1.csv"  # Ensure this file is available

data = pd.read_csv(file_path)

# Extract features and labels
X = data[['discharge_rate', 'ambient_temperature']].values
y = data['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the quantum device
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

def qnn(weights, x):
    # Encode input features into quantum states
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)
    
    # Apply variational layers
    for i in range(len(weights)):
        qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=0)
        qml.Rot(weights[i, 3], weights[i, 4], weights[i, 5], wires=1)
        qml.CNOT(wires=[0, 1])
    
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def circuit(weights, x):
    return qnn(weights, x)

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
predictions = [1 if circuit(weights, x) > 0.4 else 0 for x in X_test]

# Accuracy calculation
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot decision boundary
plt.figure(figsize=(8, 6))
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

Z = np.array([circuit(weights, [i, j]) for i, j in np.c_[xx.ravel(), yy.ravel()]])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[Z.min(), 0.5, Z.max()], alpha=0.5, colors=['#FFB6C1', '#ADD8E6'])
plt.contour(xx, yy, Z, colors='k', levels=[0.5], alpha=0.7)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=50, edgecolors='k', label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', s=50, edgecolors='k', marker='x', label='Test Data')

plt.title('QNN for EV Battery Performance Classification')
plt.xlabel('Discharge Rate (C-rate)')
plt.ylabel('Ambient Temperature (°C)')
plt.legend()
plt.colorbar()
plt.show()
