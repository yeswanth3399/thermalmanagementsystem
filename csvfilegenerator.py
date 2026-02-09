import pandas as pd
import random

# Generate sample data
num_samples = 100

data = {
    "Discharge Rate (C)": [round(random.uniform(0.5, 3.0), 2) for _ in range(num_samples)],
    "Charge Rate (C)": [round(random.uniform(0.5, 2.5), 2) for _ in range(num_samples)],
    "Ambient Temperature (°C)": [round(random.uniform(10, 45), 2) for _ in range(num_samples)],
    "Battery Temperature (°C)": [round(random.uniform(20, 60), 2) for _ in range(num_samples)],
}

# Define label conditions
labels = []
for i in range(num_samples):
    if data["Battery Temperature (°C)"][i] > 50:
        labels.append("Overheating")
    elif data["Discharge Rate (C)"][i] > 2.5:
        labels.append("High Discharge")
    elif data["Charge Rate (C)"][i] > 2.0:
        labels.append("High Charging")
    elif data["Ambient Temperature (°C)"][i] < 15:
        labels.append("Cold Environment")
    else:
        labels.append("Normal")

data["Label"] = labels

# Create DataFrame
df = pd.DataFrame(data)

# Define numeric labels for each category
label_mapping = {
    "Overheating": 3,
    "High Discharge": 2,
    "High Charging": 2,
    "Cold Environment": 1,
    "Normal": 0
}

# Replace labels with numeric values
df["Label"] = df["Label"].map(label_mapping)

# Save to CSV file
file_path = "EV_battery_data_numeric_labels.csv"
df.to_csv(file_path, index=False)

print(f"CSV file saved as {file_path}")
