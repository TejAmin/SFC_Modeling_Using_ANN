import os
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Define your data directory
data_dir = r"E:\MLME\release\data"

# Step 2: Check if directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory not found: {data_dir}")

# Step 3: Get the first .txt file
data_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
if not data_files:
    raise FileNotFoundError("No .txt files found in the directory.")
first_file_path = os.path.join(data_dir, data_files[0])
print(f"Loading: {first_file_path}")

# Step 4: Load the first file
df = pd.read_csv(first_file_path, delim_whitespace=True)

# Step 5: Rename state vector columns
df.rename(columns={
    'c': 'cPM',
    'T_PM': 'TPM',
    'd10': 'd10',
    'd50': 'd50',
    'd90': 'd90',
    'T_TM': 'TTM'
}, inplace=True)

# Step 6: Select state vector
state_vector = ['TPM', 'cPM', 'd10', 'd50', 'd90', 'TTM']
state_data = df[state_vector]

# Step 7: Plot each state variable
for var in state_vector:
    plt.figure(figsize=(10, 4))
    plt.plot(state_data.index, state_data[var])
    plt.title(f"Time Series of {var}")
    plt.xlabel("Sample Index (Proxy for Time)")
    plt.ylabel(var)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
