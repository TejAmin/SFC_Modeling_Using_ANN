import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN for consistent performance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Input, Dropout # type: ignore
#--------------------------------------------------------------------------------------
# --- Configuration ---
data_dir = "C:/Users/send2/OneDrive/Desktop/MLME/PAS_Project/project_release/release/Data"  # Folder containing all .txt files
n_lags = 2
output_csv = 'narx_full_preprocessed_dataset.csv'

# NARX variable definitions
output_cols = ['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM']
input_cols = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']

# Storage
X_all = []
Y_all = []
trajectory_ids = []
traj_id = 0

# List all .txt files
file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])
print(f"✅ Total files found: {len(file_list)}")

for filename in file_list:
    path = os.path.join(data_dir, filename)
    print(f"📂 Processing file {traj_id + 1}/{len(file_list)}: {filename}")
    try:
        df = pd.read_csv(path, sep='\t')

        # Ensure all required columns exist
        for col in output_cols + input_cols:
            if col not in df.columns:
                df[col] = np.nan

        df = df[output_cols + input_cols].dropna()

        if len(df) <= n_lags:
            print(f"⚠️ Skipping {filename} (too few rows)")
            continue

        for i in range(n_lags, len(df) - 1):
            y_hist = []
            u_hist = []
            for lag in range(n_lags, -1, -1):
                y_hist.extend(df[output_cols].iloc[i - lag].values)
                u_hist.extend(df[input_cols].iloc[i - lag].values)

            x = y_hist + u_hist
            y = df[output_cols].iloc[i + 1].values

            X_all.append(x)
            Y_all.append(y)
            trajectory_ids.append(traj_id)

        traj_id += 1

    except Exception as e:
        print(f"❌ Skipped {filename} due to error: {e}")
        continue

# Convert to DataFrame
X_all = np.array(X_all)
Y_all = np.array(Y_all)
trajectory_ids = np.array(trajectory_ids)

# Generate readable column names
feature_names = []
for lag in range(n_lags, -1, -1):
    feature_names += [f"{col}_lag{lag}" for col in output_cols]
for lag in range(n_lags, -1, -1):
    feature_names += [f"{col}_lag{lag}" for col in input_cols]

feature_df = pd.DataFrame(X_all, columns=feature_names)
for j, col in enumerate(output_cols):
    feature_df[col + '_target'] = Y_all[:, j]
feature_df['trajectory_id'] = trajectory_ids

# Save to CSV
feature_df.to_csv(output_csv, index=False)
print(f"✅ Preprocessing complete. Saved to {output_csv}")
