import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 1: Define data directory ===
data_dir = r"E:\MLME\release\data"  # Update this to your actual local path
data_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

# === Step 2: Define column renaming map ===
rename_map = {
    'c': 'cPM',
    'T_PM': 'TPM',
    'd10': 'd10',
    'd50': 'd50',
    'd90': 'd90',
    'T_TM': 'TTM',
    'mf_PM': 'QPM',
    'mf_TM': 'QTM',
    'Q_g': 'Qair',
    'w_crystal': 'wcryst',
    'c_in': 'cin',
    'T_PM_in': 'TPMin',
    'T_TM_in': 'TTMin'
}

# === Step 3: Load all files and compute mean of each feature per file ===
mean_records = []

for file in data_files:
    path = os.path.join(data_dir, file)
    try:
        df = pd.read_csv(path, delim_whitespace=True)
        df.rename(columns=rename_map, inplace=True)
        numeric_df = df.select_dtypes(include='number')
        mean_vector = numeric_df.mean()
        mean_vector['source_file'] = file
        mean_records.append(mean_vector)
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Combine all mean vectors into a summary table
mean_df = pd.DataFrame(mean_records)
mean_df.reset_index(drop=True, inplace=True)

# === Step 4: Correlation heatmap across features (from mean values) ===
corr_matrix = mean_df.drop(columns='source_file').corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, linecolor='gray', cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap of Feature Means (Across Files)")
plt.tight_layout()
plt.show()

# === Step 5: Plot mean value trajectory across files ===
mean_data_only = mean_df.drop(columns='source_file')

for col in mean_data_only.columns:
    plt.figure(figsize=(10, 4))
    plt.plot(mean_data_only[col], marker='o')
    plt.title(f"Mean Value of {col} Across Product Runs")
    plt.xlabel("File Index (Sample Proxy)")
    plt.ylabel(f"Mean {col}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# === Step 6: Save the summary table to a CSV file ===
mean_df.to_csv(os.path.join(data_dir, 'mean_feature_summary.csv'), index=False) 