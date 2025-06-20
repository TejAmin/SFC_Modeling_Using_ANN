# ✅ Step 3: Train Model for Cluster 0 (model_cluster0_train.py)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# --- Load clustered data ---
df = pd.read_csv("narx_clustered_dataset.csv")

# --- Filter for Cluster 0 only ---
df_cluster0 = df[df["cluster_id"] == 0].reset_index(drop=True)

# --- Define inputs and outputs ---
input_cols = [col for col in df.columns if '_lag' in col]
output_cols = ['d10_target', 'd50_target', 'd90_target']

X = df_cluster0[input_cols].values
y = df_cluster0[output_cols].values

# --- Normalize inputs ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# --- Build ANN model ---
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3)  # 3 outputs: d10, d50, d90
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- Train model ---
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1
)

# --- Evaluate ---
y_pred = model.predict(X_test)
# Convert predictions and actuals from meters to microns
y_pred_um = y_pred * 1e6
y_test_um = y_test * 1e6

print("MAE:", mean_absolute_error(y_test_um, y_pred_um))
print("MSE:", mean_squared_error(y_test_um, y_pred_um))

# --- Save model and scaler ---
model.save("model_cluster0.keras")
joblib.dump(scaler, "scaler_cluster0.pkl")
print("✅ Model and scaler saved for Cluster 0")

import matplotlib.pyplot as plt

# Extract metrics from history
mse = history.history['loss']
val_mse = history.history['val_loss']
mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = range(1, len(mse) + 1)

# Plot MSE
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, mse, label='Training MSE')
plt.plot(epochs, val_mse, label='Validation MSE')
plt.title('Mean Squared Error Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(epochs, mae, label='Training MAE')
plt.plot(epochs, val_mae, label='Validation MAE')
plt.title('Mean Absolute Error Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
