import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

DATA_DIR = "data/processed"

# input files
file_bases = sorted(set(
    f.replace("_x.npy", "")
    for f in os.listdir(DATA_DIR) if f.endswith("_x.npy")
))

# Load all data
X_all = []  # shape: (num_files * 30, 129, 14)
Y_all = []  # shape: (num_files * 30, 6)

for base in file_bases:
    x = np.load(os.path.join(DATA_DIR, f"{base}_x.npy"))
    y = np.load(os.path.join(DATA_DIR, f"{base}_y_types.npy"))
    X_all.append(x)
    Y_all.append(y)

X_all = np.concatenate(X_all, axis=0)
Y_all = np.concatenate(Y_all, axis=0)

# Train/test split
X_train, X_val, Y_train, Y_val = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)

# Normalize input
X_train = X_train.astype("float32") / np.max(X_train)
X_val = X_val.astype("float32") / np.max(X_val)

# Expand dimensions to fit CNN input shape
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)

# Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(129, 14, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(6, activation='sigmoid')  # Multi-label output
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(
    X_train, Y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, Y_val)
)

# Save model
model.save("models/vehicle_classifier.h5")
print("Model saved as models/vehicle_classifier.h5")
