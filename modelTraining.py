import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load preprocessed dataset
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
X_val = np.load("X_val.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
y_val = np.load("y_val.npy")

# Define CNN Model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (Free or Occupied)
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save model
model.save("parking_lot_model.h5")
print("Model trained and saved successfully!")
