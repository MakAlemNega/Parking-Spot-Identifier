import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define dataset paths
DATASET_DIR = "dataset"
CATEGORIES = ["free", "occupied"]
IMG_SIZE = 64  # Resize images to 64x64

# Load and preprocess images
def load_data():
    data, labels = [], []
    for category in CATEGORIES:
        path = os.path.join(DATASET_DIR, category)
        label = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
                data.append(image)
                labels.append(label)
    
    return np.array(data), np.array(labels)

# Load dataset
X, y = load_data()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Save data as NumPy arrays
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("X_val.npy", X_val)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("y_val.npy", y_val)

print("Dataset prepared successfully!")
