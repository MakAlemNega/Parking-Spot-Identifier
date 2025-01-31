import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("parking_lot_model.h5")

# Load a video file instead of a webcam
video_path = "dataset/test/parking_crop_loop.mp4"  # Change to your actual file path
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Preprocess the frame
    frame_resized = cv2.resize(frame, (64, 64))  # Resize to match model input
    frame_normalized = frame_resized / 255.0  # Normalize
    frame_expanded = np.expand_dims(frame_normalized, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(frame_expanded)
    label = "Occupied" if prediction[0][0] > 0.5 else "Free"

    # Display prediction on video
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Parking Lot Video", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
