import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("parking_lot_model.h5")

# Load a video file
video_path = "dataset/test/parking_crop_loop.mp4"  # Change this to your actual video file path
cap = cv2.VideoCapture(video_path)

# Define parking space coordinates manually (format: (x, y, width, height))
parking_spaces = [
    (0, 2, 962, 98),
    (907, 2, 433, 345),
    (1221, 2, 39, 27),
    (1258, 2, 45, 31),
    (1262, 2, 5, 3),
    (1293, 2, 31, 24),
    (1306, 2, 121, 39),
    (1326, 2, 27, 30),
    (1326, 2, 5, 11),
    (1354, 2, 4, 3)
]  # Adjust coordinates to match your video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if the video ends

    occupied_count = 0
    total_spaces = len(parking_spaces)

    for (x, y, w, h) in parking_spaces:
        # Extract parking spot image
        parking_spot = frame[y:y + h, x:x + w]

        # Preprocess the image for the model
        spot_resized = cv2.resize(parking_spot, (64, 64))
        spot_normalized = spot_resized / 255.0
        spot_expanded = np.expand_dims(spot_normalized, axis=0)

        # Make a prediction
        prediction = model.predict(spot_expanded)
        label = "Occupied" if prediction[0][0] > 0.5 else "Free"

        # Set colors: Red for occupied, Green for free
        color = (0, 0, 255) if label == "Occupied" else (0, 255, 0)

        # Draw a rectangle around the parking spot
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Put label inside the box
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Count occupied spaces
        if label == "Occupied":
            occupied_count += 1

    # Display overall status in the top-left corner
    status_text = f"Occupied: {occupied_count}/{total_spaces}"
    status_color = (0, 0, 255) if occupied_count > 0 else (0, 255, 0)  # Red if any space is occupied

    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    # Show video with rectangles
    cv2.imshow("Parking Lot Video", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
