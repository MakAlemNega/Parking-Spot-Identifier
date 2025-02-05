i# Parking-Spot-Identifier
# Parking Lot Occupancy Detection

## Overview
This project uses a **Convolutional Neural Network (CNN)** to detect and classify parking spaces in images and videos. The model is trained on labeled images of parking spots, distinguishing between **occupied** and **free** spaces.

## Project Workflow
The system follows these main steps:

1. **Data Preprocessing**: Loads images, resizes, normalizes, and splits them into training, validation, and test sets.
2. **Model Training**: A CNN model is trained on the dataset to classify parking spaces.
3. **Model Inference on Video**: The trained model is used to classify real-time parking spots in a video.
4. **Parking Spot Detection in Video**: Detects multiple parking spots in a video, labels them, and counts occupied vs. free spaces.

## Folder Structure
```
├── dataset/
│   ├── free/            # Images of free parking spaces
│   ├── occupied/        # Images of occupied parking spaces
│   ├── test/            # Test videos for model inference
├── scripts/
│   ├── data_preprocessing.py  # Prepares the dataset
│   ├── model_training.py      # Trains the CNN model
│   ├── video_inference.py     # Classifies parking spots in a video
│   ├── spot_detection.py   # Detects multiple parking spots
├── parking_lot_model.h5  # Trained model file
├── README.md             # Project documentation
```

## Steps to Run the Project
### 1. Install Dependencies
Make sure you have Python installed, then install the required libraries:
```sh
pip install tensorflow numpy opencv-python scikit-learn
```

### 2. Preprocess Data
Run the data preprocessing script to load and prepare images:
```sh
python scripts/data_preprocessing.py
```
This script:
- Loads images from the **dataset/** folder
- Resizes them to **64x64** pixels
- Normalizes pixel values (0-1 range)
- Splits the data into **training, validation, and test sets**

### 3. Train the Model
Run the training script:
```sh
python scripts/model_training.py
```
This script:
- Defines a **CNN model**
- Trains it using the dataset
- Saves the trained model as `parking_lot_model.h5`

### 4. Run Model Inference on Video
To test the trained model on a video, run:
```sh
python scripts/video_inference.py
```
This script:
- Reads a video file
- Applies the CNN model to classify each frame
- Displays predictions ("Free" or "Occupied") on the video

### 5. Detect Parking Spots in Video
To detect multiple parking spots in a video, run:
```sh
python scripts/parking_detection.py
```
This script:
- Reads a parking lot video
- Detects predefined parking spaces
- Classifies each space as **free** or **occupied**
- Draws boxes and labels on each parking spot
- Counts the number of occupied vs. free spaces

## Notes
- You need to adjust **parking spot coordinates** in `parking_detection.py` to match your video.
- You can use any **labeled dataset** of parking spaces for training.

## Future Improvements
- Implement **real-time detection** using a webcam.
- Improve **model accuracy** with more training data.
- Add **object detection** to automatically find parking spots.

## Contributors
- **Makbel A.** – Developer
- **Samson S.** - Developer 
---

Now you're ready to run the project! 🚀 Let me know if you have any questions.

 
