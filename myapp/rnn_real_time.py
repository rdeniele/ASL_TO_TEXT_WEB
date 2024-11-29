import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')


import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time
from collections import deque

# --------------------- Configuration ---------------------
NUM_LANDMARKS = 42
IMG_SIZE = 224  # Input image size for the model
MODEL_PATH = "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_WEB/ml_models/rnn_asl_model.h5"
LABEL_ENCODER_PATH = "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_WEB/ml_data/labels/rnn_label_encoder.pkl"
SPEECH_DELAY = 3
CONFIDENCE_THRESHOLD = 0.8
AUDIO_FILE_LIFETIME = 2
DEBUG_FRAME_LIFETIME = 5
UNKNOWN_LABEL = "unknown"
FRAME_WIDTH = 600
FRAME_HEIGHT = 500
VISUALIZATION_SIZE = 400  # Size for the preprocessing visualization
SEQUENCE_LENGTH = 30  # Length of the sequence for RNN
# ----------------------------------------------------------

# Load trained RNN model and label encoder
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=10,
                       min_detection_confidence=0.9, 
                       min_tracking_confidence=0.9) 

cap = cv2.VideoCapture(0)

# Global variables
last_speech_time = 0
previous_label = ""
debug_frame_time = 0

# Prediction smoothing using a deque 
prediction_history = deque(maxlen=5) # Store last 5 predictions
landmark_sequence = deque(maxlen=SEQUENCE_LENGTH)  # Store the sequence of landmarks

def extract_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    return landmarks

def normalize_landmarks(landmarks):
    x_coords = [lm[0] for lm in landmarks]
    y_coords = [lm[1] for lm in landmarks]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    normalized_landmarks = [
        [(x - min_x) / (max_x - min_x), (y - min_y) / (max_y - min_y), z]
        for x, y, z in landmarks
    ]
    return normalized_landmarks

def preprocess_landmarks(landmarks, img_size=IMG_SIZE):
    normalized_landmarks = normalize_landmarks(landmarks)
    landmarks_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    if normalized_landmarks:
        # Draw connections
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            x1, y1 = int(normalized_landmarks[start_idx][0] * (img_size - 1)), int(
                normalized_landmarks[start_idx][1] * (img_size - 1))
            x2, y2 = int(normalized_landmarks[end_idx][0] * (img_size - 1)), int(
                normalized_landmarks[end_idx][1] * (img_size - 1))
            cv2.line(landmarks_image, (x1, y1),
                     (x2, y2), (255, 255, 255), 2)

        # Draw landmark points
        for lm in normalized_landmarks:
            x, y = int(lm[0] * (img_size - 1)), int(lm[1] * (img_size - 1))
            cv2.circle(landmarks_image, (x, y), 5, (255, 0, 0), 3)

    return landmarks_image, np.expand_dims(landmarks_image / 255.0, axis=0)

# Simplified prediction smoothing
def smooth_predictions(new_prediction):
    prediction_history.append(new_prediction)
    return np.mean(prediction_history, axis=0)

class MovingAverageFilter:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.predictions = []

    def update(self, new_prediction):
        self.predictions.append(new_prediction)
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
        return np.mean(self.predictions, axis=0)

ma_filter = MovingAverageFilter()

# Create a named window for the preprocessing visualization
cv2.namedWindow("Preprocessed Landmarks", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Preprocessed Landmarks", VISUALIZATION_SIZE, VISUALIZATION_SIZE)

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    predicted_label = UNKNOWN_LABEL
    confidence = 0.0
    
    # Create a black background for preprocessed visualization
    preprocessed_viz = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = extract_landmarks(hand_landmarks)
            for lm in landmarks:
                x = int(lm[0] * frame.shape[1])
                y = int(lm[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Get both the visualization and model input
            preprocessed_viz, input_data = preprocess_landmarks(landmarks)
            landmark_sequence.append(input_data)
            
            if len(landmark_sequence) == SEQUENCE_LENGTH:
                input_sequence = np.expand_dims(np.array(landmark_sequence), axis=0)
                input_sequence = np.squeeze(input_sequence, axis=2)  # Remove the extra dimension
                raw_predictions = model.predict(input_sequence, verbose=0)[0]
                filtered_predictions = smooth_predictions(raw_predictions)
                predicted_class_index = np.argmax(filtered_predictions)

                try:
                    predicted_label = le.inverse_transform([predicted_class_index])[0]
                except IndexError:
                    predicted_label = UNKNOWN_LABEL

                confidence = filtered_predictions[predicted_class_index]

            hand_label_position = (int(landmarks[0][0] * frame.shape[1]), int(landmarks[0][1] * frame.shape[0]) - 30)
            cv2.putText(frame, f"{predicted_label}",
                        hand_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    else:  
        cv2.putText(frame, f"Prediction: {predicted_label}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the preprocessed landmarks visualization
    cv2.imshow("Preprocessed Landmarks", preprocessed_viz)

    current_time = time.time()

    if debug_frame_time and (time.time() - debug_frame_time) >= DEBUG_FRAME_LIFETIME:
        frame = np.zeros_like(frame)
        debug_frame_time = 0

    cv2.imshow("Real-time ASL Detection", frame)

    if cv2.getWindowProperty("Real-time ASL Detection", cv2.WND_PROP_VISIBLE) >= 1:
        debug_frame_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()