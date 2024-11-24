import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import os

class ASLProcessor:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        
        # Load model and label encoder
        
        # self.model = tf.keras.models.load_model('path/to/your/model.h5')
        self.model = tf.keras.models.load_model('C:/Users/ronde/PROJECTS/ASL_TO_TEXT_WEB/ml_models/asl_mobilenetv2_model.h5')
        # with open('path/to/your/label_encoder.pkl', 'rb') as f:
        with open('C:/Users/ronde/PROJECTS/ASL_TO_TEXT_WEB/ml_data/labels/cnn_label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        self.img_size = 224
        
    def extract_landmarks(self, hand_landmarks):
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        return landmarks
    
    def normalize_landmarks(self, landmarks):
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        normalized_landmarks = [
            [(x - min_x) / (max_x - min_x), (y - min_y) / (max_y - min_y), z]
            for x, y, z in landmarks
        ]
        return normalized_landmarks
    
    def preprocess_landmarks(self, landmarks):
        normalized_landmarks = self.normalize_landmarks(landmarks)
        landmarks_image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        if normalized_landmarks:
            for connection in self.mp_hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                x1, y1 = int(normalized_landmarks[start_idx][0] * (self.img_size - 1)), \
                         int(normalized_landmarks[start_idx][1] * (self.img_size - 1))
                x2, y2 = int(normalized_landmarks[end_idx][0] * (self.img_size - 1)), \
                         int(normalized_landmarks[end_idx][1] * (self.img_size - 1))
                cv2.line(landmarks_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                
            for lm in normalized_landmarks:
                x, y = int(lm[0] * (self.img_size - 1)), int(lm[1] * (self.img_size - 1))
                cv2.circle(landmarks_image, (x, y), 5, (255, 0, 0), 3)
                
        return np.expand_dims(landmarks_image / 255.0, axis=0)
    
    def process_frame(self, frame):
        
        frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            landmarks = self.extract_landmarks(results.multi_hand_landmarks[0])
            input_data = self.preprocess_landmarks(landmarks)
            
            predictions = self.model.predict(input_data, verbose=0)[0]
            predicted_class_index = np.argmax(predictions)
            confidence = float(predictions[predicted_class_index])
            
            try:
                predicted_label = self.label_encoder.inverse_transform([predicted_class_index])[0]
            except IndexError:
                predicted_label = "unknown"
                
            return {
                'prediction': predicted_label,
                'confidence': confidence
            }
            
        return {
            'prediction': 'unknown',
            'confidence': 0.0
        }