import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
import pygame
import time
import os
import gc
from gtts import gTTS
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NUM_LANDMARKS = 42
IMG_SIZE = 320
SEQUENCE_LENGTH = 30
MODELS_CONFIG = {
    'cnn': {
        'path': "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/models/gesture_model_cnn.h5",
        'weight': 0.6
    },
    'rnn': {
        'path': "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/models/gesture_model_rnn.h5",
        'weight': 0.4
    }
}
LABEL_ENCODER_PATHS = {
    'cnn': "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/data/labels/cnn_label_encoder.pkl",
    'rnn': "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_FILES/data/labels/rnn_label_encoder.pkl"
}

class ASLDetector:
    def __init__(self):
        self.setup_parameters()
        self.initialize_components()
        
    def setup_parameters(self):
        """Initialize configuration parameters"""
        self.SPEECH_DELAY = 3
        self.CONFIDENCE_THRESHOLD = 0.85
        self.FRAME_WIDTH = 600
        self.FRAME_HEIGHT = 500
        self.MIN_HAND_DETECTION_CONFIDENCE = 0.7
        self.PREDICTION_SMOOTHING_WINDOW = 5
        
    def initialize_components(self):
        """Initialize all components and resources"""
        try:
            pygame.init()
            pygame.mixer.init()
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            
        self.models = self._load_models()
        self.label_encoders = self._load_label_encoders()
        
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils  # Add this for drawing
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.MIN_HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=0.5
        )
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.audio_file_counter = 0
        self.last_speech_time = 0
        self.previous_label = ""
        self.frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.recent_predictions = deque(maxlen=self.PREDICTION_SMOOTHING_WINDOW)
        self.audio_executor = ThreadPoolExecutor(max_workers=2)

    def _load_models(self) -> Dict[str, tf.keras.Model]:
        """Load models with error handling"""
        models = {}
        for model_name, config in MODELS_CONFIG.items():
            try:
                models[model_name] = tf.keras.models.load_model(config['path'])
                logger.info(f"Successfully loaded {model_name} model")
            except Exception as e:
                logger.error(f"Error loading {model_name} model: {e}")
                models[model_name] = None
        return models
    
    def _load_label_encoders(self) -> Dict[str, object]:
        """Load label encoders for each model"""
        encoders = {}
        for model_type, path in LABEL_ENCODER_PATHS.items():
            try:
                with open(path, 'rb') as f:
                    encoders[model_type] = pickle.load(f)
                logger.info(f"Successfully loaded {model_type} label encoder")
            except Exception as e:
                logger.error(f"Error loading {model_type} label encoder: {e}")
                encoders[model_type] = None
        return encoders

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame with enhancement techniques"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        return cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)

    def extract_landmarks(self, hand_landmarks) -> List[List[float]]:
        """Extract and normalize landmarks with additional processing"""
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        
        # Calculate velocity features (movement between frames)
        if hasattr(self, 'previous_landmarks') and self.previous_landmarks is not None:
            for i in range(len(landmarks)):
                velocity = [
                    landmarks[i][j] - self.previous_landmarks[i][j]
                    for j in range(3)
                ]
                landmarks[i].extend(velocity)
        else:
            # If no previous landmarks, add zero velocity
            for landmark in landmarks:
                landmark.extend([0.0, 0.0, 0.0])
        
        self.previous_landmarks = landmarks
        return landmarks

    def preprocess_landmarks(self, landmarks: List[List[float]]) -> np.ndarray:
        """Convert landmarks to image representation"""
        landmarks_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        
        # Normalize coordinates to image space
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Avoid division by zero
        x_range = max_x - min_x
        y_range = max_y - min_y
        
        if x_range == 0: x_range = 1
        if y_range == 0: y_range = 1
        
        normalized_landmarks = [
            ((x - min_x) / x_range * (IMG_SIZE - 1),
             (y - min_y) / y_range * (IMG_SIZE - 1))
            for x, y in zip(x_coords, y_coords)
        ]
        
        # Draw connections
        for connection in self.mp_hands.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start_point = tuple(map(int, normalized_landmarks[start_idx]))
            end_point = tuple(map(int, normalized_landmarks[end_idx]))
            
            cv2.line(landmarks_image, start_point, end_point, (255, 255, 255), 2)
        
        # Draw landmark points
        for point in normalized_landmarks:
            x, y = map(int, point)
            cv2.circle(landmarks_image, (x, y), 5, (0, 0, 255), -1)
            
        return landmarks_image

    def draw_landmarks(self, frame: np.ndarray, landmarks: List[List[float]], hand_landmarks) -> None:
        """Draw hand landmarks and connections on the frame"""
        # Draw the hand mesh
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            self.mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
        )
        
        # Draw additional visualization
        height, width = frame.shape[:2]
        for lm in landmarks:
            x, y = int(lm[0] * width), int(lm[1] * height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    def display_prediction(self, frame: np.ndarray, predicted_label: str, confidence: float, fps: float) -> None:
        """Display prediction and FPS information on frame"""
        # Display prediction
        cv2.putText(frame,
                   f"Sign: {predicted_label}",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1,
                   (0, 255, 0) if confidence >= self.CONFIDENCE_THRESHOLD else (0, 165, 255),
                   2)
        
        # Display confidence
        cv2.putText(frame,
                   f"Confidence: {confidence:.2f}",
                   (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1,
                   (0, 255, 0) if confidence >= self.CONFIDENCE_THRESHOLD else (0, 165, 255),
                   2)
        
        # Display FPS
        cv2.putText(frame,
                   f"FPS: {fps:.2f}",
                   (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1,
                   (255, 0, 0),
                   2)

    def get_ensemble_prediction(self, input_data: np.ndarray) -> Tuple[str, float]:
        """Get ensemble prediction from both CNN and RNN models"""
        cnn_model = self.models['cnn']
        rnn_model = self.models['rnn']
        cnn_encoder = self.label_encoders['cnn']
        rnn_encoder = self.label_encoders['rnn']
        
        if cnn_model is None or rnn_model is None or cnn_encoder is None or rnn_encoder is None:
            return "Unknown", 0.0
        
        cnn_prediction = cnn_model.predict(input_data)
        rnn_prediction = rnn_model.predict(input_data)
        
        cnn_confidence = np.max(cnn_prediction)
        rnn_confidence = np.max(rnn_prediction)
        
        cnn_label = cnn_encoder.inverse_transform([np.argmax(cnn_prediction)])[0]
        rnn_label = rnn_encoder.inverse_transform([np.argmax(rnn_prediction)])[0]
        
        if cnn_confidence * MODELS_CONFIG['cnn']['weight'] > rnn_confidence * MODELS_CONFIG['rnn']['weight']:
            return cnn_label, cnn_confidence
        else:
            return rnn_label, rnn_confidence

    def handle_text_to_speech(self, predicted_label: str, confidence: float) -> None:
        """Handle text-to-speech output with timing control"""
        current_time = time.time()
        
        if (predicted_label != self.previous_label and
            confidence >= self.CONFIDENCE_THRESHOLD and
            (current_time - self.last_speech_time) >= self.SPEECH_DELAY):
            
            try:
                tts = gTTS(text=predicted_label, lang='en')
                audio_filename = f"temp_{self.audio_file_counter}.mp3"
                tts.save(audio_filename)
                
                def play_and_cleanup():
                    try:
                        pygame.mixer.music.load(audio_filename)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            pygame.time.Clock().tick(10)
                        os.remove(audio_filename)
                    except Exception as e:
                        logger.error(f"Error in audio playback: {e}")
                
                self.audio_executor.submit(play_and_cleanup)
                self.audio_file_counter += 1
                self.last_speech_time = current_time
                self.previous_label = predicted_label
                
            except Exception as e:
                logger.error(f"Error in text-to-speech: {e}")

    def cleanup_old_files(self):
        """Clean up old temporary audio files"""
        current_time = time.time()
        for filename in os.listdir():
            if filename.startswith("temp_") and filename.endswith(".mp3"):
                file_path = os.path.join(os.getcwd(), filename)
                if current_time - os.path.getctime(file_path) > 300:  # 5 minutes
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Error removing old file {filename}: {e}")

    def run(self):
        """Enhanced main detection loop"""
        try:
            frame_count = 0
            fps_start_time = time.time()
            fps = 0  # Initialize fps variable
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Calculate FPS
                if current_time - fps_start_time >= 1:
                    fps = frame_count / (current_time - fps_start_time)
                    frame_count = 0
                    fps_start_time = current_time
                
                frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
                frame = cv2.flip(frame, 1)
                
                processed_frame = self.preprocess_frame(frame)
                results = self.hands.process(processed_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        preprocessed_image = self.preprocess_landmarks(landmarks)
                        self.frame_sequence.append(preprocessed_image)
                        self.draw_landmarks(frame, landmarks, hand_landmarks)
                
                if len(self.frame_sequence) == SEQUENCE_LENGTH:
                    input_data = np.expand_dims(np.array(self.frame_sequence), axis=0)
                    predicted_label, confidence = self.get_ensemble_prediction(input_data)
                    self.display_prediction(frame, predicted_label, confidence, fps)
                    self.handle_text_to_speech(predicted_label, confidence)
                
                cv2.imshow("Real-time ASL Detection", frame)
                
                if frame_count % 300 == 0:  # Every 300 frames
                    self.cleanup_old_files()
                    gc.collect()
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Enhanced cleanup with error handling"""
        try:
            self.cap.release()
            cv2.destroyAllWindows()
            self.audio_executor.shutdown(wait=True)
            pygame.mixer.quit()
            pygame.quit()
            self.cleanup_old_files()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

if __name__ == "__main__":
    detector = ASLDetector()
    detector.run()