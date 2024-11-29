from collections import deque
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Only show errors
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

class ASLProcessor:
    def __init__(self, model_type='cnn'):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        
        # Set model parameters
        self.model_type = model_type
        self.img_size = 224
        self.sequence_length = 30
        self.frame_sequence = deque(maxlen=self.sequence_length)
        
        try:
            # Clear any existing TF session
            tf.keras.backend.clear_session()
            
            # Get base directory for models and data
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            if model_type == 'cnn':
                # Load CNN model using proper path joining
                model_path = os.path.join(base_dir, 'ml_models', 'asl_mobilenetv2_model.h5')
                encoder_path = os.path.join(base_dir, 'ml_data', 'labels', 'cnn_label_encoder.pkl')
                
                # Verify files exist
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"CNN model not found at {model_path}")
                if not os.path.exists(encoder_path):
                    raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
                
                # Load model
                self.model = tf.keras.models.load_model(model_path, compile=False)
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Load label encoder
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                    
            else:  # RNN
                # Load RNN model using proper path joining
                model_path = os.path.join(base_dir, 'ml_models', 'rnn_asl_model.h5')
                encoder_path = os.path.join(base_dir, 'ml_data', 'labels', 'rnn_label_encoder.pkl')
                
                # Verify files exist
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"RNN model not found at {model_path}")
                if not os.path.exists(encoder_path):
                    raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
                
                # Load model
                self.model = tf.keras.models.load_model(model_path, compile=False)
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Load label encoder
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            
        except Exception as e:
            print(f"Error initializing {model_type} model: {str(e)}")
            raise

        # Initialize drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
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
            preprocessed = self.preprocess_landmarks(landmarks)
            
            if self.model_type == 'rnn':
                # Add frame to sequence
                self.frame_sequence.append(preprocessed)
                
                # Only predict when we have enough frames
                if len(self.frame_sequence) == self.sequence_length:
                    try:
                        sequence = np.array(list(self.frame_sequence))
                        sequence = np.expand_dims(sequence, axis=0)
                        predictions = self.model.predict(sequence, verbose=0)
                        predicted_class_index = np.argmax(predictions[0])
                        confidence = float(predictions[0][predicted_class_index])
                        predicted_label = self.label_encoder.inverse_transform([predicted_class_index])[0]
                        
                        return {
                            'prediction': predicted_label,
                            'confidence': confidence
                        }
                    except Exception as e:
                        print(f"Error during RNN prediction: {str(e)}")
                        return {
                            'prediction': 'Error',
                            'confidence': 0.0
                        }
                
                return {
                    'prediction': 'Collecting frames...',
                    'confidence': 0.0
                }
                
            else:  # CNN processing
                try:
                    predictions = self.model.predict(preprocessed, verbose=0)
                    predicted_class_index = np.argmax(predictions[0])
                    confidence = float(predictions[0][predicted_class_index])
                    predicted_label = self.label_encoder.inverse_transform([predicted_class_index])[0]
                    
                    return {
                        'prediction': predicted_label,
                        'confidence': confidence
                    }
                except Exception as e:
                    print(f"Error during CNN prediction: {str(e)}")
                    return {
                        'prediction': 'Error',
                        'confidence': 0.0
                    }
        
        return {
            'prediction': 'No hand detected',
            'confidence': 0.0
        }