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
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ASLProcessor, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_type='cnn'):
        if hasattr(self, 'initialized') and self.initialized:
            return

        # Initialize MediaPipe and basic parameters first
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        
        self.model_type = model_type
        self.img_size = 224  # Match the expected input size
        self.sequence_length = 30
        self.frame_sequence = deque(maxlen=self.sequence_length)
        self.prediction_history = deque(maxlen=5)
        
        try:
            # Clear any existing TF session
            tf.keras.backend.clear_session()
            
            # Get absolute paths
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Set paths based on model type
            if model_type == 'rnn':
                model_path =  "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_WEB/ml_models/rnn_asl_model.h5"
                encoder_path = "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_WEB/ml_data/labels/rnn_label_encoder.pkl"
            else:
                model_path = os.path.join(base_dir, 'ml_models', 'asl_mobilenetv2_model.h5') 
                encoder_path = os.path.join(base_dir, 'ml_data', 'labels', 'cnn_label_encoder.pkl')

            # Verify files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Label encoder not found at {encoder_path}")

            # Load and compile model with error handling
            try:
                print(f"Loading model from {model_path}")
                self.model = tf.keras.models.load_model(model_path, compile=False)
                self.model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy']
                )
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                self.model = None
                raise

            # Load label encoder with error handling 
            try:
                print(f"Loading label encoder from {encoder_path}")
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print("Label encoder loaded successfully")
            except Exception as e:
                print(f"Error loading encoder: {str(e)}")
                self.label_encoder = None
                raise

            # Verify model and label encoder are loaded
            if self.model is not None:
                print("Model is initialized")
            if self.label_encoder is not None:
                print("Label encoder is initialized")

        except Exception as e:
            print(f"Error initializing {model_type} model: {str(e)}")
            self.model = None 
            self.label_encoder = None

        self.initialized = True
        
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
        try:
            # Verify model and encoder are loaded
            print(f"Model: {self.model}")
            print(f"Label Encoder: {self.label_encoder}")
            if self.model is None or self.label_encoder is None:
                print("Model or label encoder not initialized")
                return {
                    'prediction': 'Model not initialized',
                    'confidence': 0.0
                }

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if not results.multi_hand_landmarks:
                return {
                    'prediction': 'No hand detected',
                    'confidence': 0.0
                }

            landmarks = self.extract_landmarks(results.multi_hand_landmarks[0])
            preprocessed = self.preprocess_landmarks(landmarks)

            if self.model_type == 'rnn':
                # Add frame to sequence
                self.frame_sequence.append(preprocessed)
                print(f"Frame sequence length: {len(self.frame_sequence)}")

                # Only predict when we have enough frames
                if len(self.frame_sequence) == self.sequence_length:
                    try:
                        sequence = np.array(list(self.frame_sequence))
                        print(f"Sequence shape before expand_dims: {sequence.shape}")
                        sequence = np.expand_dims(sequence, axis=0)
                        print(f"Sequence shape after expand_dims: {sequence.shape}")
                        print("Making RNN prediction")
                        predictions = self.model.predict(sequence, verbose=0)
                        print(f"RNN predictions: {predictions}")

                        if len(predictions) == 0 or len(predictions[0]) == 0:
                            return {
                                'prediction': 'Invalid prediction',
                                'confidence': 0.0
                            }

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
                    print(f"Preprocessed shape before expand_dims: {preprocessed.shape}")
                    preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension
                    print(f"Preprocessed shape after expand_dims: {preprocessed.shape}")
                    print("Making CNN prediction")
                    predictions = self.model.predict(preprocessed, verbose=0)
                    print(f"CNN predictions: {predictions}")

                    if len(predictions) == 0 or len(predictions[0]) == 0:
                        return {
                            'prediction': 'Invalid prediction',
                            'confidence': 0.0
                        }

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
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return {
                'prediction': 'Error',
                'confidence': 0.0
            }