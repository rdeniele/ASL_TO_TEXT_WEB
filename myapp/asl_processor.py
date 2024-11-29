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
import pickle
import time

class ASLProcessorBase:
    def __init__(self):
        self.img_size = 224  # Match the expected input size
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
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
            return self.predict(preprocessed)
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return {
                'prediction': 'Error',
                'confidence': 0.0
            }

class ASLProcessorCNN(ASLProcessorBase):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml_models', 'asl_mobilenetv2_model.h5')
        self.encoder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml_data', 'labels', 'cnn_label_encoder.pkl')
        self.load_model_and_encoder()

    def load_model_and_encoder(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            if not os.path.exists(self.encoder_path):
                raise FileNotFoundError(f"Label encoder not found at {self.encoder_path}")

            print(f"Loading model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy']
            )
            print("Model loaded successfully")

            print(f"Loading label encoder from {self.encoder_path}")
            with open(self.encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("Label encoder loaded successfully")
        except Exception as e:
            print(f"Error initializing CNN model: {str(e)}")
            self.model = None
            self.label_encoder = None

    def predict(self, preprocessed):
        try:
            print(f"Preprocessed shape before expand_dims: {preprocessed.shape}")
            preprocessed = np.squeeze(preprocessed, axis=0)
            print(f"Preprocessed shape after squeeze: {preprocessed.shape}")
            preprocessed = np.expand_dims(preprocessed, axis=0)
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

class ASLProcessorRNN(ASLProcessorBase):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ASLProcessorRNN, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized') and self.initialized:
            return

        super().__init__()
        self.sequence_length = 5  # Set the sequence length to 5
        self.frame_sequence = deque(maxlen=self.sequence_length)
        self.model_path = "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_WEB/ml_models/rnn_asl_model.h5"
        self.encoder_path = "C:/Users/ronde/PROJECTS/ASL_TO_TEXT_WEB/ml_data/labels/rnn_label_encoder.pkl"
        self.model = None
        self.label_encoder = None
        self.load_model_and_encoder()
        self.initialized = True
        self.paused = False  # Add a paused flag

    def load_model_and_encoder(self):
        if self.model is None or self.label_encoder is None:
            try:
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Model not found at {self.model_path}")
                if not os.path.exists(self.encoder_path):
                    raise FileNotFoundError(f"Label encoder not found at {self.encoder_path}")

                print(f"Loading model from {self.model_path}")
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                self.model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy']
                )
                print("Model loaded successfully")

                print(f"Loading label encoder from {self.encoder_path}")
                with open(self.encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print("Label encoder loaded successfully")
            except Exception as e:
                print(f"Error initializing RNN model: {str(e)}")
                self.model = None
                self.label_encoder = None

    def predict(self, preprocessed):
        try:
            if self.paused:
                return {
                    'prediction': 'Paused',
                    'confidence': 0.0
                }

            self.frame_sequence.append(preprocessed)
            print(f"Frame sequence length: {len(self.frame_sequence)}")

            if len(self.frame_sequence) == self.sequence_length:
                try:
                    sequence = np.array(list(self.frame_sequence))
                    print(f"Sequence shape before squeeze: {sequence.shape}")
                    sequence = np.squeeze(sequence, axis=1)  # Remove the extra dimension
                    print(f"Sequence shape after squeeze: {sequence.shape}")
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

                    # Clear the frame sequence after prediction
                    self.frame_sequence.clear()
                    self.paused = True  # Pause after prediction

                    # Flash the prediction to the web
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
        except Exception as e:
            print(f"Error during RNN prediction: {str(e)}")
            return {
                'prediction': 'Error',
                'confidence': 0.0
            }

    def resume(self):
        self.paused = False  

# Define ASLProcessor as a factory function to create instances of ASLProcessorCNN or ASLProcessorRNN
def ASLProcessor(model_type='cnn'):
    if model_type == 'cnn':
        return ASLProcessorCNN()
    elif model_type == 'rnn':
        return ASLProcessorRNN()
    else:
        raise ValueError(f"Unknown model type: {model_type}")