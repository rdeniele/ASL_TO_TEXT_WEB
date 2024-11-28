import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import logging
from tensorflow.keras.models import load_model
import base64

class ASLProcessor:
    def __init__(self):
        self.cnn_model = None
        self.rnn_model = None
        self.current_model = 'cnn'
        self.label_encoder = None
        self.img_size = 224
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.initialize_models()

    def initialize_models(self):
        try:
            base_path = 'ml_models'
            self.cnn_model = load_model(f'{base_path}/asl_mobilenetv2_model.h5')
            self.rnn_model = load_model(f'{base_path}/rnn_asl_model.h5')
            
            with open('ml_data/labels/cnn_label_encoder.pkl', 'rb') as f:
                self.cnn_label_encoder = pickle.load(f)
                
            with open('ml_data/labels/rnn_label_encoder.pkl', 'rb') as f:
                self.rnn_label_encoder = pickle.load(f)
                
            self.label_encoder = self.cnn_label_encoder
            logging.info("Models initialized successfully")
            
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")

    def set_model(self, model_type):
        if model_type not in ['cnn', 'rnn']:
            logging.error(f"Invalid model type: {model_type}")
            return False

        try:
            self.current_model = model_type
            self.label_encoder = self.cnn_label_encoder if model_type == 'cnn' else self.rnn_label_encoder
            logging.info(f"Switched to {model_type.upper()} model")
            return True
        except Exception as e:
            logging.error(f"Error switching models: {str(e)}")
            return False

    def preprocess_image(self, image_data):
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (self.img_size, self.img_size))
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                landmarks = self.extract_landmarks(results.multi_hand_landmarks[0])
                return self.preprocess_landmarks(landmarks)

            if self.current_model == 'rnn':
                return np.zeros((1, 30, self.img_size, self.img_size, 3))
            else:
                return np.zeros((1, self.img_size, self.img_size, 3))

        except Exception as e:
            logging.error(f"Error preprocessing image: {str(e)}")
            raise

    def process_image(self, image_data):
        if image_data is None:
            return {'prediction': 'unknown', 'confidence': 0.0}

        try:
            input_data = self.preprocess_image(image_data)
            model = self.cnn_model if self.current_model == 'cnn' else self.rnn_model
            predictions = model.predict(input_data, verbose=0)[0]
            predicted_class_index = np.argmax(predictions)
            confidence = float(predictions[predicted_class_index])

            try:
                predicted_label = self.label_encoder.inverse_transform([predicted_class_index])[0]
            except IndexError:
                predicted_label = "unknown"

            return {
                'prediction': predicted_label,
                'confidence': confidence,
                'model_type': self.current_model
            }
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'model_type': self.current_model
            }

    def extract_landmarks(self, hand_landmarks):
        return [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

    def normalize_landmarks(self, landmarks):
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        return [[(x - min_x) / (max_x - min_x), (y - min_y) / (max_y - min_y), z] for x, y, z in landmarks]

    def preprocess_landmarks(self, landmarks):
        normalized_landmarks = self.normalize_landmarks(landmarks)
        landmarks_image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        if normalized_landmarks:
            for connection in self.mp_hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                x1, y1 = int(normalized_landmarks[start_idx][0] * (self.img_size - 1)), int(normalized_landmarks[start_idx][1] * (self.img_size - 1))
                x2, y2 = int(normalized_landmarks[end_idx][0] * (self.img_size - 1)), int(normalized_landmarks[end_idx][1] * (self.img_size - 1))
                cv2.line(landmarks_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

            for lm in normalized_landmarks:
                x, y = int(lm[0] * (self.img_size - 1)), int(lm[1] * (self.img_size - 1))
                cv2.circle(landmarks_image, (x, y), 5, (255, 0, 0), 3)

            landmarks_image = landmarks_image.astype(np.float32) / 255.0

            if self.current_model == 'rnn':
                landmarks_image = np.expand_dims(landmarks_image, axis=0)
                landmarks_image = np.expand_dims(landmarks_image, axis=1)
                landmarks_image = np.repeat(landmarks_image, 30, axis=1)
            else:
                landmarks_image = np.expand_dims(landmarks_image, axis=0)

        return landmarks_image

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            results = self.hands.process(rgb_frame)
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            prediction_result = {
                'prediction': 'No hand detected',
                'confidence': 0.0,
                'frame': frame
            }

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        self.mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )

                    landmarks = self.extract_landmarks(hand_landmarks)
                    processed_input = self.preprocess_landmarks(landmarks)
                    model = self.cnn_model if self.current_model == 'cnn' else self.rnn_model

                    if model is not None:
                        predictions = model.predict(processed_input, verbose=0)[0]
                        predicted_idx = np.argmax(predictions)
                        confidence = float(predictions[predicted_idx])

                        try:
                            predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
                        except:
                            predicted_label = "unknown"

                        cv2.putText(
                            frame,
                            f"{predicted_label} ({confidence:.2f})",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )

                        prediction_result.update({
                            'prediction': predicted_label,
                            'confidence': confidence,
                            'frame': frame
                        })

                    cv2.rectangle(frame, (100, 100), (324, 324), (0, 255, 0), 2)

            return prediction_result

        except Exception as e:
            logging.error(f"Error processing frame: {str(e)}")
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'frame': frame
            }
