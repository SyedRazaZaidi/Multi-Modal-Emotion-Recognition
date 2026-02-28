import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from facenet_pytorch import MTCNN

class VisionEmotionAnalyzer:
    def __init__(self, model_weights_path=None):
        # Alphabetical order matching the FER2013 training folders
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.is_loaded = model_weights_path is not None
        
        # 1. Initialize Face Detector (MTCNN is highly accurate for surveillance/webcams)
        self.face_detector = MTCNN(keep_all=False, device='cpu')
        
        # 2. Rebuild Model Architecture to load weights
        self.model = None
        if self.is_loaded:
            print(f"👁️ Loading Vision Model Weights from {model_weights_path}...")
            self.model = self._build_model()
            self.model.load_weights(model_weights_path)
            print("✅ Vision Model Ready for Inference!")
        else:
            print("⚠️ Vision Model initialized WITHOUT weights (Mock Mode).")

    def _build_model(self):
        """Rebuilds the exact MobileNetV2 architecture we used for training."""
        base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(7, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def extract_face(self, frame):
        """Detects and crops the largest face from a BGR image frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = self.face_detector.detect(rgb_frame)
        
        if boxes is not None:
            box = boxes[0].astype(int)
            # Ensure coordinates stay within image boundaries
            x1, y1 = max(0, box[0]), max(0, box[1])
            x2, y2 = min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
            cropped_face = rgb_frame[y1:y2, x1:x2]
            return cropped_face
        return None

    def predict_emotion(self, frame):
        """End-to-end prediction from raw frame to emotion probabilities."""
        if not self.is_loaded:
            return {'Angry': 0.1, 'Disgust': 0.05, 'Fear': 0.05, 'Happy': 0.6, 'Neutral': 0.05, 'Sad': 0.1, 'Surprise': 0.05}, "Happy"

        # 1. Extract the face using MTCNN
        face = self.extract_face(frame)
        if face is None or face.size == 0:
            empty_probs = {emotion: 0.0 for emotion in self.emotions}
            return empty_probs, "No Face Detected"

        # 2. Preprocess the face for MobileNetV2
        face_resized = cv2.resize(face, (224, 224))
        face_array = np.expand_dims(face_resized, axis=0)
        face_array = face_array / 255.0  # Crucial: Rescale exactly like we did during training!

        # 3. Predict the emotion
        preds = self.model.predict(face_array, verbose=0)[0]
        
        # 4. Format Output
        emotion_probs = {self.emotions[i]: float(preds[i]) for i in range(len(self.emotions))}
        top_emotion = max(emotion_probs, key=emotion_probs.get)
        
        return emotion_probs, top_emotion