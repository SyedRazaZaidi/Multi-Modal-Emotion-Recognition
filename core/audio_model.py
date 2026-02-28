import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

class AudioEmotionAnalyzer:
    def __init__(self, model_weights_path=None):
        # Must match the exact order of the training labels (0 to 7)
        self.emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
        self.is_loaded = model_weights_path is not None
        
        self.model = None
        if self.is_loaded:
            print(f"🔊 Loading Audio Model Weights from {model_weights_path}...")
            self.model = self._build_model()
            self.model.load_weights(model_weights_path)
            print("✅ Audio Model Ready for Inference!")
        else:
            print("⚠️ Audio Model initialized WITHOUT weights (Mock Mode).")

    def _build_model(self):
        """Rebuilds the exact 1D CNN architecture we used for training."""
        model = Sequential([
            Conv1D(256, 5, padding='same', activation='relu', input_shape=(40, 1)),
            MaxPooling1D(pool_size=5, strides=2, padding='same'),
            Conv1D(128, 5, padding='same', activation='relu'),
            MaxPooling1D(pool_size=5, strides=2, padding='same'),
            Dropout(0.2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(8, activation='softmax')
        ])
        return model

    def extract_features(self, file_path):
        """Extracts MFCC features from the audio file."""
        try:
            X, sample_rate = librosa.load(file_path)
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            return mfccs
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return None

    def predict_emotion(self, audio_file_path):
        """End-to-end prediction from raw audio to emotion probabilities."""
        if not self.is_loaded:
            mock_probs = {e: 0.125 for e in self.emotions}
            return mock_probs, "Neutral"

        # 1. Extract the sound features
        features = self.extract_features(audio_file_path)
        if features is None:
            empty_probs = {e: 0.0 for e in self.emotions}
            return empty_probs, "Processing Error"

        # 2. Reshape for the 1D CNN: (1 sample, 40 timesteps, 1 feature)
        features_reshaped = np.expand_dims(features, axis=0)
        features_reshaped = np.expand_dims(features_reshaped, axis=2)

        # 3. Predict the emotion
        preds = self.model.predict(features_reshaped, verbose=0)[0]
        
        # 4. Format Output
        emotion_probs = {self.emotions[i]: float(preds[i]) for i in range(len(self.emotions))}
        top_emotion = max(emotion_probs, key=emotion_probs.get)
        
        return emotion_probs, top_emotion