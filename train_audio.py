import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

print("🚀 Booting up Audio Training Engine...")

DATA_PATH = "data/ravdess/"
WEIGHTS_DIR = "data/weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# RAVDESS emotion mapping (1 to 8)
emotions = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def extract_feature(file_name):
    """Extracts MFCC features from the audio file."""
    try:
        # FIX: Removed res_type='kaiser_fast' to prevent the librosa/resampy crash
        X, sample_rate = librosa.load(file_name)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        # FIX: Upgraded error logging to show the exact Python exception
        print(f"Error parsing {file_name}: {e}")
        return None

def load_data():
    """Scans the RAVDESS dataset and prepares the X (features) and y (labels)."""
    x, y = [], []
    print("📂 Extracting Audio Features (This takes a moment)...")
    
    # Loop through all actor folders
    for actor_dir in os.listdir(DATA_PATH):
        actor_path = os.path.join(DATA_PATH, actor_dir)
        if not os.path.isdir(actor_path):
            continue
            
        # Loop through all audio files per actor
        for file in os.listdir(actor_path):
            if not file.endswith('.wav'):
                continue
                
            file_path = os.path.join(actor_path, file)
            # The emotion is the 3rd part of the RAVDESS filename
            emotion_code = file.split('-')[2] 
            
            features = extract_feature(file_path)
            if features is not None:
                x.append(features)
                y.append(int(emotion_code) - 1) # Convert '01'-'08' to 0-7 for the neural network

    return np.array(x), np.array(y)

# 1. Load and Prepare Data
X, y = load_data()

# Safety check to ensure data loaded correctly before reshaping
if len(X) == 0:
    raise ValueError("CRITICAL ERROR: No audio features were extracted. Check your dataset path and librosa installation.")

# Reshape X for 1D CNN: (samples, timesteps, features)
X = np.expand_dims(X, axis=2) 

# Convert labels to categorical (one-hot encoding)
y = tf.keras.utils.to_categorical(y, num_classes=8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build the 1D CNN Architecture
print("🏗️ Building 1D CNN Architecture for Audio...")
model = Sequential([
    Conv1D(256, 5, padding='same', activation='relu', input_shape=(40, 1)),
    MaxPooling1D(pool_size=5, strides=2, padding='same'),
    Conv1D(128, 5, padding='same', activation='relu'),
    MaxPooling1D(pool_size=5, strides=2, padding='same'),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(8, activation='softmax') # 8 target emotions
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Train the Model
checkpoint = ModelCheckpoint(f"{WEIGHTS_DIR}/audio_cnn_best.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

print("🔥 Starting Audio Training Phase...")
history = model.fit(
    X_train, y_train, 
    batch_size=32, 
    epochs=50, # Audio trains much faster than images!
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)

print("✅ Audio Training Complete. Weights saved!")