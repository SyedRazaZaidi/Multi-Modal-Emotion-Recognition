import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print("🚀 Booting up Vision Training Engine...")

# Paths
TRAIN_DIR = 'data/fer2013/train'
VAL_DIR = 'data/fer2013/test'
WEIGHTS_DIR = 'data/weights'

os.makedirs(WEIGHTS_DIR, exist_ok=True)

# 1. Data Generators (Converting grayscale to 3-channel RGB for MobileNet)
print("📂 Loading Dataset...")
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(224, 224), batch_size=64, class_mode='categorical', color_mode='rgb'
)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(224, 224), batch_size=64, class_mode='categorical', color_mode='rgb'
)

# 2. Build the Transfer Learning Architecture
print("🏗️ Building MobileNetV2 Architecture...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # Freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation='softmax')(x) # 7 Emotions

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# 3. Callbacks & Training
callbacks = [
    ModelCheckpoint(f'{WEIGHTS_DIR}/vision_mobilenetv2_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
]

print("🔥 Starting Training Phase...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=callbacks
)

print("✅ Training Complete. Best weights saved to data/weights/")