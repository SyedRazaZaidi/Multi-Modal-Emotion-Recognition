from fastapi import FastAPI, UploadFile, File, Form
from core.fusion_layer import EmotionFuser  # <--- NEW IMPORT
import uvicorn
import cv2
import numpy as np
import shutil
import os

from core.vision_model import VisionEmotionAnalyzer
from core.audio_model import AudioEmotionAnalyzer # <--- NEW IMPORT
from core.text_model import TextEmotionAnalyzer 

app = FastAPI(title="Multi-Modal Emotion AI", version="1.0")

vision_analyzer = VisionEmotionAnalyzer(model_weights_path="data/weights/vision_mobilenetv2_best.h5")
audio_analyzer = AudioEmotionAnalyzer(model_weights_path="data/weights/audio_cnn_best.h5")
text_analyzer = TextEmotionAnalyzer()
fuser = EmotionFuser(vision_weight=0.4, audio_weight=0.4, text_weight=0.2)

# Temporary folder to save audio uploads before processing
os.makedirs("temp", exist_ok=True)

@app.get("/")
def health_check():
    return {"status": "System Online", "message": "Multi-Modal AI is ready for input. 🔥"}

@app.post("/predict_vision")
async def predict_vision(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image file provided."}

    emotion_probs, top_emotion = vision_analyzer.predict_emotion(frame)

    return {
        "modality": "vision",
        "predicted_emotion": top_emotion,
        "probabilities": emotion_probs
    }

# --- NEW AUDIO ENDPOINT ---
@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)):
    # Save the uploaded file temporarily so librosa can read it
    temp_file_path = f"temp/{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Pass to the audio model
    emotion_probs, top_emotion = audio_analyzer.predict_emotion(temp_file_path)
    
    # Clean up the temp file
    os.remove(temp_file_path)
    
    return {
        "modality": "audio",
        "predicted_emotion": top_emotion,
        "probabilities": emotion_probs
    }
# --- NEW TEXT ENDPOINT ---
@app.post("/predict_text")
async def predict_text(text: str = Form(...)):
    emotion_probs, top_emotion = text_analyzer.predict_emotion(text)
    
    return {
        "modality": "text",
        "predicted_emotion": top_emotion,
        "probabilities": emotion_probs
    }
@app.post("/predict_combined")
async def predict_combined(
    image: UploadFile = File(...), 
    audio: UploadFile = File(...), 
    transcript: str = Form(...)
):
    # 1. Process Vision
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    vision_probs, _ = vision_analyzer.predict_emotion(frame)

    # 2. Process Audio
    temp_file_path = f"temp/{audio.filename}"
    with open(temp_file_path, "wb") as buffer:
        import shutil
        shutil.copyfileobj(audio.file, buffer)
    audio_probs, _ = audio_analyzer.predict_emotion(temp_file_path)
    os.remove(temp_file_path)

    # 3. Process Text
    text_probs, _ = text_analyzer.predict_emotion(transcript)

    # 4. FUSE THE MODALITIES
    final_emotion, confidence, fused_probs = fuser.fuse_predictions(
        vision_probs, audio_probs, text_probs
    )

    return {
        "final_prediction": {
            "emotion": final_emotion,
            "confidence": confidence,
            "fused_probabilities": fused_probs
        },
        "breakdown": {
            "vision_probabilities": vision_probs,
            "audio_probabilities": audio_probs,
            "text_probabilities": text_probs
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)