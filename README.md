# Multi-Modal-Emotion-Recognition
Multi-modal emotion detection API fusing computer vision, acoustic analysis, and natural language processing.artificial-intelligence deep-learning computer-vision nlp fastapi streamlit emotion-recognition python
# 🧠 Multi-Modal Emotion Detection System

An advanced, research-grade Artificial Intelligence architecture that detects human emotion by simultaneously analyzing facial expressions, vocal acoustics, and spoken text. 

Built with a specialized **Weighted Late Fusion Strategy**, this system merges the predictive capabilities of three independent deep learning models into a centralized FastAPI backend, visualized through a highly interactive 4-tab Streamlit dashboard.

## 🏗️ System Architecture

1. **👁️ Vision Modality (Facial Expressions)**
   * **Model:** Fine-tuned `MobileNetV2` + `MTCNN` (Face Detection)
   * **Dataset:** FER-2013
   * **Function:** Extracts cropped faces from webcam feeds/images and predicts spatial emotional features.

2. **🔊 Acoustic Modality (Vocal Intonation)**
   * **Model:** Custom `1D Convolutional Neural Network (CNN)`
   * **Dataset:** RAVDESS
   * **Function:** Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from raw `.wav` audio to detect micro-tremors and pitch variations.

3. **💬 Textual Modality (NLP Sentiment)**
   * **Model:** `DistilRoBERTa` (Hugging Face Transformers)
   * **Function:** Analyzes the linguistic semantics of the spoken transcript.

4. **🧩 The Fusion Engine**
   * Uses a customized **Weighted Late Fusion algorithm** to mathematically align disparate label structures (e.g., merging "Calm" and "Neutral") and calculate the final probability distribution across all combined modalities.

## 💻 Tech Stack
* **Deep Learning:** TensorFlow/Keras, PyTorch, Hugging Face Transformers
* **Signal/Image Processing:** OpenCV, Librosa, facenet-pytorch
* **Backend API:** FastAPI, Uvicorn
* **Frontend UI:** Streamlit, Pandas

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install torch torchvision
2. Train the Local Models (Optional)
Note: Datasets (FER2013 & RAVDESS) must be downloaded locally to data/ before training.

Bash
python train_vision.py
python train_audio.py
3. Start the FastAPI Backend (Inference Engine)
Bash
python -m uvicorn api.main:app --reload
4. Start the Streamlit Frontend (Dashboard)
Bash
python -m streamlit run frontend/app.py
👨‍💻 Team
Lead Architect: [Your Name]

Collaborator/Partner: Sher Ali Saleem


### 🚀 Step 4: Push to GitHub
Now that your folder is completely organized and secured by the `.gitignore`, you are ready to push. 

Open your terminal (inside the `multi_modal_emotion_ai` folder) and run these exact commands one by one:

```powershell
# 1. Initialize the repository
git init

# 2. Add all your code (the .gitignore will automatically block the heavy files)
git add .

# 3. Create your first commit
git commit -m "Initial commit: Multi-Modal Emotion AI System Architecture"

# 4. Link it to your GitHub (Replace the URL with your actual empty GitHub repo link)
git remote add origin https://github.com/YOUR_USERNAME/multi-modal-emotion-ai.git

# 5. Push the code to the main branch
git branch -M main
git push -u origin main
