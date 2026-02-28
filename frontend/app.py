import streamlit as st
import requests
import pandas as pd

# Define the base FastAPI URL
BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Multi-Modal Emotion AI", page_icon="🧠", layout="wide")

# --- Helper Function for Charts ---
def plot_probabilities(probs_dict):
    """Converts a probability dictionary into a beautiful Streamlit bar chart."""
    df_probs = pd.DataFrame(list(probs_dict.items()), columns=['Emotion', 'Probability'])
    df_probs['Probability'] = df_probs['Probability'] * 100 # Convert to percentage
    st.bar_chart(df_probs.set_index('Emotion'))

# --- Header ---
st.title("🧠 Multi-Modal Emotion Detection System")
st.markdown("""
**Research-Level Architecture:** Test individual modalities (Vision, Audio, Text) in isolation, or combine them using a Weighted Late Fusion strategy for robust emotion prediction.
""")
st.divider()

# --- Create 4 Interactive Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["👁️ Vision Only", "🔊 Audio Only", "💬 Text Only", "🧩 Multi-Modal Fusion"])

# ==========================================
# TAB 1: VISION ONLY
# ==========================================
with tab1:
    st.subheader("👁️ Vision Analysis (MobileNetV2)")
    col_v1, col_v2 = st.columns([1, 2])
    
    with col_v1:
        vision_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"], key="v_img")
        if vision_file:
            st.image(vision_file, caption="Input Feed", use_column_width=True)
            
            if st.button("Analyze Face", use_container_width=True):
                with st.spinner("Processing spatial features..."):
                    files = {"file": (vision_file.name, vision_file.getvalue(), vision_file.type)}
                    res = requests.post(f"{BASE_URL}/predict_vision", files=files).json()
                    
                    with col_v2:
                        st.success("Vision Analysis Complete!")
                        st.metric(label="Predicted Emotion", value=res["predicted_emotion"])
                        plot_probabilities(res["probabilities"])

# ==========================================
# TAB 2: AUDIO ONLY
# ==========================================
with tab2:
    st.subheader("🔊 Audio Analysis (1D CNN)")
    col_a1, col_a2 = st.columns([1, 2])
    
    with col_a1:
        audio_file = st.file_uploader("Upload Voice Clip", type=["wav"], key="a_wav")
        if audio_file:
            st.audio(audio_file, format='audio/wav')
            
            if st.button("Analyze Audio", use_container_width=True):
                with st.spinner("Extracting MFCCs and analyzing acoustics..."):
                    files = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                    res = requests.post(f"{BASE_URL}/predict_audio", files=files).json()
                    
                    with col_a2:
                        st.success("Audio Analysis Complete!")
                        st.metric(label="Predicted Emotion", value=res["predicted_emotion"])
                        plot_probabilities(res["probabilities"])

# ==========================================
# TAB 3: TEXT ONLY
# ==========================================
with tab3:
    st.subheader("💬 Text Analysis (DistilRoBERTa)")
    col_t1, col_t2 = st.columns([1, 2])
    
    with col_t1:
        transcript = st.text_area("Spoken Transcript", placeholder="Type what the person is saying here...", key="t_txt")
        if transcript:
            if st.button("Analyze Text", use_container_width=True):
                with st.spinner("Processing linguistic semantics..."):
                    data = {"text": transcript}
                    res = requests.post(f"{BASE_URL}/predict_text", data=data).json()
                    
                    with col_t2:
                        st.success("Text Analysis Complete!")
                        st.metric(label="Predicted Emotion", value=res["predicted_emotion"])
                        plot_probabilities(res["probabilities"])

# ==========================================
# TAB 4: THE FUSION LAYER
# ==========================================
with tab4:
    st.subheader("🧩 Multi-Modal Fusion Engine")
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        f_img = st.file_uploader("1. Upload Face", type=["jpg", "jpeg", "png"], key="f_img")
    with col_f2:
        f_aud = st.file_uploader("2. Upload Audio", type=["wav"], key="f_aud")
    with col_f3:
        f_txt = st.text_area("3. Transcript", placeholder="Type transcript...", key="f_txt")

    st.divider()

    if st.button("🔥 Run Multi-Modal Fusion Analysis", use_container_width=True):
        if not f_img or not f_aud or not f_txt:
            st.warning("⚠️ Please provide all three modalities (Image, Audio, and Text) to run the fusion analysis.")
        else:
            with st.spinner("Processing embeddings & mathematically fusing vectors..."):
                try:
                    files = {
                        "image": (f_img.name, f_img.getvalue(), f_img.type),
                        "audio": (f_aud.name, f_aud.getvalue(), f_aud.type)
                    }
                    data = {"transcript": f_txt}
                    
                    response = requests.post(f"{BASE_URL}/predict_combined", files=files, data=data)
                    response.raise_for_status() 
                    
                    result = response.json()
                    final_data = result["final_prediction"]
                    
                    st.success("Fusion Complete!")
                    
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric(label="🏆 Final Predicted Emotion", value=final_data["emotion"])
                    with m2:
                        st.metric(label="🎯 System Confidence", value=f"{final_data['confidence'] * 100:.2f}%")
                    
                    st.subheader("📊 Fused Probability Distribution")
                    plot_probabilities(final_data["fused_probabilities"])
                    
                except Exception as e:
                    st.error(f"Error connecting to backend API: {e}")