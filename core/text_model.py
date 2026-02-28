from transformers import pipeline

class TextEmotionAnalyzer:
    def __init__(self):
        print("💬 Booting up Text NLP Engine...")
        # We use a specialized emotion model based on DistilRoBERTa
        self.classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
        
        # Map Hugging Face's lowercase labels to our System's Capitalized labels
        self.emotion_map = {
            'anger': 'Angry',
            'disgust': 'Disgust',
            'fear': 'Fear',
            'joy': 'Happy',
            'neutral': 'Neutral',
            'sadness': 'Sad',
            'surprise': 'Surprise'
        }
        print("✅ Text Model Ready for Inference!")

    def predict_emotion(self, text):
        """End-to-end prediction from raw text to emotion probabilities."""
        if not text or len(text.strip()) == 0:
            empty_probs = {e: 0.0 for e in self.emotion_map.values()}
            return empty_probs, "No Text Provided"

        # 1. Pass text through the Transformer model
        results = self.classifier(text)[0]
        
        # 2. Format the output probabilities to match our standard dictionary
        emotion_probs = {}
        for res in results:
            mapped_label = self.emotion_map[res['label']]
            emotion_probs[mapped_label] = float(res['score'])
            
        # 3. Find the highest probability
        top_emotion = max(emotion_probs, key=emotion_probs.get)
        
        return emotion_probs, top_emotion