class EmotionFuser:
    def __init__(self, vision_weight=0.4, audio_weight=0.4, text_weight=0.2):
        self.weights = {
            'vision': vision_weight,
            'audio': audio_weight,
            'text': text_weight
        }
        self.base_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        print("🧩 Fusion Layer Initialized (Weights: V=0.4, A=0.4, T=0.2)")

    def align_audio_emotions(self, audio_probs):
        """Maps RAVDESS's 8 emotions to our standard 7 Vision/Text emotions."""
        aligned = {e: 0.0 for e in self.base_emotions}
        
        for k, v in audio_probs.items():
            if k == 'Fearful': 
                aligned['Fear'] += v
            elif k == 'Surprised': 
                aligned['Surprise'] += v
            elif k == 'Calm': 
                aligned['Neutral'] += v
            elif k in aligned: 
                aligned[k] += v
                
        return aligned

    def fuse_predictions(self, vision_probs, audio_probs, text_probs):
        """
        Calculates the weighted sum of probability vectors.
        P_final = (w_v * P_v) + (w_a * P_a) + (w_t * P_t)
        """
        # Align audio first
        audio_aligned = self.align_audio_emotions(audio_probs)
        fused_probs = {emotion: 0.0 for emotion in self.base_emotions}

        # Apply Weights
        for emotion in self.base_emotions:
            v_val = vision_probs.get(emotion, 0.0) * self.weights['vision']
            a_val = audio_aligned.get(emotion, 0.0) * self.weights['audio']
            t_val = text_probs.get(emotion, 0.0) * self.weights['text']
            
            fused_probs[emotion] = v_val + a_val + t_val

        # Normalize to ensure probabilities sum perfectly to 1.0 (100%)
        total = sum(fused_probs.values())
        if total > 0:
            fused_probs = {k: v / total for k, v in fused_probs.items()}

        final_emotion = max(fused_probs, key=fused_probs.get)
        confidence = fused_probs[final_emotion]

        return final_emotion, confidence, fused_probs