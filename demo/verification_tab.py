#!/usr/bin/env python3
"""Speaker Verification Tab for Unified Demo"""

import gradio as gr
import torch
import numpy as np
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'speaker_verification'))

from speaker_verification.config.config import Config as VerificationConfig
from speaker_verification.models.cam_plus_plus import CAMPlusPlus
from speaker_verification.models.frontend import AudioFrontend
from speaker_verification.data.preprocessing import AudioPreprocessor


class SpeakerVerificationEngine:
    """Engine for speaker verification inference."""
    
    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = VerificationConfig()
        
        # Initialize model
        self.model = self._load_model(checkpoint_path)
        self.preprocessor = AudioPreprocessor(self.config)
        self.frontend = AudioFrontend(self.config).to(self.device)
        
        # Threshold for verification
        self.threshold = 0.5
    
    def _load_model(self, checkpoint_path):
        """Load the verification model."""
        model = CAMPlusPlus(
            input_dim=self.config.n_mels,
            embedding_dim=self.config.embedding_dim,
            num_classes=self.config.num_classes
        ).to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print("Warning: No checkpoint loaded, using random weights")
        
        model.eval()
        return model
    
    def extract_embedding(self, audio_path):
        """Extract speaker embedding from audio file."""
        # Load and preprocess audio
        waveform = self.preprocessor.load_audio(audio_path)
        waveform = waveform.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.frontend(waveform)
            embedding = self.model.extract_embedding(features)
        
        return embedding.cpu().numpy()
    
    def compute_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between two embeddings."""
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()
        
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return float(similarity)
    
    def verify(self, audio_path1, audio_path2):
        """Verify if two audio samples are from the same speaker."""
        embedding1 = self.extract_embedding(audio_path1)
        embedding2 = self.extract_embedding(audio_path2)
        
        similarity = self.compute_similarity(embedding1, embedding2)
        is_same_speaker = similarity >= self.threshold
        
        return similarity, is_same_speaker


# Global engine instance
verification_engine = None


def get_verification_engine():
    """Get or create verification engine."""
    global verification_engine
    if verification_engine is None:
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'speaker_verification', 'checkpoints', 'best_model.pt'
        )
        verification_engine = SpeakerVerificationEngine(checkpoint_path)
    return verification_engine


def verify_speakers(audio1, audio2, threshold):
    """Gradio function for speaker verification."""
    if audio1 is None or audio2 is None:
        return "⚠️ Please upload both audio files", "", ""
    
    try:
        engine = get_verification_engine()
        engine.threshold = threshold
        
        similarity, is_same_speaker = engine.verify(audio1, audio2)
        
        # Format results
        similarity_text = f"**Similarity Score:** {similarity:.4f}"
        
        if is_same_speaker:
            result = f"""
            <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: #155724;">✅ VERIFIED - Same Speaker</h2>
                <p>The two audio samples are from the <strong>same speaker</strong></p>
                <p>Confidence: {similarity:.2%}</p>
            </div>
            """
        else:
            result = f"""
            <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: #721c24;">❌ NOT VERIFIED - Different Speakers</h2>
                <p>The two audio samples are from <strong>different speakers</strong></p>
                <p>Similarity: {similarity:.2%}</p>
            </div>
            """
        
        details = f"""
        ### Detailed Results
        - **Similarity Score:** {similarity:.4f}
        - **Threshold:** {threshold:.4f}
        - **Decision:** {"Same Speaker" if is_same_speaker else "Different Speakers"}
        - **Device:** {engine.device}
        """
        
        return result, similarity_text, details
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""


def create_verification_tab():
    """Create the speaker verification tab interface."""
    
    gr.Markdown("""
    ## 🔐 Speaker Verification
    
    Upload two audio files to verify if they belong to the same speaker.
    
    **How it works:**
    1. Upload an enrollment audio (reference speaker)
    2. Upload a test audio (speaker to verify)
    3. Adjust the threshold if needed
    4. Click "Verify" to compare
    """)
    
    with gr.Row():
        with gr.Column():
            audio1 = gr.Audio(
                label="📎 Enrollment Audio (Reference)",
                type="filepath",
                sources=["upload", "microphone"]
            )
        
        with gr.Column():
            audio2 = gr.Audio(
                label="🎤 Test Audio (To Verify)",
                type="filepath",
                sources=["upload", "microphone"]
            )
    
    with gr.Row():
        threshold = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.5,
            step=0.01,
            label="Verification Threshold",
            info="Higher threshold = stricter verification"
        )
    
    verify_btn = gr.Button("🔍 Verify Speakers", variant="primary", size="lg")
    
    # Results
    result_html = gr.HTML(label="Result")
    similarity_text = gr.Markdown(label="Similarity Score")
    details_text = gr.Markdown(label="Details")
    
    # Connect function
    verify_btn.click(
        fn=verify_speakers,
        inputs=[audio1, audio2, threshold],
        outputs=[result_html, similarity_text, details_text]
    )
    
    # Examples
    gr.Markdown("### 📚 Example Usage")
    gr.Markdown("""
    - **Same speaker verification**: Upload two recordings of the same person
    - **Different speaker verification**: Upload recordings from different people
    - **Threshold tuning**: Lower threshold (0.3-0.4) for lenient matching, higher (0.6-0.7) for strict matching
    """)