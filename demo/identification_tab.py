#!/usr/bin/env python3
"""Speaker Identification Tab for Unified Demo"""

import gradio as gr
import torch
import numpy as np
import os
import sys
import json

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'speaker_identification'))

from speaker_identification.config.config import Config as IdentificationConfig
from speaker_identification.models.wsi_model import WSIModel
from speaker_identification.utils.audio_utils import load_audio, preprocess_audio


class SpeakerIdentificationEngine:
    """Engine for speaker identification inference."""
    
    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = IdentificationConfig()
        
        # Initialize model
        self.model = self._load_model(checkpoint_path)
        
        # Speaker database (embeddings + labels)
        self.speaker_database = {}
        self.speaker_embeddings = None
        self.speaker_labels = []
    
    def _load_model(self, checkpoint_path):
        """Load the identification model."""
        model = WSIModel(self.config).to(self.device)
        
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
        waveform = load_audio(audio_path, sr=self.config.sample_rate)
        waveform = preprocess_audio(waveform, self.config)
        waveform = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.extract_embedding(waveform)
        
        return embedding.cpu().numpy().flatten()
    
    def enroll_speaker(self, audio_path, speaker_name):
        """Enroll a new speaker."""
        embedding = self.extract_embedding(audio_path)
        
        if speaker_name in self.speaker_database:
            # Update existing speaker (average embeddings)
            existing = self.speaker_database[speaker_name]
            self.speaker_database[speaker_name] = (existing + embedding) / 2
        else:
            self.speaker_database[speaker_name] = embedding
        
        self._update_embedding_matrix()
        return f"✅ Enrolled speaker: {speaker_name}"
    
    def _update_embedding_matrix(self):
        """Update the embedding matrix for efficient search."""
        self.speaker_labels = list(self.speaker_database.keys())
        if self.speaker_labels:
            self.speaker_embeddings = np.stack(
                [self.speaker_database[name] for name in self.speaker_labels]
            )
        else:
            self.speaker_embeddings = None
    
    def identify_speaker(self, audio_path, top_k=3):
        """Identify the speaker from enrolled database."""
        if not self.speaker_database:
            return None, []
        
        query_embedding = self.extract_embedding(audio_path)
        
        # Compute similarities
        similarities = np.dot(self.speaker_embeddings, query_embedding) / (
            np.linalg.norm(self.speaker_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [
            (self.speaker_labels[i], float(similarities[i]))
            for i in top_indices
        ]
        
        return results[0] if results else None, results
    
    def get_enrolled_speakers(self):
        """Get list of enrolled speakers."""
        return list(self.speaker_database.keys())
    
    def clear_database(self):
        """Clear the speaker database."""
        self.speaker_database = {}
        self.speaker_embeddings = None
        self.speaker_labels = []
        return "🗑️ Speaker database cleared"


# Global engine instance
identification_engine = None


def get_identification_engine():
    """Get or create identification engine."""
    global identification_engine
    if identification_engine is None:
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'speaker_identification', 'outputs', 'best_model.pt'
        )
        identification_engine = SpeakerIdentificationEngine(checkpoint_path)
    return identification_engine


def enroll_speaker(audio, speaker_name):
    """Gradio function for speaker enrollment."""
    if audio is None:
        return "⚠️ Please upload an audio file", ""
    
    if not speaker_name or speaker_name.strip() == "":
        return "⚠️ Please enter a speaker name", ""
    
    try:
        engine = get_identification_engine()
        result = engine.enroll_speaker(audio, speaker_name.strip())
        
        enrolled_list = engine.get_enrolled_speakers()
        enrolled_text = "### 📋 Enrolled Speakers:\n" + "\n".join(
            [f"- {name}" for name in enrolled_list]
        )
        
        return result, enrolled_text
        
    except Exception as e:
        return f"❌ Error: {str(e)}", ""


def identify_speaker(audio):
    """Gradio function for speaker identification."""
    if audio is None:
        return "⚠️ Please upload an audio file", ""
    
    try:
        engine = get_identification_engine()
        
        if not engine.get_enrolled_speakers():
            return "⚠️ No speakers enrolled. Please enroll speakers first.", ""
        
        top_result, all_results = engine.identify_speaker(audio, top_k=5)
        
        if top_result is None:
            return "❌ Could not identify speaker", ""
        
        speaker_name, confidence = top_result
        
        result_html = f"""
        <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: #155724;">🎯 Identified Speaker</h2>
            <h1 style="color: #155724;">{speaker_name}</h1>
            <p>Confidence: {confidence:.2%}</p>
        </div>
        """
        
        details = "### 📊 Top Matches:\n"
        for i, (name, score) in enumerate(all_results, 1):
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            details += f"{i}. **{name}**: {score:.2%} |{bar}|\n"
        
        return result_html, details
        
    except Exception as e:
        return f"❌ Error: {str(e)}", ""


def clear_database():
    """Clear the speaker database."""
    engine = get_identification_engine()
    return engine.clear_database(), ""


def create_identification_tab():
    """Create the speaker identification tab interface."""
    
    gr.Markdown("""
    ## 🔍 Speaker Identification
    
    Identify speakers from a database of enrolled voices.
    
    **How it works:**
    1. First, enroll speakers with their audio samples
    2. Then, upload a test audio to identify the speaker
    """)
    
    with gr.Tabs():
        # Enrollment sub-tab
        with gr.TabItem("📝 Enroll Speaker"):
            gr.Markdown("### Add new speaker to the database")
            
            with gr.Row():
                with gr.Column():
                    enroll_audio = gr.Audio(
                        label="📎 Speaker Audio Sample",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    speaker_name = gr.Textbox(
                        label="Speaker Name",
                        placeholder="Enter speaker name..."
                    )
                    enroll_btn = gr.Button("✅ Enroll Speaker", variant="primary")
                
                with gr.Column():
                    enroll_result = gr.Markdown(label="Enrollment Result")
                    enrolled_list = gr.Markdown(label="Enrolled Speakers")
            
            clear_btn = gr.Button("🗑️ Clear All Speakers", variant="secondary")
            
            enroll_btn.click(
                fn=enroll_speaker,
                inputs=[enroll_audio, speaker_name],
                outputs=[enroll_result, enrolled_list]
            )
            
            clear_btn.click(
                fn=clear_database,
                inputs=[],
                outputs=[enroll_result, enrolled_list]
            )
        
        # Identification sub-tab
        with gr.TabItem("🔍 Identify Speaker"):
            gr.Markdown("### Identify who is speaking")
            
            with gr.Row():
                with gr.Column():
                    test_audio = gr.Audio(
                        label="🎤 Audio to Identify",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    identify_btn = gr.Button("🔍 Identify Speaker", variant="primary", size="lg")
                
                with gr.Column():
                    id_result_html = gr.HTML(label="Identification Result")
                    id_details = gr.Markdown(label="Match Details")
            
            identify_btn.click(
                fn=identify_speaker,
                inputs=[test_audio],
                outputs=[id_result_html, id_details]
            )
    
    gr.Markdown("""
    ### 💡 Tips
    - Enroll multiple samples per speaker for better accuracy
    - Use clean audio recordings (minimal background noise)
    - Ensure at least 3-5 seconds of speech per sample
    """)