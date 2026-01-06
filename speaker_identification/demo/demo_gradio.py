# demo/demo_gradio.py
"""
Gradio Web Demo for WSI Speaker Identification
Beautiful, interactive web interface for speaker verification
"""

import sys
from pathlib import Path

# Add parent directory to path (speaker_identification folder)
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import torch
import numpy as np
from typing import Tuple, Optional
import tempfile
import os

# Install gradio if not available
try:
    import gradio as gr
except ImportError:
    print("Installing gradio...")
    os.system("pip install gradio")
    import gradio as gr

# Import WSI modules
from config.config import WSIConfig
from data.preprocessing import AudioPreprocessor
from models.wsi_model import WSIModel


class SpeakerVerificationDemo:
    """Speaker Verification Demo using WSI model."""
    
    def __init__(self, checkpoint_path: str = None):
        """Initialize the demo with trained model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = WSIConfig()
        
        # Find checkpoint
        if checkpoint_path is None:
            possible_paths = [
                parent_dir / "outputs_v2" / "best_model.pt",
                parent_dir / "outputs" / "final_model.pt",
                parent_dir / "outputs" / "best_model.pt",
            ]
            for path in possible_paths:
                if path.exists():
                    checkpoint_path = str(path)
                    break
        
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                "No trained model found. Please train the model first:\n"
                "  python run_training_v2.py --epochs 20 --batch_size 32"
            )
        
        # Load model
        print(f"Loading WSI model from {checkpoint_path}...")
        self.model = WSIModel(
            whisper_model_name=self.config.model.whisper_model_name,
            embedding_dim=self.config.model.embedding_dim,
            projection_hidden_dim=self.config.model.projection_hidden_dim
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model loaded successfully!")
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(
            sample_rate=self.config.data.sample_rate,
            fixed_frames=self.config.data.fixed_input_frames,
            whisper_model_name=self.config.model.whisper_model_name
        )
        
        # Speaker database for enrollment
        self.enrolled_speakers = {}
        
        # Default threshold
        self.threshold = 0.3
    
    def get_embedding(self, audio_path: str) -> torch.Tensor:
        """Extract speaker embedding from audio file."""
        features = self.preprocessor.preprocess(audio_path).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.get_embedding(features)
        return embedding
    
    def verify_speakers(
        self, 
        audio1: str, 
        audio2: str,
        threshold: float = 0.3
    ) -> Tuple[str, float, str]:
        """
        Verify if two audio samples are from the same speaker.
        
        Returns:
            Tuple of (decision, similarity_score, details)
        """
        if audio1 is None or audio2 is None:
            return "❌ Please upload both audio files", 0.0, ""
        
        try:
            # Get embeddings
            emb1 = self.get_embedding(audio1)
            emb2 = self.get_embedding(audio2)
            
            # Compute similarity
            similarity = self.model.compute_similarity(emb1, emb2).item()
            
            # Make decision
            is_same = similarity >= threshold
            
            if is_same:
                decision = f"✅ SAME SPEAKER (Confidence: {similarity:.1%})"
            else:
                decision = f"❌ DIFFERENT SPEAKERS (Similarity: {similarity:.1%})"
            
            details = f"""
### 📊 Analysis Details

| Metric | Value |
|--------|-------|
| Similarity Score | {similarity:.4f} |
| Threshold | {threshold:.4f} |
| Decision | {"Same Speaker" if is_same else "Different Speakers"} |

### 📈 Score Interpretation
- **Score > 0.7**: Very likely same speaker
- **Score 0.4-0.7**: Possibly same speaker  
- **Score < 0.4**: Likely different speakers
            """
            
            return decision, similarity, details
            
        except Exception as e:
            return f"❌ Error: {str(e)}", 0.0, ""
    
    def enroll_speaker(
        self, 
        audio: str, 
        speaker_name: str
    ) -> str:
        """Enroll a new speaker in the database."""
        if audio is None:
            return "❌ Please upload an audio file"
        
        if not speaker_name or speaker_name.strip() == "":
            return "❌ Please enter a speaker name"
        
        try:
            # Get embedding
            embedding = self.get_embedding(audio)
            
            # Store in database
            speaker_name = speaker_name.strip()
            self.enrolled_speakers[speaker_name] = embedding
            
            return f"✅ Successfully enrolled: **{speaker_name}**\n\nTotal enrolled speakers: {len(self.enrolled_speakers)}"
            
        except Exception as e:
            return f"❌ Error enrolling speaker: {str(e)}"
    
    def identify_speaker(
        self, 
        audio: str,
        threshold: float = 0.3
    ) -> str:
        """Identify a speaker from enrolled database."""
        if audio is None:
            return "❌ Please upload an audio file"
        
        if len(self.enrolled_speakers) == 0:
            return "❌ No speakers enrolled. Please enroll speakers first."
        
        try:
            # Get embedding
            embedding = self.get_embedding(audio)
            
            # Compare with all enrolled speakers
            results = []
            for name, enrolled_emb in self.enrolled_speakers.items():
                similarity = self.model.compute_similarity(embedding, enrolled_emb).item()
                results.append((name, similarity))
            
            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Best match
            best_name, best_score = results[0]
            
            if best_score >= threshold:
                output = f"## ✅ Identified as: {best_name}\n"
                output += f"**Confidence: {best_score:.1%}**\n\n"
            else:
                output = f"## ❓ Unknown Speaker\n"
                output += f"Closest match: {best_name} ({best_score:.1%})\n\n"
            
            output += "### 📊 All Matches:\n\n"
            output += "| Speaker | Score | Match |\n"
            output += "|---------|-------|-------|\n"
            
            for name, score in results[:5]:  # Top 5
                bar = "🟩" * int(score * 10) + "⬜" * (10 - int(score * 10))
                status = "✅" if score >= threshold else "❌"
                output += f"| {name} | {bar} {score:.1%} | {status} |\n"
            
            return output
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def get_enrolled_speakers(self) -> str:
        """Get list of enrolled speakers."""
        if len(self.enrolled_speakers) == 0:
            return "No speakers enrolled yet."
        
        output = f"### 📋 Enrolled Speakers ({len(self.enrolled_speakers)})\n\n"
        for i, name in enumerate(self.enrolled_speakers.keys(), 1):
            output += f"{i}. {name}\n"
        
        return output
    
    def clear_database(self) -> str:
        """Clear all enrolled speakers."""
        count = len(self.enrolled_speakers)
        self.enrolled_speakers.clear()
        return f"✅ Cleared {count} enrolled speakers."


def create_demo():
    """Create and launch the Gradio demo."""
    
    # Initialize demo
    try:
        demo_app = SpeakerVerificationDemo()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None
    
    # Create Gradio interface
    with gr.Blocks(
        title="WSI Speaker Identification Demo",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
        # 🎤 WSI Speaker Identification Demo
        
        **Whisper Speaker Identification** - A state-of-the-art speaker recognition system 
        using pre-trained Whisper encoder with triplet loss and NT-Xent optimization.
        
        ---
        """)
        
        with gr.Tabs():
            
            # Tab 1: Speaker Verification
            with gr.TabItem("🔍 Speaker Verification"):
                gr.Markdown("""
                ### Compare Two Audio Samples
                Upload two audio files to check if they're from the same speaker.
                """)
                
                with gr.Row():
                    with gr.Column():
                        audio1 = gr.Audio(
                            label="🎵 Audio 1",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                    with gr.Column():
                        audio2 = gr.Audio(
                            label="🎵 Audio 2",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.3,
                    step=0.05,
                    label="Decision Threshold"
                )
                
                verify_btn = gr.Button("🔍 Verify Speakers", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        result_text = gr.Textbox(label="Result", lines=2, scale=2)
                    with gr.Column(scale=1):
                        similarity_score = gr.Number(label="Similarity Score", precision=4)
                
                details_text = gr.Markdown(label="Details")
                
                verify_btn.click(
                    fn=demo_app.verify_speakers,
                    inputs=[audio1, audio2, threshold_slider],
                    outputs=[result_text, similarity_score, details_text]
                )
                
                # Example files
                gr.Markdown("""
                ---
                💡 **Tips:**
                - Use clear audio recordings (2-10 seconds)
                - Avoid background noise
                - Supported formats: WAV, MP3, FLAC
                """)
            
            # Tab 2: Speaker Enrollment
            with gr.TabItem("📝 Speaker Enrollment"):
                gr.Markdown("""
                ### Enroll New Speakers
                Add speakers to the database for later identification.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        enroll_audio = gr.Audio(
                            label="🎵 Audio Sample",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        speaker_name = gr.Textbox(
                            label="Speaker Name",
                            placeholder="Enter speaker name..."
                        )
                        enroll_btn = gr.Button("📝 Enroll Speaker", variant="primary", size="lg")
                        enroll_result = gr.Markdown(label="Enrollment Result")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Enrolled Speakers")
                        enrolled_list = gr.Markdown(value="No speakers enrolled yet.")
                        refresh_btn = gr.Button("🔄 Refresh List")
                        clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
                
                enroll_btn.click(
                    fn=demo_app.enroll_speaker,
                    inputs=[enroll_audio, speaker_name],
                    outputs=[enroll_result]
                ).then(
                    fn=demo_app.get_enrolled_speakers,
                    outputs=[enrolled_list]
                )
                
                refresh_btn.click(
                    fn=demo_app.get_enrolled_speakers,
                    outputs=[enrolled_list]
                )
                
                clear_btn.click(
                    fn=demo_app.clear_database,
                    outputs=[enroll_result]
                ).then(
                    fn=demo_app.get_enrolled_speakers,
                    outputs=[enrolled_list]
                )
            
            # Tab 3: Speaker Identification
            with gr.TabItem("🔎 Speaker Identification"):
                gr.Markdown("""
                ### Identify Speaker
                Upload an audio file to identify who is speaking from enrolled speakers.
                """)
                
                with gr.Row():
                    with gr.Column():
                        identify_audio = gr.Audio(
                            label="🎵 Audio to Identify",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        identify_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.3,
                            step=0.05,
                            label="Identification Threshold"
                        )
                        identify_btn = gr.Button("🔎 Identify Speaker", variant="primary", size="lg")
                    
                    with gr.Column():
                        identify_result = gr.Markdown(label="Identification Result")
                
                identify_btn.click(
                    fn=demo_app.identify_speaker,
                    inputs=[identify_audio, identify_threshold],
                    outputs=[identify_result]
                )
            
            # Tab 4: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ### About WSI Speaker Identification
                
                **WSI (Whisper Speaker Identification)** is a framework that repurposes 
                the pre-trained Whisper ASR model for speaker recognition tasks.
                
                #### 🎯 Key Features
                - **Pre-trained Whisper Encoder**: Leverages multilingual acoustic representations
                - **Joint Loss Optimization**: Combines triplet loss with NT-Xent loss
                - **Multilingual Support**: Works across diverse languages
                - **Fast Inference**: Real-time speaker verification
                
                #### 🏗️ Model Architecture
                ```
                Audio → Whisper Encoder → Mean Pooling → Projection Head → Speaker Embedding (256-dim)
                ```
                
                #### 📊 Performance
                | Metric | Value |
                |--------|-------|
                | EER | ~9% |
                | AUC | 0.96+ |
                | Embedding Dim | 256 |
                
                #### 📚 Citation
                ```bibtex
                @article{wsi2025,
                  title={Whisper Speaker Identification: Leveraging Pre-trained 
                         Multilingual Transformers for Robust Speaker Embeddings},
                  author={Emon et al.},
                  year={2025}
                }
                ```
                
                ---
                
                **Built with** ❤️ using PyTorch, Transformers, and Gradio
                """)
        
        gr.Markdown("""
        ---
        💡 **Tips:**
        - Use clear audio recordings for best results
        - Recommended audio length: 2-10 seconds
        - Supported formats: WAV, MP3, FLAC, OGG
        """)
    
    return demo


def main():
    demo = create_demo()
    if demo:
        print("\n" + "="*60)
        print("🎤 WSI Speaker Identification Demo")
        print("="*60)
        print("Starting Gradio server...")
        print("Local URL: http://localhost:7860")
        print("="*60 + "\n")
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,  # Create public link
            show_error=True
        )


if __name__ == "__main__":
    main()