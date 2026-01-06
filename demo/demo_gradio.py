#!/usr/bin/env python3
"""
Unified Speaker Recognition Demo
Combines Speaker Verification, Identification, and Diarization
"""

import gradio as gr
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.verification_tab import create_verification_tab
from demo.identification_tab import create_identification_tab
from demo.diarization_tab import create_diarization_tab


def create_demo():
    """Create the unified Gradio demo interface."""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .tab-nav button {
        font-size: 16px;
        font-weight: bold;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 10px;
    }
    .success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .failure {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Speaker Recognition System") as demo:
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🎤 Speaker Recognition System</h1>
            <p>Unified platform for Speaker Verification, Identification, and Diarization</p>
        </div>
        """)
        
        # Main tabs
        with gr.Tabs():
            # Tab 1: Speaker Verification
            with gr.TabItem("🔐 Speaker Verification", id=0):
                create_verification_tab()
            
            # Tab 2: Speaker Identification
            with gr.TabItem("🔍 Speaker Identification", id=1):
                create_identification_tab()
            
            # Tab 3: Speaker Diarization
            with gr.TabItem("👥 Speaker Diarization", id=2):
                create_diarization_tab()
            
            # Tab 4: About
            with gr.TabItem("ℹ️ About", id=3):
                gr.Markdown("""
                ## About This System
                
                This unified speaker recognition system provides three main functionalities:
                
                ### 🔐 Speaker Verification
                - **Purpose**: Verify if two audio samples belong to the same speaker
                - **Use Case**: Authentication, access control
                - **Output**: Similarity score and verification decision
                
                ### 🔍 Speaker Identification
                - **Purpose**: Identify who is speaking from a set of enrolled speakers
                - **Use Case**: Speaker tagging, personalized services
                - **Output**: Identified speaker with confidence score
                
                ### 👥 Speaker Diarization
                - **Purpose**: Segment audio by speaker ("who spoke when")
                - **Use Case**: Meeting transcription, call center analytics
                - **Output**: Timeline with speaker segments
                
                ---
                
                ### Technical Details
                - **Verification Model**: CAM++ / DTDNN with AAM-Softmax
                - **Identification Model**: Whisper-based Speaker Identification (WSI)
                - **Diarization Model**: Segmentation + Clustering approach
                
                ### Datasets Used
                - VoxCeleb1/2 for verification
                - LibriSpeech for identification
                - AISHELL-4 for diarization
                """)
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 20px; padding: 10px; color: #666;">
            <p>Speaker Recognition System | Built with ❤️ using PyTorch and Gradio</p>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )