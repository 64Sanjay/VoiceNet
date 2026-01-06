# demo_streamlit.py
"""
Streamlit Dashboard for WSI Speaker Identification
Interactive analytics and visualization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import tempfile
import os

# Import WSI modules
from config.config import WSIConfig
from data.preprocessing import AudioPreprocessor
from models.wsi_model import WSIModel


@st.cache_resource
def load_model():
    """Load the WSI model (cached)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = WSIConfig()
    
    model = WSIModel(
        whisper_model_name=config.model.whisper_model_name,
        embedding_dim=config.model.embedding_dim,
        projection_hidden_dim=config.model.projection_hidden_dim
    )
    
    # Try different checkpoint paths
    checkpoint_paths = ["outputs_v2/best_model.pt", "outputs/final_model.pt"]
    checkpoint_path = None
    
    for path in checkpoint_paths:
        if Path(path).exists():
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        st.error("❌ No trained model found. Please train the model first.")
        return None, None, None
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    preprocessor = AudioPreprocessor(
        sample_rate=config.data.sample_rate,
        fixed_frames=config.data.fixed_input_frames,
        whisper_model_name=config.model.whisper_model_name
    )
    
    return model, preprocessor, device


def get_embedding(model, preprocessor, device, audio_path):
    """Extract speaker embedding."""
    features = preprocessor.preprocess(audio_path).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.get_embedding(features)
    return embedding


def main():
    st.set_page_config(
        page_title="WSI Speaker Identification",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 WSI Speaker Identification Dashboard")
    st.markdown("---")
    
    # Load model
    model, preprocessor, device = load_model()
    
    if model is None:
        st.stop()
    
    # Initialize session state
    if 'enrolled_speakers' not in st.session_state:
        st.session_state.enrolled_speakers = {}
    if 'verification_history' not in st.session_state:
        st.session_state.verification_history = []
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.3, 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.markdown(f"- **Device**: {device}")
    st.sidebar.markdown(f"- **Embedding Dim**: 256")
    st.sidebar.markdown(f"- **Enrolled Speakers**: {len(st.session_state.enrolled_speakers)}")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Verification", 
        "📝 Enrollment", 
        "🔎 Identification",
        "📈 Analytics"
    ])
    
    # Tab 1: Verification
    with tab1:
        st.header("Speaker Verification")
        st.markdown("Compare two audio samples to check if they're from the same speaker.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎵 Audio 1")
            audio1 = st.file_uploader("Upload first audio", type=['wav', 'mp3', 'flac'], key="audio1")
        
        with col2:
            st.subheader("🎵 Audio 2")
            audio2 = st.file_uploader("Upload second audio", type=['wav', 'mp3', 'flac'], key="audio2")
        
        if st.button("🔍 Verify Speakers", type="primary"):
            if audio1 and audio2:
                with st.spinner("Processing..."):
                    # Save temp files
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f1:
                        f1.write(audio1.read())
                        path1 = f1.name
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f2:
                        f2.write(audio2.read())
                        path2 = f2.name
                    
                    try:
                        # Get embeddings
                        emb1 = get_embedding(model, preprocessor, device, path1)
                        emb2 = get_embedding(model, preprocessor, device, path2)
                        
                        # Compute similarity
                        similarity = model.compute_similarity(emb1, emb2).item()
                        is_same = similarity >= threshold
                        
                        # Display result
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Similarity Score", f"{similarity:.4f}")
                        
                        with col2:
                            st.metric("Threshold", f"{threshold:.4f}")
                        
                        with col3:
                            if is_same:
                                st.success("✅ SAME SPEAKER")
                            else:
                                st.error("❌ DIFFERENT SPEAKERS")
                        
                        # Visualization
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=similarity,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Similarity Score"},
                            gauge={
                                'axis': {'range': [-1, 1]},
                                'bar': {'color': "green" if is_same else "red"},
                                'steps': [
                                    {'range': [-1, 0], 'color': "lightgray"},
                                    {'range': [0, threshold], 'color': "lightyellow"},
                                    {'range': [threshold, 1], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': threshold
                                }
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Save to history
                        st.session_state.verification_history.append({
                            'similarity': similarity,
                            'threshold': threshold,
                            'result': 'Same' if is_same else 'Different'
                        })
                        
                    finally:
                        os.unlink(path1)
                        os.unlink(path2)
            else:
                st.warning("Please upload both audio files.")
    
    # Tab 2: Enrollment
    with tab2:
        st.header("Speaker Enrollment")
        st.markdown("Add speakers to the database for later identification.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            enroll_audio = st.file_uploader("Upload audio sample", type=['wav', 'mp3', 'flac'], key="enroll")
            speaker_name = st.text_input("Speaker Name")
            
            if st.button("📝 Enroll Speaker", type="primary"):
                if enroll_audio and speaker_name:
                    with st.spinner("Enrolling..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                            f.write(enroll_audio.read())
                            path = f.name
                        
                        try:
                            embedding = get_embedding(model, preprocessor, device, path)
                            st.session_state.enrolled_speakers[speaker_name] = embedding.cpu()
                            st.success(f"✅ Enrolled: {speaker_name}")
                        finally:
                            os.unlink(path)
                else:
                    st.warning("Please provide both audio and name.")
        
        with col2:
            st.subheader("Enrolled Speakers")
            if st.session_state.enrolled_speakers:
                for name in st.session_state.enrolled_speakers:
                    st.markdown(f"• {name}")
            else:
                st.info("No speakers enrolled yet.")
            
            if st.button("🗑️ Clear All"):
                st.session_state.enrolled_speakers.clear()
                st.rerun()
    
    # Tab 3: Identification
    with tab3:
        st.header("Speaker Identification")
        st.markdown("Identify who is speaking from enrolled speakers.")
        
        if not st.session_state.enrolled_speakers:
            st.warning("Please enroll some speakers first.")
        else:
            identify_audio = st.file_uploader("Upload audio to identify", type=['wav', 'mp3', 'flac'], key="identify")
            
            if st.button("🔎 Identify", type="primary"):
                if identify_audio:
                    with st.spinner("Identifying..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                            f.write(identify_audio.read())
                            path = f.name
                        
                        try:
                            embedding = get_embedding(model, preprocessor, device, path)
                            
                            results = []
                            for name, enrolled_emb in st.session_state.enrolled_speakers.items():
                                enrolled_emb = enrolled_emb.to(device)
                                sim = model.compute_similarity(embedding, enrolled_emb).item()
                                results.append({'Speaker': name, 'Similarity': sim})
                            
                            df = pd.DataFrame(results)
                            df = df.sort_values('Similarity', ascending=False)
                            
                            # Best match
                            best = df.iloc[0]
                            if best['Similarity'] >= threshold:
                                st.success(f"✅ Identified as: **{best['Speaker']}** ({best['Similarity']:.1%})")
                            else:
                                st.warning(f"❓ Unknown speaker. Closest: {best['Speaker']} ({best['Similarity']:.1%})")
                            
                            # Chart
                            fig = px.bar(df, x='Speaker', y='Similarity', 
                                        color='Similarity', 
                                        color_continuous_scale='RdYlGn')
                            fig.add_hline(y=threshold, line_dash="dash", 
                                         annotation_text=f"Threshold ({threshold})")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        finally:
                            os.unlink(path)
    
    # Tab 4: Analytics
    with tab4:
        st.header("Analytics")
        
        if st.session_state.verification_history:
            df = pd.DataFrame(st.session_state.verification_history)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Verification History")
                fig = px.histogram(df, x='similarity', color='result',
                                  title="Similarity Score Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Results Summary")
                result_counts = df['result'].value_counts()
                fig = px.pie(values=result_counts.values, names=result_counts.index,
                            title="Verification Results")
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df)
        else:
            st.info("No verification history yet. Start verifying speakers to see analytics.")


if __name__ == "__main__":
    main()