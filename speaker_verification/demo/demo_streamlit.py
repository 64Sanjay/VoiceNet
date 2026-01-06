# speaker_verification/demo/demo_streamlit.py
"""
Streamlit Web Demo for CAM++ Speaker Verification.

Run with: streamlit run demo/demo_streamlit.py
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import tempfile
import os
import io

try:
    import streamlit as st
except ImportError:
    print("Installing streamlit...")
    os.system("pip install streamlit")
    import streamlit as st

try:
    import soundfile as sf
except ImportError:
    os.system("pip install soundfile")
    import soundfile as sf

try:
    from audio_recorder_streamlit import audio_recorder
    HAS_AUDIO_RECORDER = True
except ImportError:
    HAS_AUDIO_RECORDER = False
    print("Note: Install audio-recorder-streamlit for microphone recording:")
    print("  pip install audio-recorder-streamlit")

from data.preprocessing import AudioPreprocessor
from models.cam_plus_plus import CAMPlusPlusClassifier


# ============================================================================
# Model Configuration Inference
# ============================================================================

def infer_model_config_from_checkpoint(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Infer model configuration from checkpoint state_dict.
    """
    config = {}
    
    # Infer FCM channels
    if 'encoder.fcm.conv1.weight' in state_dict:
        config['fcm_channels'] = state_dict['encoder.fcm.conv1.weight'].shape[0]
    else:
        config['fcm_channels'] = 32
    
    # Count FCM blocks
    fcm_block_count = 0
    for key in state_dict.keys():
        if key.startswith('encoder.fcm.blocks.') and '.conv1.weight' in key:
            block_idx = int(key.split('.')[3])
            fcm_block_count = max(fcm_block_count, block_idx + 1)
    config['fcm_num_blocks'] = fcm_block_count if fcm_block_count > 0 else 2
    
    # Infer init_channels
    if 'encoder.input_tdnn.0.weight' in state_dict:
        config['init_channels'] = state_dict['encoder.input_tdnn.0.weight'].shape[0]
    else:
        config['init_channels'] = 128
    
    # Infer growth_rate
    if 'encoder.dtdnn_blocks.0.layers.0.tdnn.weight' in state_dict:
        config['growth_rate'] = state_dict['encoder.dtdnn_blocks.0.layers.0.tdnn.weight'].shape[0]
    else:
        config['growth_rate'] = 32
    
    # Infer bn_size
    if 'encoder.dtdnn_blocks.0.layers.0.fc1.weight' in state_dict:
        bn_channels = state_dict['encoder.dtdnn_blocks.0.layers.0.fc1.weight'].shape[0]
        growth_rate = config['growth_rate']
        config['bn_size'] = bn_channels // growth_rate if growth_rate > 0 else 4
    else:
        config['bn_size'] = 4
    
    # Count D-TDNN blocks and layers
    dtdnn_blocks = {}
    for key in state_dict.keys():
        if key.startswith('encoder.dtdnn_blocks.') and '.layers.' in key:
            parts = key.split('.')
            block_idx = int(parts[2])
            layer_idx = int(parts[4])
            if block_idx not in dtdnn_blocks:
                dtdnn_blocks[block_idx] = 0
            dtdnn_blocks[block_idx] = max(dtdnn_blocks[block_idx], layer_idx + 1)
    
    if dtdnn_blocks:
        config['dtdnn_blocks'] = tuple([dtdnn_blocks.get(i, 6) for i in range(len(dtdnn_blocks))])
    else:
        config['dtdnn_blocks'] = (12, 24, 16)
    
    # Infer embedding_dim
    if 'encoder.embedding.weight' in state_dict:
        config['embedding_dim'] = state_dict['encoder.embedding.weight'].shape[0]
    else:
        config['embedding_dim'] = 192
    
    # Infer num_classes
    if 'weight' in state_dict:
        config['num_classes'] = state_dict['weight'].shape[0]
    elif 'classifier.weight' in state_dict:
        config['num_classes'] = state_dict['classifier.weight'].shape[0]
    else:
        config['num_classes'] = 100
    
    # Infer n_mels
    if 'encoder.input_tdnn.0.weight' in state_dict:
        input_channels = state_dict['encoder.input_tdnn.0.weight'].shape[1]
        fcm_ch = config['fcm_channels']
        n_mels_div4 = input_channels // fcm_ch
        config['n_mels'] = n_mels_div4 * 4
    else:
        config['n_mels'] = 80
    
    return config


# ============================================================================
# Model Loading
# ============================================================================

@st.cache_resource
def load_model(checkpoint_path: str = None):
    """Load CAM++ model with caching."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find checkpoint
    if checkpoint_path is None:
        possible_paths = [
            parent_dir / "checkpoints" / "best_model.pt",
            parent_dir / "checkpoints" / "final_model.pt",
            parent_dir / "checkpoints" / "latest_model.pt",
        ]
        for path in possible_paths:
            if path.exists():
                checkpoint_path = str(path)
                break
    
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        return None, None, None, "No checkpoint found"
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Get state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Infer config
        config = infer_model_config_from_checkpoint(state_dict)
        
        # Check saved config
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            if hasattr(saved_config, 'model'):
                model_cfg = saved_config.model
                if hasattr(saved_config, 'audio'):
                    config['n_mels'] = getattr(saved_config.audio, 'n_mels', config['n_mels'])
                config['embedding_dim'] = getattr(model_cfg, 'embedding_dim', config['embedding_dim'])
                config['fcm_channels'] = getattr(model_cfg, 'fcm_channels', config['fcm_channels'])
                config['fcm_num_blocks'] = getattr(model_cfg, 'fcm_num_blocks', config['fcm_num_blocks'])
                dtdnn_blocks = getattr(model_cfg, 'dtdnn_blocks', config['dtdnn_blocks'])
                config['dtdnn_blocks'] = tuple(dtdnn_blocks) if isinstance(dtdnn_blocks, list) else dtdnn_blocks
                config['growth_rate'] = getattr(model_cfg, 'growth_rate', config['growth_rate'])
                config['init_channels'] = getattr(model_cfg, 'init_channels', config['init_channels'])
                config['bn_size'] = getattr(model_cfg, 'bn_size', config['bn_size'])
        
        # Create model
        cam_kwargs = {
            'n_mels': config['n_mels'],
            'fcm_channels': config['fcm_channels'],
            'fcm_num_blocks': config['fcm_num_blocks'],
            'dtdnn_blocks': config['dtdnn_blocks'],
            'growth_rate': config['growth_rate'],
            'init_channels': config['init_channels'],
            'bn_size': config['bn_size'],
        }
        
        model = CAMPlusPlusClassifier(
            num_classes=config['num_classes'],
            embedding_dim=config['embedding_dim'],
            **cam_kwargs
        )
        
        # Load weights
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        
        # Create preprocessor
        preprocessor = AudioPreprocessor(
            sample_rate=16000,
            n_mels=config['n_mels']
        )
        
        return model, preprocessor, device, None
        
    except Exception as e:
        import traceback
        return None, None, None, f"Error loading model: {str(e)}\n{traceback.format_exc()}"


def get_embedding(model, preprocessor, device, audio_path: str) -> torch.Tensor:
    """Extract speaker embedding from audio file."""
    features = preprocessor.process(audio_path).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.extract_embedding(features)
    return embedding


def save_uploaded_audio(uploaded_file) -> str:
    """Save uploaded audio to temporary file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def save_recorded_audio(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """Save recorded audio bytes to temporary file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        # audio_recorder returns raw bytes, need to save as WAV
        tmp_file.write(audio_bytes)
        return tmp_file.name


# ============================================================================
# Streamlit UI Components
# ============================================================================

def create_similarity_bar(similarity: float, width: int = 20) -> str:
    """Create a text-based similarity bar."""
    filled = int(similarity * width)
    empty = width - filled
    return "🟩" * filled + "⬜" * empty


def display_similarity_gauge(similarity: float):
    """Display similarity as a visual gauge."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Color based on similarity
        if similarity >= 0.5:
            color = "#28a745"  # Green
        elif similarity >= 0.3:
            color = "#ffc107"  # Yellow
        else:
            color = "#dc3545"  # Red
        
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 48px; font-weight: bold; color: {color};">
                {similarity:.1%}
            </div>
            <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px; width: 100%;">
                <div style="background-color: {color}; border-radius: 10px; height: 20px; width: {similarity*100}%;"></div>
            </div>
            <div style="margin-top: 5px; color: #666;">Similarity Score</div>
        </div>
        """, unsafe_allow_html=True)


def audio_input_component(key: str, label: str = "Audio"):
    """Create an audio input component with upload and optional recording."""
    audio_path = None
    
    tab1, tab2 = st.tabs(["📁 Upload", "🎤 Record"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            f"Upload {label}",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            key=f"{key}_upload"
        )
        if uploaded_file is not None:
            st.audio(uploaded_file)
            audio_path = save_uploaded_audio(uploaded_file)
    
    with tab2:
        if HAS_AUDIO_RECORDER:
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e74c3c",
                neutral_color="#3498db",
                key=f"{key}_record"
            )
            if audio_bytes:
                st.audio(audio_bytes)
                audio_path = save_recorded_audio(audio_bytes)
        else:
            st.info("🎤 Microphone recording requires `audio-recorder-streamlit` package.")
            st.code("pip install audio-recorder-streamlit")
    
    return audio_path


# ============================================================================
# Main Application Pages
# ============================================================================

def page_verification(model, preprocessor, device):
    """Speaker Verification page."""
    st.header("🔍 Speaker Verification")
    st.markdown("Compare two audio samples to check if they're from the same speaker.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Audio 1")
        audio1_path = audio_input_component("audio1", "first audio")
    
    with col2:
        st.subheader("Audio 2")
        audio2_path = audio_input_component("audio2", "second audio")
    
    st.divider()
    
    # Threshold slider
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Higher threshold = stricter matching"
    )
    
    # Verify button
    if st.button("🔍 Verify Speakers", type="primary", use_container_width=True):
        if audio1_path is None or audio2_path is None:
            st.error("❌ Please provide both audio samples.")
        else:
            with st.spinner("Analyzing audio samples..."):
                try:
                    # Get embeddings
                    emb1 = get_embedding(model, preprocessor, device, audio1_path)
                    emb2 = get_embedding(model, preprocessor, device, audio2_path)
                    
                    # Calculate similarity
                    similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
                    is_same = similarity >= threshold
                    
                    # Display results
                    st.divider()
                    
                    if is_same:
                        st.success(f"## ✅ SAME SPEAKER")
                    else:
                        st.error(f"## ❌ DIFFERENT SPEAKERS")
                    
                    display_similarity_gauge(similarity)
                    
                    # Details
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Similarity", f"{similarity:.4f}")
                    col2.metric("Threshold", f"{threshold:.4f}")
                    col3.metric("Decision", "Same" if is_same else "Different")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                finally:
                    # Cleanup temp files
                    for path in [audio1_path, audio2_path]:
                        if path and os.path.exists(path):
                            try:
                                os.unlink(path)
                            except:
                                pass


def page_enrollment(model, preprocessor, device):
    """Speaker Enrollment page."""
    st.header("📝 Speaker Enrollment")
    st.markdown("Add speakers to the database for later identification.")
    
    # Initialize session state for enrolled speakers
    if 'enrolled_speakers' not in st.session_state:
        st.session_state.enrolled_speakers = {}
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enroll New Speaker")
        
        speaker_name = st.text_input(
            "Speaker Name",
            placeholder="Enter a unique name...",
            key="enroll_name"
        )
        
        audio_path = audio_input_component("enroll", "enrollment audio")
        
        if st.button("📝 Enroll Speaker", type="primary", use_container_width=True):
            if not speaker_name or not speaker_name.strip():
                st.error("❌ Please enter a speaker name.")
            elif audio_path is None:
                st.error("❌ Please provide an audio sample.")
            else:
                with st.spinner("Processing enrollment..."):
                    try:
                        embedding = get_embedding(model, preprocessor, device, audio_path)
                        name = speaker_name.strip()
                        
                        is_update = name in st.session_state.enrolled_speakers
                        st.session_state.enrolled_speakers[name] = embedding.cpu()
                        
                        if is_update:
                            st.success(f"🔄 Updated: **{name}**")
                        else:
                            st.success(f"✅ Enrolled: **{name}**")
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                    finally:
                        if audio_path and os.path.exists(audio_path):
                            try:
                                os.unlink(audio_path)
                            except:
                                pass
    
    with col2:
        st.subheader("📋 Enrolled Speakers")
        
        if len(st.session_state.enrolled_speakers) == 0:
            st.info("No speakers enrolled yet.")
        else:
            st.markdown(f"**Total: {len(st.session_state.enrolled_speakers)}**")
            
            for i, name in enumerate(sorted(st.session_state.enrolled_speakers.keys()), 1):
                col_name, col_del = st.columns([4, 1])
                col_name.markdown(f"{i}. {name}")
                if col_del.button("🗑️", key=f"del_{name}", help=f"Remove {name}"):
                    del st.session_state.enrolled_speakers[name]
                    st.rerun()
            
            st.divider()
            
            if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
                st.session_state.enrolled_speakers = {}
                st.success("Database cleared!")
                st.rerun()


def page_identification(model, preprocessor, device):
    """Speaker Identification page."""
    st.header("🔎 Speaker Identification")
    st.markdown("Identify who is speaking from the enrolled database.")
    
    # Check if speakers are enrolled
    if 'enrolled_speakers' not in st.session_state or len(st.session_state.enrolled_speakers) == 0:
        st.warning("⚠️ No speakers enrolled. Please enroll speakers first in the **Enrollment** tab.")
        return
    
    st.info(f"📋 {len(st.session_state.enrolled_speakers)} speaker(s) in database")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        audio_path = audio_input_component("identify", "audio to identify")
    
    with col2:
        threshold = st.slider(
            "Recognition Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05,
            help="Minimum similarity to consider a match"
        )
    
    if st.button("🔎 Identify Speaker", type="primary", use_container_width=True):
        if audio_path is None:
            st.error("❌ Please provide an audio sample.")
        else:
            with st.spinner("Identifying speaker..."):
                try:
                    embedding = get_embedding(model, preprocessor, device, audio_path)
                    
                    # Compare with all enrolled speakers
                    results = []
                    for name, enrolled_emb in st.session_state.enrolled_speakers.items():
                        enrolled_emb = enrolled_emb.to(device)
                        sim = torch.nn.functional.cosine_similarity(
                            embedding, enrolled_emb
                        ).item()
                        results.append((name, sim))
                    
                    # Sort by similarity
                    results.sort(key=lambda x: x[1], reverse=True)
                    best_name, best_score = results[0]
                    
                    st.divider()
                    
                    # Display result
                    if best_score >= threshold:
                        st.success(f"## ✅ Identified: **{best_name}**")
                        st.markdown(f"**Confidence: {best_score:.1%}**")
                    else:
                        st.warning(f"## ❓ Unknown Speaker")
                        st.markdown(f"Closest match: **{best_name}** ({best_score:.1%})")
                        st.caption(f"Score below threshold ({threshold:.1%})")
                    
                    display_similarity_gauge(best_score)
                    
                    # Show all matches
                    st.divider()
                    st.subheader("📊 All Matches")
                    
                    for i, (name, score) in enumerate(results, 1):
                        col1, col2, col3 = st.columns([1, 3, 1])
                        col1.markdown(f"**#{i}**")
                        col2.markdown(f"{name}")
                        
                        # Progress bar for score
                        if score >= threshold:
                            col3.markdown(f"✅ {score:.1%}")
                        else:
                            col3.markdown(f"❌ {score:.1%}")
                        
                        st.progress(score)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                finally:
                    if audio_path and os.path.exists(audio_path):
                        try:
                            os.unlink(audio_path)
                        except:
                            pass


def page_about():
    """About page."""
    st.header("ℹ️ About CAM++")
    
    st.markdown("""
    **CAM++** is a fast and efficient speaker verification network designed for real-world applications.
    
    ## 🏗️ Architecture
    
    | Component | Description |
    |-----------|-------------|
    | **FCM** | Front-end Convolution Module - 2D CNN for time-frequency processing |
    | **D-TDNN** | Densely connected Time Delay Neural Network backbone |
    | **CAM** | Context-Aware Masking - focuses on discriminative speaker features |
    | **ASP** | Attentive Statistics Pooling - captures global temporal context |
    
    ## 📊 Pipeline
    ```
    Audio → Mel-Spectrogram → FCM → D-TDNN (with CAM) → ASP → Embedding
    ```
    
    ## ✨ Key Features
    
    - ⚡ **Fast inference** - Optimized for real-time applications
    - 💾 **Low memory** - Efficient dense connections  
    - 🎯 **High accuracy** - State-of-the-art on VoxCeleb benchmarks
    - 🔊 **Robust** - Handles noise and channel variations
    
    ## 📈 Performance (from paper)
    
    | Dataset | EER (%) | MinDCF |
    |---------|---------|--------|
    | VoxCeleb1-O | 0.73 | 0.0911 |
    | VoxCeleb1-E | 0.87 | 0.1020 |
    | VoxCeleb1-H | 1.63 | 0.1680 |
    
    ## 📚 Reference
    
    > Hui Wang, Siqi Zheng, Yafeng Chen, Luyao Cheng, Qian Chen.  
    > **"CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking"**  
    > *INTERSPEECH 2023*
    
    ## 🔗 Links
    
    - [Paper on arXiv](https://arxiv.org/abs/2303.00332)
    - [3D-Speaker Project](https://github.com/alibaba-damo-academy/3D-Speaker)
    """)


def page_batch_processing(model, preprocessor, device):
    """Batch processing page for multiple files."""
    st.header("📦 Batch Processing")
    st.markdown("Process multiple audio files at once.")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Audio Files",
        type=["wav", "mp3", "flac", "ogg"],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if not uploaded_files:
        st.info("Upload multiple audio files to extract embeddings or compare them.")
        return
    
    st.success(f"📁 {len(uploaded_files)} file(s) uploaded")
    
    # Processing options
    option = st.radio(
        "Processing Mode",
        ["Extract Embeddings", "Pairwise Comparison", "Compare to Reference"],
        horizontal=True
    )
    
    if option == "Extract Embeddings":
        if st.button("🔄 Extract All Embeddings", type="primary"):
            embeddings = {}
            progress = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        audio_path = save_uploaded_audio(file)
                        emb = get_embedding(model, preprocessor, device, audio_path)
                        embeddings[file.name] = emb.cpu().numpy()
                        os.unlink(audio_path)
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {e}")
                
                progress.progress((i + 1) / len(uploaded_files))
            
            st.success(f"✅ Extracted {len(embeddings)} embeddings")
            
            # Option to download
            if embeddings:
                import pickle
                emb_bytes = pickle.dumps(embeddings)
                st.download_button(
                    "📥 Download Embeddings",
                    data=emb_bytes,
                    file_name="embeddings.pkl",
                    mime="application/octet-stream"
                )
    
    elif option == "Pairwise Comparison":
        if len(uploaded_files) < 2:
            st.warning("Need at least 2 files for pairwise comparison.")
        elif st.button("🔄 Compare All Pairs", type="primary"):
            # Extract all embeddings first
            embeddings = {}
            progress = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                audio_path = save_uploaded_audio(file)
                emb = get_embedding(model, preprocessor, device, audio_path)
                embeddings[file.name] = emb
                os.unlink(audio_path)
                progress.progress((i + 1) / len(uploaded_files))
            
            # Compute pairwise similarities
            st.subheader("📊 Similarity Matrix")
            
            names = list(embeddings.keys())
            n = len(names)
            matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    sim = torch.nn.functional.cosine_similarity(
                        embeddings[names[i]], embeddings[names[j]]
                    ).item()
                    matrix[i, j] = sim
            
            # Display as dataframe
            import pandas as pd
            df = pd.DataFrame(matrix, index=names, columns=names)
            st.dataframe(df.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=1))
    
    elif option == "Compare to Reference":
        st.markdown("Select one file as reference and compare all others to it.")
        
        ref_file = st.selectbox(
            "Reference File",
            options=[f.name for f in uploaded_files]
        )
        
        threshold = st.slider("Threshold", 0.1, 0.9, 0.3, 0.05)
        
        if st.button("🔄 Compare to Reference", type="primary"):
            # Get reference embedding
            ref_uploaded = next(f for f in uploaded_files if f.name == ref_file)
            ref_path = save_uploaded_audio(ref_uploaded)
            ref_emb = get_embedding(model, preprocessor, device, ref_path)
            os.unlink(ref_path)
            
            # Compare all others
            results = []
            progress = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                if file.name != ref_file:
                    audio_path = save_uploaded_audio(file)
                    emb = get_embedding(model, preprocessor, device, audio_path)
                    sim = torch.nn.functional.cosine_similarity(ref_emb, emb).item()
                    results.append((file.name, sim, sim >= threshold))
                    os.unlink(audio_path)
                
                progress.progress((i + 1) / len(uploaded_files))
            
            # Display results
            st.subheader("📊 Results")
            
            for name, sim, is_match in sorted(results, key=lambda x: x[1], reverse=True):
                col1, col2, col3 = st.columns([3, 1, 1])
                col1.markdown(f"**{name}**")
                col2.markdown(f"{sim:.1%}")
                col3.markdown("✅" if is_match else "❌")
                st.progress(sim)


# ============================================================================
# Main Application
# ============================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="CAM++ Speaker Verification",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .main > div {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🎤 CAM++")
        st.markdown("**Speaker Verification System**")
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🔍 Verification", "📝 Enrollment", "🔎 Identification", 
             "📦 Batch Processing", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Model status
        st.markdown("### 📊 Model Status")
    
    # Load model
    model, preprocessor, device, error = load_model()
    
    with st.sidebar:
        if error:
            st.error(f"❌ {error}")
        else:
            st.success("✅ Model loaded")
            st.caption(f"Device: {device}")
            
            # Show enrolled speakers count
            if 'enrolled_speakers' in st.session_state:
                st.metric("Enrolled Speakers", len(st.session_state.enrolled_speakers))
    
    # Main content
    if error:
        st.error("## ❌ Model Loading Failed")
        st.markdown(f"""
        ```
        {error}
        ```
        
        ### 📋 Troubleshooting
        
        1. **Train a model first:**
           ```bash
           python train.py
           ```
        
        2. **Check checkpoint exists:**
           ```bash
           ls -la checkpoints/
           ```
        
        3. **Verify model architecture matches training config**
        """)
        return
    
    # Route to pages
    if page == "🔍 Verification":
        page_verification(model, preprocessor, device)
    elif page == "📝 Enrollment":
        page_enrollment(model, preprocessor, device)
    elif page == "🔎 Identification":
        page_identification(model, preprocessor, device)
    elif page == "📦 Batch Processing":
        page_batch_processing(model, preprocessor, device)
    elif page == "ℹ️ About":
        page_about()
    
    # Footer
    st.divider()
    st.markdown(
        "<center><p style='color: #888;'>Made with ❤️ using CAM++ | Speaker Verification System</p></center>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()