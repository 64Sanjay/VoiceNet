# speaker_verification/demo/demo_gradio.py
"""
Gradio Web Demo for CAM++ Speaker Verification.
"""

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
import tempfile
import os

try:
    import gradio as gr
except ImportError:
    print("Installing gradio...")
    os.system("pip install gradio")
    import gradio as gr

from data.preprocessing import AudioPreprocessor
from models.cam_plus_plus import CAMPlusPlusClassifier


def infer_model_config_from_checkpoint(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Infer model configuration from checkpoint state_dict.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Dictionary with inferred model configuration
    """
    config = {}
    
    # Infer FCM channels from first conv layer
    if 'encoder.fcm.conv1.weight' in state_dict:
        config['fcm_channels'] = state_dict['encoder.fcm.conv1.weight'].shape[0]
    else:
        config['fcm_channels'] = 32  # default
    
    # Count FCM blocks
    fcm_block_count = 0
    for key in state_dict.keys():
        if key.startswith('encoder.fcm.blocks.') and '.conv1.weight' in key:
            block_idx = int(key.split('.')[3])
            fcm_block_count = max(fcm_block_count, block_idx + 1)
    config['fcm_num_blocks'] = fcm_block_count if fcm_block_count > 0 else 2
    
    # Infer init_channels from input_tdnn
    if 'encoder.input_tdnn.0.weight' in state_dict:
        config['init_channels'] = state_dict['encoder.input_tdnn.0.weight'].shape[0]
    else:
        config['init_channels'] = 128  # default
    
    # Infer growth_rate from first D-TDNN layer output (tdnn.weight shape[0])
    if 'encoder.dtdnn_blocks.0.layers.0.tdnn.weight' in state_dict:
        config['growth_rate'] = state_dict['encoder.dtdnn_blocks.0.layers.0.tdnn.weight'].shape[0]
    else:
        config['growth_rate'] = 32  # default
    
    # Infer bn_size from fc1 layer
    # fc1 projects from input_channels to bn_channels (= growth_rate * bn_size)
    # But we need to figure out bn_size = bn_channels / growth_rate
    if 'encoder.dtdnn_blocks.0.layers.0.fc1.weight' in state_dict:
        bn_channels = state_dict['encoder.dtdnn_blocks.0.layers.0.fc1.weight'].shape[0]
        growth_rate = config['growth_rate']
        config['bn_size'] = bn_channels // growth_rate if growth_rate > 0 else 4
    else:
        config['bn_size'] = 4  # default
    
    # Count D-TDNN blocks and layers per block
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
        config['dtdnn_blocks'] = (12, 24, 16)  # default from paper
    
    # Infer embedding_dim
    if 'encoder.embedding.weight' in state_dict:
        config['embedding_dim'] = state_dict['encoder.embedding.weight'].shape[0]
    else:
        config['embedding_dim'] = 192  # default
    
    # Infer num_classes from classifier weight
    if 'weight' in state_dict:
        config['num_classes'] = state_dict['weight'].shape[0]
    elif 'classifier.weight' in state_dict:
        config['num_classes'] = state_dict['classifier.weight'].shape[0]
    else:
        config['num_classes'] = 100  # default
    
    # Infer n_mels from FCM input or input_tdnn
    # The FCM output goes to input_tdnn, and FCM does downsampling by 4 in freq dimension
    # input_tdnn input channels = fcm_out_channels = fcm_channels * (n_mels // 4)
    if 'encoder.input_tdnn.0.weight' in state_dict:
        input_channels = state_dict['encoder.input_tdnn.0.weight'].shape[1]
        fcm_ch = config['fcm_channels']
        # For FCM: output_channels = fcm_channels * (n_mels // 4)
        # So: n_mels = (input_channels // fcm_channels) * 4
        n_mels_div4 = input_channels // fcm_ch
        config['n_mels'] = n_mels_div4 * 4
    else:
        config['n_mels'] = 80  # default
    
    return config


class CAMPlusPlusDemo:
    """CAM++ Speaker Verification Demo with automatic config detection."""
    
    def __init__(self, checkpoint_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
            raise FileNotFoundError(
                "No trained model found. Please train the model first:\n"
                "  python train.py"
            )
        
        # Load checkpoint
        print(f"Loading CAM++ model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Get state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Infer model configuration from checkpoint
        inferred_config = infer_model_config_from_checkpoint(state_dict)
        
        # Check if config was saved in checkpoint and merge
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            print(f"Found saved config in checkpoint, using it...")
            if hasattr(saved_config, 'model'):
                model_cfg = saved_config.model
                if hasattr(saved_config, 'audio'):
                    inferred_config['n_mels'] = getattr(saved_config.audio, 'n_mels', inferred_config.get('n_mels', 80))
                inferred_config['embedding_dim'] = getattr(model_cfg, 'embedding_dim', inferred_config.get('embedding_dim', 192))
                inferred_config['fcm_channels'] = getattr(model_cfg, 'fcm_channels', inferred_config.get('fcm_channels', 32))
                inferred_config['fcm_num_blocks'] = getattr(model_cfg, 'fcm_num_blocks', inferred_config.get('fcm_num_blocks', 2))
                
                # Handle dtdnn_blocks - could be list or tuple
                dtdnn_blocks = getattr(model_cfg, 'dtdnn_blocks', inferred_config.get('dtdnn_blocks', (12, 24, 16)))
                if isinstance(dtdnn_blocks, list):
                    dtdnn_blocks = tuple(dtdnn_blocks)
                inferred_config['dtdnn_blocks'] = dtdnn_blocks
                
                inferred_config['growth_rate'] = getattr(model_cfg, 'growth_rate', inferred_config.get('growth_rate', 32))
                inferred_config['init_channels'] = getattr(model_cfg, 'init_channels', inferred_config.get('init_channels', 128))
                inferred_config['bn_size'] = getattr(model_cfg, 'bn_size', inferred_config.get('bn_size', 4))
        
        # Store config for preprocessor
        self.n_mels = inferred_config.get('n_mels', 80)
        self.sample_rate = 16000
        
        print(f"\nCreating model with inferred config:")
        for k, v in sorted(inferred_config.items()):
            print(f"  {k}: {v}")
        
        # Create model with inferred configuration
        # Build kwargs for CAMPlusPlus (pass through cam_kwargs)
        cam_kwargs = {
            'n_mels': inferred_config['n_mels'],
            'fcm_channels': inferred_config['fcm_channels'],
            'fcm_num_blocks': inferred_config['fcm_num_blocks'],
            'dtdnn_blocks': inferred_config['dtdnn_blocks'],
            'growth_rate': inferred_config['growth_rate'],
            'init_channels': inferred_config['init_channels'],
            'bn_size': inferred_config['bn_size'],
        }
        
        self.model = CAMPlusPlusClassifier(
            num_classes=inferred_config['num_classes'],
            embedding_dim=inferred_config['embedding_dim'],
            **cam_kwargs
        )
        
        # Load state dict
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print("\n✅ Model loaded successfully (strict mode)!")
        except RuntimeError as e:
            error_msg = str(e)
            print(f"\n⚠️ Strict loading failed. Error preview:")
            print(f"  {error_msg[:300]}...")
            
            # Try to load with strict=False to see what's missing/unexpected
            print("\nTrying non-strict loading to diagnose...")
            try:
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                
                loaded_count = len(state_dict) - len(unexpected)
                total_params = len(list(self.model.state_dict().keys()))
                
                print(f"\n📊 Loading summary:")
                print(f"  - Total model parameters: {total_params}")
                print(f"  - Loaded from checkpoint: {loaded_count}")
                print(f"  - Missing in checkpoint: {len(missing)}")
                print(f"  - Unexpected in checkpoint: {len(unexpected)}")
                
                if missing:
                    print(f"\n⚠️ Missing keys ({len(missing)}):")
                    for k in missing[:10]:
                        print(f"    - {k}")
                    if len(missing) > 10:
                        print(f"    ... and {len(missing) - 10} more")
                
                if unexpected:
                    print(f"\n⚠️ Unexpected keys ({len(unexpected)}):")
                    for k in unexpected[:10]:
                        print(f"    - {k}")
                    if len(unexpected) > 10:
                        print(f"    ... and {len(unexpected) - 10} more")
                
                # If we have size mismatches, the error will have been raised
                # Check if it's just missing/unexpected or actual size mismatches
                if "size mismatch" in error_msg:
                    print("\n❌ Size mismatches detected. Cannot proceed.")
                    raise RuntimeError(
                        f"Model architecture mismatch. The checkpoint was trained with different parameters.\n"
                        f"Inferred config: {inferred_config}\n"
                        f"Please check your model configuration."
                    )
                else:
                    print("\n✅ Model loaded with non-strict matching (some keys may be missing).")
                    
            except RuntimeError as e2:
                if "size mismatch" in str(e2):
                    print(f"\n❌ Size mismatch error: {str(e2)[:500]}")
                    raise RuntimeError(
                        f"Model architecture mismatch.\n"
                        f"Inferred config: {inferred_config}\n"
                        f"The checkpoint parameters don't match the model architecture."
                    )
                raise
        
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessor
        self.preprocessor = AudioPreprocessor(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels
        )
        
        # Speaker database
        self.enrolled_speakers = {}
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n📊 Model Statistics:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Device: {self.device}")
        print(f"  - Embedding dim: {inferred_config['embedding_dim']}")
        print(f"  - Num classes: {inferred_config['num_classes']}")
    
    def get_embedding(self, audio_path: str) -> torch.Tensor:
        """Extract speaker embedding."""
        features = self.preprocessor.process(audio_path).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.extract_embedding(features)
        return embedding
    
    def verify_speakers(
        self,
        audio1: str,
        audio2: str,
        threshold: float = 0.3
    ) -> Tuple[str, float, str]:
        """Verify if two audio samples are from the same speaker."""
        if audio1 is None or audio2 is None:
            return "❌ Please upload both audio files", 0.0, ""
        
        try:
            emb1 = self.get_embedding(audio1)
            emb2 = self.get_embedding(audio2)
            
            similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
            is_same = similarity >= threshold
            
            if is_same:
                decision = f"✅ SAME SPEAKER (Confidence: {similarity:.1%})"
            else:
                decision = f"❌ DIFFERENT SPEAKERS (Similarity: {similarity:.1%})"
            
            # Visual similarity bar
            bar_filled = int(similarity * 20)
            bar_empty = 20 - bar_filled
            similarity_bar = "🟩" * bar_filled + "⬜" * bar_empty
            
            details = f"""
### 📊 Analysis Details

**Similarity Score:** {similarity_bar} **{similarity:.1%}**

| Metric | Value |
|--------|-------|
| Cosine Similarity | {similarity:.4f} |
| Threshold | {threshold:.4f} |
| Decision | {"✅ Same Speaker" if is_same else "❌ Different Speakers"} |

---
*Higher similarity means the voices are more alike. Typical thresholds range from 0.25 to 0.35.*
            """
            
            return decision, similarity, details
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}", 0.0, ""
    
    def enroll_speaker(self, audio: str, speaker_name: str) -> str:
        """Enroll a speaker."""
        if audio is None:
            return "❌ Please upload an audio file"
        if not speaker_name or speaker_name.strip() == "":
            return "❌ Please enter a speaker name"
        
        try:
            name = speaker_name.strip()
            embedding = self.get_embedding(audio)
            
            # Check if updating existing speaker
            is_update = name in self.enrolled_speakers
            self.enrolled_speakers[name] = embedding
            
            if is_update:
                return f"🔄 Updated: **{name}**\n\nTotal enrolled: {len(self.enrolled_speakers)}"
            else:
                return f"✅ Enrolled: **{name}**\n\nTotal enrolled: {len(self.enrolled_speakers)}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}"
    
    def identify_speaker(self, audio: str, threshold: float = 0.3) -> str:
        """Identify a speaker."""
        if audio is None:
            return "❌ Please upload an audio file"
        if len(self.enrolled_speakers) == 0:
            return "❌ No speakers enrolled. Please enroll speakers first in the Enrollment tab."
        
        try:
            embedding = self.get_embedding(audio)
            
            results = []
            for name, enrolled_emb in self.enrolled_speakers.items():
                sim = torch.nn.functional.cosine_similarity(embedding, enrolled_emb).item()
                results.append((name, sim))
            
            results.sort(key=lambda x: x[1], reverse=True)
            best_name, best_score = results[0]
            
            if best_score >= threshold:
                output = f"## ✅ Identified: **{best_name}**\n\n"
                output += f"**Confidence: {best_score:.1%}**\n\n"
            else:
                output = f"## ❓ Unknown Speaker\n\n"
                output += f"Closest match: **{best_name}** ({best_score:.1%})\n\n"
                output += f"*Score below threshold ({threshold:.1%})*\n\n"
            
            output += "---\n### 📋 All Matches\n\n"
            output += "| Rank | Speaker | Similarity | Match |\n"
            output += "|------|---------|------------|-------|\n"
            
            for i, (name, score) in enumerate(results[:10], 1):
                bar = "🟩" * int(score * 10) + "⬜" * (10 - int(score * 10))
                match_icon = "✅" if score >= threshold else "❌"
                output += f"| {i} | {name} | {bar} {score:.1%} | {match_icon} |\n"
            
            if len(results) > 10:
                output += f"\n*... and {len(results) - 10} more speakers*"
            
            return output
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}"
    
    def get_enrolled_list(self) -> str:
        """Get list of enrolled speakers."""
        if len(self.enrolled_speakers) == 0:
            return "📭 No speakers enrolled yet.\n\nUse the form above to enroll speakers."
        
        output = f"### 👥 Enrolled Speakers ({len(self.enrolled_speakers)})\n\n"
        for i, name in enumerate(sorted(self.enrolled_speakers.keys()), 1):
            output += f"{i}. **{name}**\n"
        
        return output
    
    def clear_database(self) -> str:
        """Clear enrolled speakers."""
        count = len(self.enrolled_speakers)
        self.enrolled_speakers.clear()
        if count > 0:
            return f"🗑️ Cleared {count} speaker(s) from database."
        else:
            return "📭 Database was already empty."


def create_demo():
    """Create Gradio demo."""
    try:
        demo_app = CAMPlusPlusDemo()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Error loading model: {e}")
        return None
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    """
    
    with gr.Blocks(
        title="CAM++ Speaker Verification", 
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        gr.Markdown("""
        # 🎤 CAM++ Speaker Verification Demo
        
        **CAM++** - A Fast and Efficient Network for Speaker Verification 
        using Context-Aware Masking with Multi-Granularity Pooling.
        
        ---
        """)
        
        with gr.Tabs():
            # Verification Tab
            with gr.TabItem("🔍 Verification", id=1):
                gr.Markdown("### Compare Two Audio Samples\nUpload or record two audio samples to check if they're from the same speaker.")
                
                with gr.Row():
                    audio1 = gr.Audio(
                        label="🎙️ Audio 1", 
                        type="filepath", 
                        sources=["upload", "microphone"]
                    )
                    audio2 = gr.Audio(
                        label="🎙️ Audio 2", 
                        type="filepath", 
                        sources=["upload", "microphone"]
                    )
                
                with gr.Row():
                    threshold = gr.Slider(
                        minimum=0.1, 
                        maximum=0.9, 
                        value=0.3, 
                        step=0.05, 
                        label="Decision Threshold",
                        info="Higher = stricter matching"
                    )
                
                verify_btn = gr.Button("🔍 Verify Speakers", variant="primary", size="lg")
                
                with gr.Row():
                    result = gr.Textbox(label="Decision", scale=2, lines=1)
                    score = gr.Number(label="Similarity Score", precision=4)
                
                details = gr.Markdown(label="Analysis Details")
                
                verify_btn.click(
                    demo_app.verify_speakers,
                    [audio1, audio2, threshold],
                    [result, score, details]
                )
            
            # Enrollment Tab
            with gr.TabItem("📝 Enrollment", id=2):
                gr.Markdown("### Enroll Speakers\nAdd speakers to the database for later identification.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        enroll_audio = gr.Audio(
                            label="🎙️ Speaker Audio", 
                            type="filepath", 
                            sources=["upload", "microphone"]
                        )
                        speaker_name = gr.Textbox(
                            label="Speaker Name", 
                            placeholder="Enter a unique name for this speaker..."
                        )
                        enroll_btn = gr.Button("📝 Enroll Speaker", variant="primary")
                        enroll_result = gr.Markdown()
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Database")
                        enrolled_list = gr.Markdown("📭 No speakers enrolled yet.")
                        with gr.Row():
                            refresh_btn = gr.Button("🔄 Refresh", size="sm")
                            clear_btn = gr.Button("🗑️ Clear All", variant="stop", size="sm")
                
                enroll_btn.click(
                    demo_app.enroll_speaker, 
                    [enroll_audio, speaker_name], 
                    enroll_result
                ).then(
                    demo_app.get_enrolled_list, 
                    outputs=enrolled_list
                )
                refresh_btn.click(demo_app.get_enrolled_list, outputs=enrolled_list)
                clear_btn.click(
                    demo_app.clear_database, 
                    outputs=enroll_result
                ).then(
                    demo_app.get_enrolled_list, 
                    outputs=enrolled_list
                )
                
                # Load initial list
                demo.load(demo_app.get_enrolled_list, outputs=enrolled_list)
            
            # Identification Tab
            with gr.TabItem("🔎 Identification", id=3):
                gr.Markdown("### Identify Speaker\nFind who is speaking from the enrolled database.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        id_audio = gr.Audio(
                            label="🎙️ Audio to Identify", 
                            type="filepath", 
                            sources=["upload", "microphone"]
                        )
                    with gr.Column(scale=1):
                        id_threshold = gr.Slider(
                            minimum=0.1, 
                            maximum=0.9, 
                            value=0.3, 
                            step=0.05, 
                            label="Recognition Threshold",
                            info="Minimum similarity to consider a match"
                        )
                
                id_btn = gr.Button("🔎 Identify Speaker", variant="primary", size="lg")
                id_result = gr.Markdown()
                
                id_btn.click(
                    demo_app.identify_speaker, 
                    [id_audio, id_threshold], 
                    id_result
                )
            
            # About Tab
            with gr.TabItem("ℹ️ About", id=4):
                gr.Markdown("""
                ## About CAM++
                
                **CAM++** is a fast and efficient speaker verification network designed for real-world applications.
                
                ### 🏗️ Architecture Components
                
                | Component | Description |
                |-----------|-------------|
                | **FCM** | Front-end Convolution Module - 2D CNN for time-frequency processing |
                | **D-TDNN** | Densely connected Time Delay Neural Network backbone |
                | **CAM** | Context-Aware Masking - focuses on discriminative speaker features |
                | **ASP** | Attentive Statistics Pooling - captures global temporal context |
                
                ### 📊 Architecture Flow
                ```
                Audio → Mel-Spectrogram → FCM → D-TDNN (with CAM) → ASP → Embedding
                ```
                
                ### ✨ Key Features
                - ⚡ **Fast inference** - Optimized for real-time applications
                - 💾 **Low memory** - Efficient dense connections
                - 🎯 **High accuracy** - State-of-the-art on VoxCeleb benchmarks
                - 🔊 **Robust** - Handles noise and channel variations
                
                ### 📈 Performance (from paper)
                
                | Dataset | EER (%) | MinDCF |
                |---------|---------|--------|
                | VoxCeleb1-O | 0.73 | 0.0911 |
                | VoxCeleb1-E | 0.87 | 0.1020 |
                | VoxCeleb1-H | 1.63 | 0.1680 |
                
                ### 📚 Reference
                
                > Hui Wang, Siqi Zheng, Yafeng Chen, Luyao Cheng, Qian Chen.  
                > **"CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking"**  
                > *INTERSPEECH 2023*
                
                ### 🔗 Links
                - [Paper on arXiv](https://arxiv.org/abs/2303.00332)
                - [3D-Speaker Project](https://github.com/alibaba-damo-academy/3D-Speaker)
                """)
        
        gr.Markdown("""
        ---
        <center>
        <p>Made with ❤️ using <b>CAM++</b> | Speaker Verification System</p>
        </center>
        """)
    
    return demo


def main():
    demo = create_demo()
    if demo:
        print("\n" + "=" * 60)
        print("🎤 CAM++ Speaker Verification Demo")
        print("=" * 60 + "\n")
        demo.launch(
            server_name="0.0.0.0", 
            server_port=7861, 
            share=True,
            show_error=True
        )
    else:
        print("\n" + "=" * 60)
        print("❌ Failed to create demo")
        print("=" * 60)
        print("\n📋 Troubleshooting steps:")
        print("1. Make sure you have trained a model:")
        print("   python train.py")
        print("\n2. Check that a checkpoint exists:")
        print("   ls -la checkpoints/")
        print("\n3. Verify the model code matches your training:")
        print("   - Check models/cam_plus_plus.py")
        print("   - Check config/config.py")


if __name__ == "__main__":
    main()