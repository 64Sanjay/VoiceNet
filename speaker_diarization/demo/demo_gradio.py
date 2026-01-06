#!/usr/bin/env python3
"""
Interactive Gradio demo for speaker diarization.

Usage:
    python demo/demo_gradio.py --model-path ./outputs/checkpoints/best_model.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional
import tempfile
import json

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not installed. Run: pip install gradio")

from speaker_diarization.models import load_pretrained_model, DiarizationPipeline
from speaker_diarization.utils.audio_utils import AudioProcessor


# Global variables for model
pipeline = None
audio_processor = None


def load_model(model_path: str, device: str = "cpu"):
    """Load the diarization model."""
    global pipeline, audio_processor
    
    model = load_pretrained_model(model_path, device=device)
    
    pipeline = DiarizationPipeline(
        model=model,
        clustering_method="ahc",
        clustering_threshold=0.5,
        device=device,
    )
    
    audio_processor = AudioProcessor(sample_rate=model.sample_rate)
    
    return f"Model loaded successfully on {device}!"


def format_time(seconds: float) -> str:
    """Format seconds to MM:SS.mmm string."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def create_timeline_html(
    segments: list,
    duration: float,
    colors: dict,
) -> str:
    """Create HTML timeline visualization."""
    html = '<div style="position: relative; height: 60px; background: #f0f0f0; border-radius: 5px; margin: 10px 0;">'
    
    for start, end, speaker in segments:
        left_pct = (start / duration) * 100
        width_pct = ((end - start) / duration) * 100
        color = colors.get(speaker, "#808080")
        
        html += f'''
        <div style="
            position: absolute;
            left: {left_pct}%;
            width: {width_pct}%;
            height: 100%;
            background: {color};
            opacity: 0.8;
            border-radius: 3px;
        " title="{speaker}: {format_time(start)} - {format_time(end)}"></div>
        '''
    
    html += '</div>'
    
    # Add legend
    html += '<div style="display: flex; gap: 20px; margin-top: 5px;">'
    for speaker, color in colors.items():
        html += f'''
        <div style="display: flex; align-items: center; gap: 5px;">
            <div style="width: 15px; height: 15px; background: {color}; border-radius: 3px;"></div>
            <span>{speaker}</span>
        </div>
        '''
    html += '</div>'
    
    return html


def diarize_audio(
    audio_input,
    num_speakers: Optional[int] = None,
    clustering_threshold: float = 0.5,
    min_segment_duration: float = 0.1,
) -> Tuple[str, str, str]:
    """
    Run speaker diarization on uploaded audio.
    
    Args:
        audio_input: Audio file or (sample_rate, audio_array) tuple
        num_speakers: Number of speakers (None for auto-detection)
        clustering_threshold: Clustering threshold
        min_segment_duration: Minimum segment duration
        
    Returns:
        Tuple of (results_text, timeline_html, json_output)
    """
    global pipeline, audio_processor
    
    if pipeline is None:
        return "Error: Model not loaded!", "", ""
    
    try:
        # Handle different input formats
        if isinstance(audio_input, tuple):
            # From microphone: (sample_rate, audio_array)
            sample_rate, audio_array = audio_input
            waveform = torch.from_numpy(audio_array).float()
            
            # Normalize
            if waveform.abs().max() > 1:
                waveform = waveform / 32768.0
            
            # Handle stereo
            if waveform.dim() == 2:
                waveform = waveform.mean(dim=1)
            
            waveform = waveform.unsqueeze(0)
            
        else:
            # From file upload
            waveform, sample_rate = audio_processor.load(audio_input)
        
        duration = waveform.shape[-1] / sample_rate
        
        # Update pipeline settings
        pipeline.clustering.threshold = clustering_threshold
        pipeline.min_segment_duration = min_segment_duration
        
        # Run diarization
        num_spk = int(num_speakers) if num_speakers and num_speakers > 0 else None
        
        segments = pipeline(
            waveform,
            sample_rate=sample_rate,
            num_speakers=num_spk,
        )
        
        # Get unique speakers
        speakers = sorted(set(s[2] for s in segments))
        
        # Assign colors
        colors = {}
        color_palette = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
            "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
            "#BB8FCE", "#85C1E9",
        ]
        for i, speaker in enumerate(speakers):
            colors[speaker] = color_palette[i % len(color_palette)]
        
        # Format results
        results_lines = [
            f"📊 **Diarization Results**",
            f"",
            f"**Audio Duration:** {format_time(duration)}",
            f"**Number of Speakers:** {len(speakers)}",
            f"**Number of Segments:** {len(segments)}",
            f"",
            f"---",
            f"",
            f"**Timeline:**",
        ]
        
        for start, end, speaker in sorted(segments, key=lambda x: x[0]):
            dur = end - start
            results_lines.append(
                f"• `{format_time(start)}` → `{format_time(end)}` | **{speaker}** ({dur:.2f}s)"
            )
        
        results_lines.extend([
            f"",
            f"---",
            f"",
            f"**Speaker Statistics:**",
        ])
        
        # Compute per-speaker stats
        speaker_durations = {}
        for start, end, speaker in segments:
            if speaker not in speaker_durations:
                speaker_durations[speaker] = 0.0
            speaker_durations[speaker] += end - start
        
        total_speech = sum(speaker_durations.values())
        
        for speaker in speakers:
            dur = speaker_durations.get(speaker, 0)
            pct = 100 * dur / total_speech if total_speech > 0 else 0
            results_lines.append(f"• **{speaker}:** {dur:.2f}s ({pct:.1f}%)")
        
        results_text = "\n".join(results_lines)
        
        # Create timeline
        timeline_html = create_timeline_html(segments, duration, colors)
        
        # Create JSON output
        json_output = {
            "duration": round(duration, 3),
            "num_speakers": len(speakers),
            "segments": [
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "speaker": speaker,
                }
                for start, end, speaker in segments
            ],
            "speaker_stats": {
                speaker: {
                    "duration": round(dur, 3),
                    "percentage": round(100 * dur / total_speech, 2),
                }
                for speaker, dur in speaker_durations.items()
            }
        }
        
        return results_text, timeline_html, json.dumps(json_output, indent=2)
        
    except Exception as e:
        error_msg = f"Error during diarization: {str(e)}"
        return error_msg, "", ""


def create_demo():
    """Create Gradio demo interface."""
    
    with gr.Blocks(
        title="Speaker Diarization",
        theme=gr.themes.Soft(),
    ) as demo:
        
        gr.Markdown(
            """
            # 🎙️ Speaker Diarization Demo
            
            Upload an audio file or record from your microphone to identify **who spoke when**.
            
            This system uses neural speaker embeddings and clustering to segment audio by speaker.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")
                
                audio_input = gr.Audio(
                    label="Audio Input",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    num_speakers = gr.Number(
                        label="Number of Speakers (0 = auto-detect)",
                        value=0,
                        minimum=0,
                        maximum=20,
                        step=1,
                    )
                    
                    clustering_threshold = gr.Slider(
                        label="Clustering Threshold",
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                        info="Lower = more speakers, Higher = fewer speakers",
                    )
                    
                    min_segment_duration = gr.Slider(
                        label="Minimum Segment Duration (s)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.1,
                        step=0.05,
                    )
                
                submit_btn = gr.Button("🚀 Run Diarization", variant="primary")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")
                
                timeline_output = gr.HTML(label="Timeline")
                
                results_output = gr.Markdown(label="Details")
                
                with gr.Accordion("📋 JSON Output", open=False):
                    json_output = gr.Code(
                        label="JSON",
                        language="json",
                    )
        
        # Examples
        gr.Markdown("### 📂 Examples")
        gr.Examples(
            examples=[
                ["examples/meeting_2speakers.wav", 2, 0.5, 0.1],
                ["examples/interview_3speakers.wav", 0, 0.5, 0.1],
                ["examples/podcast_2speakers.wav", 2, 0.6, 0.2],
            ],
            inputs=[audio_input, num_speakers, clustering_threshold, min_segment_duration],
            outputs=[results_output, timeline_output, json_output],
            fn=diarize_audio,
            cache_examples=False,
        )
        
        # Event handlers
        submit_btn.click(
            fn=diarize_audio,
            inputs=[
                audio_input,
                num_speakers,
                clustering_threshold,
                min_segment_duration,
            ],
            outputs=[results_output, timeline_output, json_output],
        )
        
        gr.Markdown(
            """
            ---
            
            ### 📖 How it works
            
            1. **Feature Extraction:** Audio is converted to mel-spectrogram features
            2. **Segmentation:** Neural network predicts frame-level speaker activity
            3. **Embedding:** Speaker embeddings are extracted for each segment
            4. **Clustering:** Embeddings are clustered to assign speaker identities
            5. **Post-processing:** Short segments are removed and gaps are merged
            
            ### 🔗 Resources
            
            - [Paper: AISHELL-4](https://arxiv.org/abs/2104.03603)
            - [Paper: ECAPA-TDNN](https://arxiv.org/abs/2005.07143)
            - [Paper: PyanNet](https://arxiv.org/abs/2104.04045)
            """
        )
    
    return demo


def main():
    if not GRADIO_AVAILABLE:
        print("Gradio not installed. Run: pip install gradio")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="Speaker Diarization Gradio Demo"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on",
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    status = load_model(args.model_path, args.device)
    print(status)
    
    # Create and launch demo
    demo = create_demo()
    demo.launch(
        share=args.share,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()