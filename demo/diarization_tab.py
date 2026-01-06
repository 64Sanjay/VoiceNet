#!/usr/bin/env python3
"""Speaker Diarization Tab for Unified Demo"""

import gradio as gr
import torch
import numpy as np
import os
import sys
from typing import List, Tuple

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'speaker_diarization'))

from speaker_diarization.config.config import Config as DiarizationConfig
from speaker_diarization.models.diarization_model import DiarizationModel
from speaker_diarization.models.clustering import SpectralClustering
from speaker_diarization.utils.audio_utils import load_audio
from speaker_diarization.utils.rttm_utils import segments_to_rttm


class SpeakerDiarizationEngine:
    """Engine for speaker diarization inference."""
    
    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = DiarizationConfig()
        
        # Initialize model
        self.model = self._load_model(checkpoint_path)
        self.clustering = SpectralClustering()
        
        # Diarization parameters
        self.segment_duration = 1.5  # seconds
        self.hop_duration = 0.75  # seconds
        self.min_speakers = 1
        self.max_speakers = 10
    
    def _load_model(self, checkpoint_path):
        """Load the diarization model."""
        model = DiarizationModel(self.config).to(self.device)
        
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
    
    def diarize(self, audio_path, num_speakers=None):
        """Perform speaker diarization on audio file."""
        # Load audio
        waveform, sr = load_audio(audio_path, return_sr=True)
        duration = len(waveform) / sr
        
        # Segment audio
        segments, segment_times = self._segment_audio(waveform, sr)
        
        # Extract embeddings for each segment
        embeddings = self._extract_segment_embeddings(segments, sr)
        
        # Cluster embeddings
        if num_speakers is not None:
            labels = self.clustering.cluster(embeddings, n_clusters=num_speakers)
        else:
            labels = self.clustering.cluster_auto(
                embeddings,
                min_clusters=self.min_speakers,
                max_clusters=self.max_speakers
            )
        
        # Create diarization output
        diarization_result = self._create_diarization_output(
            segment_times, labels, duration
        )
        
        return diarization_result
    
    def _segment_audio(self, waveform, sr):
        """Segment audio into overlapping windows."""
        segment_samples = int(self.segment_duration * sr)
        hop_samples = int(self.hop_duration * sr)
        
        segments = []
        segment_times = []
        
        start = 0
        while start + segment_samples <= len(waveform):
            segment = waveform[start:start + segment_samples]
            segments.append(segment)
            
            start_time = start / sr
            end_time = (start + segment_samples) / sr
            segment_times.append((start_time, end_time))
            
            start += hop_samples
        
        return segments, segment_times
    
    def _extract_segment_embeddings(self, segments, sr):
        """Extract embeddings for all segments."""
        embeddings = []
        
        for segment in segments:
            waveform = torch.FloatTensor(segment).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.extract_embedding(waveform)
            
            embeddings.append(embedding.cpu().numpy().flatten())
        
        return np.stack(embeddings)
    
    def _create_diarization_output(self, segment_times, labels, total_duration):
        """Create structured diarization output."""
        # Merge consecutive segments with same speaker
        merged_segments = []
        current_speaker = labels[0]
        current_start = segment_times[0][0]
        current_end = segment_times[0][1]
        
        for i in range(1, len(labels)):
            if labels[i] == current_speaker:
                current_end = segment_times[i][1]
            else:
                merged_segments.append({
                    'speaker': f"Speaker_{current_speaker + 1}",
                    'start': current_start,
                    'end': current_end
                })
                current_speaker = labels[i]
                current_start = segment_times[i][0]
                current_end = segment_times[i][1]
        
        # Add last segment
        merged_segments.append({
            'speaker': f"Speaker_{current_speaker + 1}",
            'start': current_start,
            'end': current_end
        })
        
        # Compute statistics
        unique_speakers = set(labels)
        num_speakers = len(unique_speakers)
        
        speaker_times = {}
        for seg in merged_segments:
            speaker = seg['speaker']
            duration = seg['end'] - seg['start']
            speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
        
        return {
            'segments': merged_segments,
            'num_speakers': num_speakers,
            'total_duration': total_duration,
            'speaker_times': speaker_times
        }


# Global engine instance
diarization_engine = None


def get_diarization_engine():
    """Get or create diarization engine."""
    global diarization_engine
    if diarization_engine is None:
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'speaker_diarization', 'outputs', 'diarization_20260105_092337', 'best_model.pt'
        )
        diarization_engine = SpeakerDiarizationEngine(checkpoint_path)
    return diarization_engine


def format_time(seconds):
    """Format seconds to MM:SS.ms format."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def diarize_audio(audio, num_speakers):
    """Gradio function for speaker diarization."""
    if audio is None:
        return "⚠️ Please upload an audio file", "", None
    
    try:
        engine = get_diarization_engine()
        
        # Set number of speakers (None for auto-detection)
        n_speakers = int(num_speakers) if num_speakers and num_speakers > 0 else None
        
        result = engine.diarize(audio, num_speakers=n_speakers)
        
        # Create timeline visualization
        timeline_html = create_timeline_html(result)
        
        # Create text summary
        summary = create_summary(result)
        
        # Create RTTM-style output
        rttm_output = create_rttm_output(result, audio)
        
        return timeline_html, summary, rttm_output
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", "", None


def create_timeline_html(result):
    """Create HTML timeline visualization."""
    segments = result['segments']
    total_duration = result['total_duration']
    
    # Color palette for speakers
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
        '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
        '#BB8FCE', '#85C1E9'
    ]
    
    # Create speaker color mapping
    speakers = list(result['speaker_times'].keys())
    speaker_colors = {s: colors[i % len(colors)] for i, s in enumerate(speakers)}
    
    html = """
    <div style="padding: 20px;">
        <h3>📊 Diarization Timeline</h3>
        <div style="position: relative; height: 60px; background: #f0f0f0; border-radius: 10px; margin: 10px 0;">
    """
    
    for seg in segments:
        start_pct = (seg['start'] / total_duration) * 100
        width_pct = ((seg['end'] - seg['start']) / total_duration) * 100
        color = speaker_colors.get(seg['speaker'], '#999')
        
        html += f"""
        <div style="position: absolute; left: {start_pct}%; width: {width_pct}%; 
                    height: 100%; background: {color}; border-radius: 5px;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 12px; color: white; font-weight: bold; overflow: hidden;"
             title="{seg['speaker']}: {format_time(seg['start'])} - {format_time(seg['end'])}">
            {seg['speaker'].split('_')[1] if '_' in seg['speaker'] else seg['speaker']}
        </div>
        """
    
    html += "</div>"
    
    # Add legend
    html += "<div style='margin-top: 10px;'><strong>Legend:</strong> "
    for speaker, color in speaker_colors.items():
        html += f'<span style="background: {color}; padding: 2px 10px; margin: 2px; border-radius: 3px; color: white;">{speaker}</span> '
    html += "</div>"
    
    # Add time markers
    html += """
    <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 12px; color: #666;">
        <span>0:00</span>
        <span>{}</span>
    </div>
    """.format(format_time(total_duration))
    
    html += "</div>"
    
    return html


def create_summary(result):
    """Create text summary of diarization."""
    segments = result['segments']
    num_speakers = result['num_speakers']
    total_duration = result['total_duration']
    speaker_times = result['speaker_times']
    
    summary = f"""
### 📋 Diarization Summary

**Number of Speakers Detected:** {num_speakers}

**Total Duration:** {format_time(total_duration)}

### ⏱️ Speaking Time per Speaker
"""
    
    for speaker, time in sorted(speaker_times.items()):
        percentage = (time / total_duration) * 100
        bar_length = int(percentage / 5)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        summary += f"- **{speaker}**: {format_time(time)} ({percentage:.1f}%) |{bar}|\n"
    
    summary += """
### 📝 Segment Details
| Speaker | Start | End | Duration |
|---------|-------|-----|----------|
"""
    
    for seg in segments[:20]:  # Limit to first 20 segments
        duration = seg['end'] - seg['start']
        summary += f"| {seg['speaker']} | {format_time(seg['start'])} | {format_time(seg['end'])} | {duration:.2f}s |\n"
    
    if len(segments) > 20:
        summary += f"\n*...and {len(segments) - 20} more segments*\n"
    
    return summary


def create_rttm_output(result, audio_path):
    """Create RTTM format output."""
    filename = os.path.basename(audio_path).replace('.wav', '').replace('.mp3', '')
    
    rttm_lines = []
    for seg in result['segments']:
        # RTTM format: SPEAKER file 1 start duration <NA> <NA> speaker <NA> <NA>
        duration = seg['end'] - seg['start']
        line = f"SPEAKER {filename} 1 {seg['start']:.3f} {duration:.3f} <NA> <NA> {seg['speaker']} <NA> <NA>"
        rttm_lines.append(line)
    
    return "\n".join(rttm_lines)


def create_diarization_tab():
    """Create the speaker diarization tab interface."""
    
    gr.Markdown("""
    ## 👥 Speaker Diarization
    
    Segment an audio file by speaker - find out "who spoke when".
    
    **How it works:**
    1. Upload an audio file with multiple speakers
    2. Optionally specify the number of speakers (or let the system detect)
    3. Click "Diarize" to segment the audio
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="📎 Upload Audio File",
                type="filepath",
                sources=["upload"]
            )
            
            num_speakers = gr.Slider(
                minimum=0,
                maximum=10,
                value=0,
                step=1,
                label="Number of Speakers (0 = Auto-detect)",
                info="Set to 0 for automatic speaker count detection"
            )
            
            diarize_btn = gr.Button("👥 Diarize Audio", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            timeline_html = gr.HTML(label="Timeline Visualization")
    
    summary_output = gr.Markdown(label="Diarization Summary")
    
    with gr.Accordion("📄 RTTM Output", open=False):
        rttm_output = gr.Textbox(
            label="RTTM Format",
            lines=10,
            interactive=False
        )
    
    diarize_btn.click(
        fn=diarize_audio,
        inputs=[audio_input, num_speakers],
        outputs=[timeline_html, summary_output, rttm_output]
    )
    
    gr.Markdown("""
    ### 💡 Tips
    - For best results, use clean audio with minimal background noise
    - If speaker count is known, specify it for more accurate results
    - Meeting recordings, interviews, and podcasts work well
    
    ### 📄 Output Formats
    - **Timeline**: Visual representation of speaker segments
    - **Summary**: Statistics about each speaker's speaking time
    - **RTTM**: Standard format for diarization evaluation
    """)