# demo_cli.py
"""
Simple command-line demo for WSI Speaker Identification
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import argparse
from typing import Dict

from config.config import WSIConfig
from data.preprocessing import AudioPreprocessor
from models.wsi_model import WSIModel


class CLIDemo:
    """Command-line demo for speaker verification."""
    
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = WSIConfig()
        
        # Load model
        self.model = WSIModel(
            whisper_model_name=self.config.model.whisper_model_name,
            embedding_dim=self.config.model.embedding_dim,
            projection_hidden_dim=self.config.model.projection_hidden_dim
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocessor = AudioPreprocessor(
            sample_rate=self.config.data.sample_rate,
            fixed_frames=self.config.data.fixed_input_frames,
            whisper_model_name=self.config.model.whisper_model_name
        )
        
        self.enrolled: Dict[str, torch.Tensor] = {}
    
    def get_embedding(self, audio_path: str) -> torch.Tensor:
        features = self.preprocessor.preprocess(audio_path).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model.get_embedding(features)
    
    def verify(self, audio1: str, audio2: str, threshold: float = 0.3) -> None:
        emb1 = self.get_embedding(audio1)
        emb2 = self.get_embedding(audio2)
        similarity = self.model.compute_similarity(emb1, emb2).item()
        
        print(f"\n{'='*50}")
        print(f"Audio 1: {audio1}")
        print(f"Audio 2: {audio2}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Threshold: {threshold:.4f}")
        print(f"Result: {'✅ SAME SPEAKER' if similarity >= threshold else '❌ DIFFERENT SPEAKERS'}")
        print(f"{'='*50}\n")
    
    def enroll(self, audio: str, name: str) -> None:
        self.enrolled[name] = self.get_embedding(audio)
        print(f"✅ Enrolled: {name}")
    
    def identify(self, audio: str, threshold: float = 0.3) -> None:
        if not self.enrolled:
            print("❌ No speakers enrolled")
            return
        
        emb = self.get_embedding(audio)
        scores = {}
        
        for name, enrolled_emb in self.enrolled.items():
            sim = self.model.compute_similarity(emb, enrolled_emb).item()
            scores[name] = sim
        
        best = max(scores, key=scores.get)
        print(f"\n{'='*50}")
        print(f"Audio: {audio}")
        print(f"Best match: {best} ({scores[best]:.4f})")
        print(f"Result: {'✅ IDENTIFIED' if scores[best] >= threshold else '❓ UNKNOWN'}")
        print(f"{'='*50}\n")
    
    def interactive(self):
        """Interactive CLI mode."""
        print("\n🎤 WSI Speaker Identification - Interactive Mode")
        print("="*50)
        print("Commands:")
        print("  v <audio1> <audio2>  - Verify two speakers")
        print("  e <audio> <name>     - Enroll speaker")
        print("  i <audio>            - Identify speaker")
        print("  l                    - List enrolled")
        print("  q                    - Quit")
        print("="*50)
        
        while True:
            try:
                cmd = input("\n> ").strip().split()
                if not cmd:
                    continue
                
                if cmd[0] == 'q':
                    break
                elif cmd[0] == 'v' and len(cmd) >= 3:
                    self.verify(cmd[1], cmd[2])
                elif cmd[0] == 'e' and len(cmd) >= 3:
                    self.enroll(cmd[1], ' '.join(cmd[2:]))
                elif cmd[0] == 'i' and len(cmd) >= 2:
                    self.identify(cmd[1])
                elif cmd[0] == 'l':
                    print(f"Enrolled: {list(self.enrolled.keys())}")
                else:
                    print("Invalid command")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="WSI CLI Demo")
    parser.add_argument('--checkpoint', default='outputs_v2/best_model.pt')
    parser.add_argument('--mode', choices=['interactive', 'verify', 'enroll', 'identify'], default='interactive')
    parser.add_argument('--audio1', help='First audio file')
    parser.add_argument('--audio2', help='Second audio file')
    parser.add_argument('--name', help='Speaker name for enrollment')
    parser.add_argument('--threshold', type=float, default=0.3)
    args = parser.parse_args()
    
    # Find checkpoint
    if not Path(args.checkpoint).exists():
        args.checkpoint = 'outputs/final_model.pt'
    
    if not Path(args.checkpoint).exists():
        print("❌ No model found. Train first with: python run_training_v2.py")
        return
    
    demo = CLIDemo(args.checkpoint)
    
    if args.mode == 'interactive':
        demo.interactive()
    elif args.mode == 'verify':
        demo.verify(args.audio1, args.audio2, args.threshold)
    elif args.mode == 'enroll':
        demo.enroll(args.audio1, args.name)
    elif args.mode == 'identify':
        demo.identify(args.audio1, args.threshold)


if __name__ == "__main__":
    main()