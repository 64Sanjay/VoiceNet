#!/usr/bin/env python3
"""
Run the unified speaker recognition demo.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from demo.demo_gradio import create_demo

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Speaker Recognition Demo")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎤 Speaker Recognition System - Unified Demo")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Share: {args.share}")
    print("=" * 60)
    
    demo = create_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )