# run_demo.py
"""
Simple launcher for WSI demos
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("="*60)
    print("🎤 WSI Speaker Identification - Demo Launcher")
    print("="*60)
    print("\nAvailable demos:")
    print("1. Gradio Web Demo (Recommended)")
    print("2. Streamlit Dashboard")
    print("3. FastAPI REST API")
    print("4. Command-line Demo")
    print("="*60)
    
    choice = input("\nSelect demo (1-4) [default=1]: ").strip() or "1"
    
    if choice == "1":
        print("\nLaunching Gradio demo...")
        try:
            import gradio as gr
        except ImportError:
            print("Installing gradio...")
            os.system("pip install gradio")
        
        from demo.demo_gradio import main as gradio_main
        gradio_main()
        
    elif choice == "2":
        print("\nLaunching Streamlit demo...")
        try:
            import streamlit
        except ImportError:
            print("Installing streamlit...")
            os.system("pip install streamlit plotly")
        
        os.system("streamlit run demo/demo_streamlit.py")
        
    elif choice == "3":
        print("\nLaunching FastAPI server...")
        try:
            import fastapi
        except ImportError:
            print("Installing fastapi...")
            os.system("pip install fastapi uvicorn python-multipart")
        
        os.system("python demo/demo_api.py")
        
    elif choice == "4":
        print("\nLaunching CLI demo...")
        os.system("python demo/demo_cli.py --mode interactive")
        
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()