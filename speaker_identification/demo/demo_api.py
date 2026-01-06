# demo_api.py
"""
FastAPI REST API for WSI Speaker Identification
Production-ready API for integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import tempfile
import os
import uvicorn

# Import WSI modules
from config.config import WSIConfig
from data.preprocessing import AudioPreprocessor
from models.wsi_model import WSIModel


# Initialize FastAPI
app = FastAPI(
    title="WSI Speaker Identification API",
    description="REST API for speaker verification and identification using Whisper embeddings",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
preprocessor = None
device = None
enrolled_speakers: Dict[str, torch.Tensor] = {}


class VerificationResponse(BaseModel):
    similarity: float
    is_same_speaker: bool
    threshold: float
    confidence: str


class EnrollmentResponse(BaseModel):
    success: bool
    speaker_name: str
    message: str
    total_enrolled: int


class IdentificationResponse(BaseModel):
    identified_speaker: Optional[str]
    confidence: float
    is_known: bool
    all_scores: Dict[str, float]


class SpeakerListResponse(BaseModel):
    speakers: List[str]
    count: int


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, preprocessor, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = WSIConfig()
    
    # Load model
    model = WSIModel(
        whisper_model_name=config.model.whisper_model_name,
        embedding_dim=config.model.embedding_dim,
        projection_hidden_dim=config.model.projection_hidden_dim
    )
    
    checkpoint_paths = ["outputs_v2/best_model.pt", "outputs/final_model.pt"]
    checkpoint_path = None
    
    for path in checkpoint_paths:
        if Path(path).exists():
            checkpoint_path = path
            break
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"✅ Model loaded from {checkpoint_path}")
    else:
        print("⚠️ No checkpoint found, using random weights")
        model.to(device)
        model.eval()
    
    preprocessor = AudioPreprocessor(
        sample_rate=config.data.sample_rate,
        fixed_frames=config.data.fixed_input_frames,
        whisper_model_name=config.model.whisper_model_name
    )


def get_embedding(audio_path: str) -> torch.Tensor:
    """Extract speaker embedding from audio file."""
    features = preprocessor.preprocess(audio_path).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.get_embedding(features)
    return embedding


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "WSI Speaker Identification API",
        "version": "1.0.0",
        "endpoints": {
            "/verify": "POST - Verify if two audio files are from same speaker",
            "/enroll": "POST - Enroll a new speaker",
            "/identify": "POST - Identify speaker from enrolled database",
            "/speakers": "GET - List enrolled speakers",
            "/speakers/{name}": "DELETE - Remove enrolled speaker"
        }
    }


@app.post("/verify", response_model=VerificationResponse)
async def verify_speakers(
    audio1: UploadFile = File(...),
    audio2: UploadFile = File(...),
    threshold: float = 0.3
):
    """Verify if two audio files are from the same speaker."""
    
    # Save uploaded files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f1:
        f1.write(await audio1.read())
        path1 = f1.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f2:
        f2.write(await audio2.read())
        path2 = f2.name
    
    try:
        # Get embeddings
        emb1 = get_embedding(path1)
        emb2 = get_embedding(path2)
        
        # Compute similarity
        similarity = model.compute_similarity(emb1, emb2).item()
        is_same = similarity >= threshold
        
        # Confidence level
        if similarity > 0.7:
            confidence = "high"
        elif similarity > 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        return VerificationResponse(
            similarity=similarity,
            is_same_speaker=is_same,
            threshold=threshold,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        os.unlink(path1)
        os.unlink(path2)


@app.post("/enroll", response_model=EnrollmentResponse)
async def enroll_speaker(
    audio: UploadFile = File(...),
    speaker_name: str = Form(...)
):
    """Enroll a new speaker in the database."""
    
    if not speaker_name.strip():
        raise HTTPException(status_code=400, detail="Speaker name cannot be empty")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
        f.write(await audio.read())
        path = f.name
    
    try:
        embedding = get_embedding(path)
        enrolled_speakers[speaker_name] = embedding.cpu()
        
        return EnrollmentResponse(
            success=True,
            speaker_name=speaker_name,
            message=f"Successfully enrolled {speaker_name}",
            total_enrolled=len(enrolled_speakers)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        os.unlink(path)


@app.post("/identify", response_model=IdentificationResponse)
async def identify_speaker(
    audio: UploadFile = File(...),
    threshold: float = 0.3
):
    """Identify a speaker from the enrolled database."""
    
    if not enrolled_speakers:
        raise HTTPException(status_code=400, detail="No speakers enrolled")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
        f.write(await audio.read())
        path = f.name
    
    try:
        embedding = get_embedding(path)
        
        scores = {}
        for name, enrolled_emb in enrolled_speakers.items():
            enrolled_emb = enrolled_emb.to(device)
            sim = model.compute_similarity(embedding, enrolled_emb).item()
            scores[name] = sim
        
        best_speaker = max(scores, key=scores.get)
        best_score = scores[best_speaker]
        is_known = best_score >= threshold
        
        return IdentificationResponse(
            identified_speaker=best_speaker if is_known else None,
            confidence=best_score,
            is_known=is_known,
            all_scores=scores
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        os.unlink(path)


@app.get("/speakers", response_model=SpeakerListResponse)
async def list_speakers():
    """List all enrolled speakers."""
    return SpeakerListResponse(
        speakers=list(enrolled_speakers.keys()),
        count=len(enrolled_speakers)
    )


@app.delete("/speakers/{speaker_name}")
async def remove_speaker(speaker_name: str):
    """Remove an enrolled speaker."""
    if speaker_name not in enrolled_speakers:
        raise HTTPException(status_code=404, detail="Speaker not found")
    
    del enrolled_speakers[speaker_name]
    return {"message": f"Removed {speaker_name}", "remaining": len(enrolled_speakers)}


@app.delete("/speakers")
async def clear_all_speakers():
    """Remove all enrolled speakers."""
    count = len(enrolled_speakers)
    enrolled_speakers.clear()
    return {"message": f"Cleared {count} speakers"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)