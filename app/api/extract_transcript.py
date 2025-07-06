import os
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.youtube_info import is_valid_youtube_url
from app.models.file_io import AudioLoader
from app.models.model import ModelLoader
from app.models.transcriber import Transcriber
from app.models.wav_extractor import WavExtractor

router = APIRouter(
    prefix="/extract-transcript",
    tags=["transcript"],
)


class Transcript(BaseModel):
    youtube_url: str
    transcript: str


def extract_audio(youtube_url, output_path):
    wav_extractor = WavExtractor()
    wav_extractor.wav_extract(youtube_url, output_path)


def transcribe_audio(audio_path):
    audio_loader = AudioLoader()
    model, processor, device = ModelLoader().load()
    transcriber = Transcriber(model, processor, device)
    audio_array, sampling_rate = audio_loader.load(audio_path)
    return transcriber.transcribe(audio_array, sampling_rate)


@router.get("/", response_model=Transcript)
def get_transcript(youtube_url: str):
    if not is_valid_youtube_url(youtube_url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    base_output_path = "./data/" + datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        extract_audio(youtube_url, base_output_path)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to extract audio: {str(e)}"
        )
    try:
        results = transcribe_audio(base_output_path + ".wav")
        transcript_str = " ".join(results)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to transcribe audio: {str(e)}"
        )

    return Transcript(
        youtube_url=youtube_url,
        transcript=transcript_str,
    )
