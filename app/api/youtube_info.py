import re

import yt_dlp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(
    prefix="/youtube-info",
    tags=["youtube_info"],
)
YOUTUBE_URL_REGEX = re.compile(r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$")


class YoutubeMeta(BaseModel):
    title: str
    upload_date: str
    duration: int  # second


def is_valid_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_URL_REGEX.match(url))


@router.get("/", response_model=YoutubeMeta)
def get_youtube_info(youtube_url: str):
    if not is_valid_youtube_url(youtube_url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    try:
        with yt_dlp.YoutubeDL() as ydl:
            info = ydl.extract_info(youtube_url, download=False)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to extract YouTube info: {str(e)}"
        )

    if not isinstance(info, dict):
        raise HTTPException(
            status_code=500,
            detail="Unexpected response from YouTube extractor (not a dict).",
        )

    return YoutubeMeta(
        title=info.get("title", "Unknown Title"),
        upload_date=info.get("upload_date", "Unknown Date"),
        duration=info.get("duration", 0),
    )
