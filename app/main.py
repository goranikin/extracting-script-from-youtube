from fastapi import FastAPI

from app.api.extract_transcript import router as transcript_router
from app.api.youtube_info import router as youtube_info_router

app = FastAPI()

app.include_router(youtube_info_router)
app.include_router(transcript_router)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
