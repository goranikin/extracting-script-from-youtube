import yt_dlp


def download_audio_as_wav(youtube_url, output_path):
    """
    Args:
        youtube_url (str): 유튜브 영상 URL
        output_path (str): 저장할 파일 경로 (예: 'output.wav' 또는 'output.%(ext)s')
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="유튜브 영상에서 WAV 오디오 추출")
    parser.add_argument("youtube_url", type=str, help="유튜브 영상 URL")
    parser.add_argument(
        "output_path",
        type=str,
        nargs="?",
        default="output",
        help="저장할 파일 경로 (기본값: output)",
    )
    args = parser.parse_args()

    download_audio_as_wav(args.youtube_url, args.output_path)
