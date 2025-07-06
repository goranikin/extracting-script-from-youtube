import yt_dlp


class WavExtractor:
    def wav_extract(self, youtube_url: str, output_path: str = "output.wav"):
        """
        Extracts audio from a YouTube video and saves it as a WAV file. (16kHz)

        Args:
            youtube_url (str): The URL of the YouTube video.
            output_path (str): The path where the WAV file will be saved.
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
            "postprocessor_args": ["-ar", "16000"],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
