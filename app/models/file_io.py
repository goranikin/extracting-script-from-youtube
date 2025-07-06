import json

import librosa


class AudioLoader:
    def load(self, audio_path, sr=16000):
        audio_array, sampling_rate = librosa.load(audio_path, sr=sr)
        return audio_array, sampling_rate


class Saver:
    def save(self, transcriptions, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f, ensure_ascii=False, indent=2)
