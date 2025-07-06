from file_io import AudioLoader, Saver
from model import ModelLoader
from transcriber import Transcriber


def main():
    audio_path = "audio_files/test1.wav"

    audio_loader = AudioLoader()
    model_loader = ModelLoader()
    model, processor, device = model_loader.load()
    transcriber = Transcriber(model, processor, device)
    saver = Saver()

    audio_array, sampling_rate = audio_loader.load(audio_path)
    results = transcriber.transcribe(audio_array, sampling_rate)
    saver.save(results, "output.json")


if __name__ == "__main__":
    main()
