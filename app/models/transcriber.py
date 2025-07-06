class Transcriber:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def transcribe(
        self, audio_array, sampling_rate: int | float = 16000, chunk_sec=30
    ) -> list[str]:
        chunk_size = int(chunk_sec * sampling_rate)
        transcriptions = []
        for start in range(0, len(audio_array), chunk_size):
            chunk = audio_array[start : start + chunk_size]
            input_features = self.processor(
                chunk, sampling_rate=sampling_rate, return_tensors="pt"
            ).input_features.to(self.device)
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )
            transcriptions.extend(transcription)
        return transcriptions
