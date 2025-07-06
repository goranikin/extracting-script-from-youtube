import json

import librosa
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

if torch.cuda.is_available():
    device = "cuda:0"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-tiny"

model = WhisperForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)

processor = WhisperProcessor.from_pretrained(model_id)


audio_path = "./audio_files/output_000.wav"
audio_array, sampling_rate = librosa.load(audio_path, sr=16000)

# input_features = processor(
#     audio_array, sampling_rate=sampling_rate, return_tensors="pt"
# ).input_features


# # generate token ids
# predicted_ids = model.generate(input_features)
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

# with open("result.json", "w", encoding="utf-8") as f:
# json.dump(transcription, f, ensure_ascii=False, indent=2)


chunk_size = 30 * sampling_rate  # 30초(480,000 샘플)
transcriptions = []

for start in range(0, len(audio_array), chunk_size):
    chunk = audio_array[start : start + chunk_size]
    input_features = processor(
        chunk, sampling_rate=16000, return_tensors="pt"
    ).input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    transcriptions.extend(transcription)

with open("result.json", "w", encoding="utf-8") as f:
    json.dump(transcriptions, f, ensure_ascii=False, indent=2)
