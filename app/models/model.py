import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class ModelLoader:
    def __init__(self, model_name="openai/whisper-large-v3-turbo", device=None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.model_name = model_name

    def load(self):
        model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)  # type: ignore
        processor = WhisperProcessor.from_pretrained(self.model_name)
        return model, processor, self.device
