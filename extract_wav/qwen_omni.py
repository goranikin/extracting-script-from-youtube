from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

wav_path = "./test1.wav"

conversation = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "audio": wav_path,
            },
            {
                "type": "text",
                "text": "Write a script with this audio file.",  # 명확하게 음성 인식 요청
            },
        ],
    },
]

USE_AUDIO_IN_VIDEO = False

# Preparation for inference
text = processor.apply_chat_template(
    conversation, add_generation_prompt=True, tokenize=False
)

# 이유는 모르는데 pyright이 자꾸 반환값 4개라 우김
audios, images, videos, _ = process_mm_info(
    conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
)
inputs = processor(
    text=text,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="pt",
    padding=True,
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
)
inputs = inputs.to(model.device).to(model.dtype)

text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

text = processor.batch_decode(
    text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(text)
