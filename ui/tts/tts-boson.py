from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent

import torch
import torchaudio
import time
import click

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base" # "ataraksea/higgs-audio-v2-W4A16-G128" # 
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

# Define the messages to send to the model for generation
# Simplified "prompt" structure (ChatML-like messages)
# This is conceptually what generation.py builds before generation.

system_prompt = (
    "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
)
belinda_prompt_text = "Twas the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year."  # from examples/voice_prompts/belinda.txt
belinda_prompt_wav = "belinda.wav"  # from examples/voice_prompts/belinda.wav

transcript = (
    "The sun rises in the east and sets in the west. "
    "This simple fact has been observed by humans for thousands of years."
)

messages = [
    Message(
        role="system",
        content=system_prompt,
    ),
    # Voice prompt
    Message(
        role="user",
        content=belinda_prompt_text,
    ),
    Message(
        role="assistant",
        content=AudioContent(audio_url=belinda_prompt_wav),
    ),
    # Your actual request
    Message(
        role="user",
        content=transcript,
    ),
]

device = "cuda" if torch.cuda.is_available() else "cpu"

serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

output: HiggsAudioResponse = serve_engine.generate(
    chat_ml_sample=ChatMLSample(messages=messages),
    max_new_tokens=1024,
    temperature=0.3,
    top_p=0.95,
    top_k=50,
    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
)
torchaudio.save(f"output.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)