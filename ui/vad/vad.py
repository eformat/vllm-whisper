import torch
from pprint import pprint

# Set the number of CPU threads if needed
torch.set_num_threads(1)

# Load the VAD model and utils
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

sampling_rate = 16000 # or 8000

# Read an audio file (replace 'your_audio.wav')
try:
    wav = read_audio('testaudio_16000_test01_20s.wav', sampling_rate=sampling_rate)
except Exception as e:
    print(f"Error reading audio: {e}")
    print("Ensure torchaudio backends (e.g., FFmpeg) are installed")
    exit()

# Get speech timestamps
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)

print("Detected speech timestamps:")
pprint(speech_timestamps)
