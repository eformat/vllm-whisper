import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import threading

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
# A chunk size that balances latency and processing overhead (e.g., 5 seconds of audio data)
CHUNK_DURATION = 5
BLOCK_SIZE = SAMPLE_RATE * CHUNK_DURATION

audio_queue = queue.Queue()
model = WhisperModel("base.en", device="cpu", compute_type="int8") # Use "cuda" if GPU is available

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status)
    audio_queue.put(bytes(indata))

def transcribe_audio():
    """Processes audio chunks from the queue in real-time."""
    full_audio_buffer = b""
    print("Listening... Say something!")

    with sd.InputStream(callback=audio_callback, dtype=DTYPE, channels=CHANNELS, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
        while True:
            full_audio_buffer += audio_queue.get()
            # Process the buffer when it reaches a certain length or based on VAD
            # A common approach is to process in sliding windows

            # In a real application, you would use VAD (Voice Activity Detection)
            # to determine when a speech segment is complete before transcribing.
            # A simpler approach for demonstration is to process fixed-size chunks:

            if len(full_audio_buffer) >= BLOCK_SIZE * 2: # Process every 10 seconds of accumulated audio
                # Convert bytes to float32 numpy array
                audio_np = np.frombuffer(full_audio_buffer, dtype=DTYPE).astype(np.float32) / 32768.0
                
                # Transcribe the chunk
                segments, info = model.transcribe(audio_np, beam_size=5)
                for segment in segments:
                    print(f"[ {segment.start:.2f}s -> {segment.end:.2f}s ] {segment.text}")
                
                # Clear the processed audio (keep the end for context if needed)
                full_audio_buffer = b""

# Start the transcription loop (this will run indefinitely)
try:
    transcribe_audio()
except KeyboardInterrupt:
    print("\nTranscription stopped.")
