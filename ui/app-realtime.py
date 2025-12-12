import streamlit as st
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from scipy.signal import resample_poly
import queue
import threading
import time
from pathlib import Path

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

# Duration (in seconds) of audio to accumulate before transcribing.
# Smaller values => lower latency, higher overhead.
CHUNK_DURATION = 5
BLOCK_SIZE = SAMPLE_RATE * CHUNK_DURATION


def resample_to_sample_rate(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio from orig_sr to target_sr using polyphase filtering.
    Works for mono or multi-channel arrays; resamples along the last axis.
    """
    if orig_sr == target_sr:
        return audio

    # Compute integer up/down factors from the sample-rate ratio
    from math import gcd

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    # Ensure we operate in float for resampling, preserve original dtype afterwards if needed
    orig_dtype = audio.dtype
    audio_f = audio.astype(np.float32)
    audio_resampled = resample_poly(audio_f, up, down, axis=-1)

    # If original was integer PCM (e.g., int16), round and cast back
    if np.issubdtype(orig_dtype, np.integer):
        audio_resampled = np.clip(np.round(audio_resampled), np.iinfo(orig_dtype).min, np.iinfo(orig_dtype).max)
        audio_resampled = audio_resampled.astype(orig_dtype)

    return audio_resampled


class RealTimeState:
    """Holds audio/transcription state shared across threads and reruns."""

    def __init__(self) -> None:
        self.audio_queue: "queue.Queue[bytes]" = queue.Queue()
        self.transcript_lock = threading.Lock()
        self.transcript: str = ""
        self.running_event = threading.Event()
        self.audio_thread: threading.Thread | None = None
        self.transcriber_thread: threading.Thread | None = None
        self.wav_feeder_thread: threading.Thread | None = None
        self.last_update_ts: float | None = None
        self.segment_count: int = 0
        self.last_error: str | None = None
        self.source: str | None = None  # "mic" or "wav"
        self.mic_device_index: int | None = None
        self.mic_device_name: str | None = None


def list_input_devices():
    """Return a list of (device_index, display_name) for input-capable devices."""
    devices = sd.query_devices()
    out = []
    for idx, d in enumerate(devices):
        try:
            max_in = int(d.get("max_input_channels", 0))
        except Exception:
            max_in = 0
        if max_in and max_in > 0:
            name = str(d.get("name", f"Device {idx}"))
            out.append((idx, f"{idx}: {name} (in_ch={max_in})"))
    return out


def get_rt_state_and_model() -> tuple[RealTimeState, WhisperModel]:
    """
    Initialize and return the shared real-time state and Whisper model
    stored in Streamlit's session_state.
    """
    if "rt_state" not in st.session_state:
        st.session_state.rt_state = RealTimeState()
    if "whisper_model" not in st.session_state:
        # Load the local Whisper model once and reuse it
        # Use "cuda" for device if you have a GPU available.
        st.session_state.whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")

    return st.session_state.rt_state, st.session_state.whisper_model


def audio_stream_worker(rt_state: RealTimeState, device_index: int | None) -> None:
    """Background worker that maintains a live input audio stream."""

    def _callback(indata, frames, time_info, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status)
        rt_state.audio_queue.put(bytes(indata))

    try:
        if device_index is None:
            # sounddevice will use sd.default.device[0] (default input)
            rt_state.mic_device_index = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
        else:
            rt_state.mic_device_index = device_index

        if rt_state.mic_device_index is not None:
            try:
                rt_state.mic_device_name = sd.query_devices(rt_state.mic_device_index).get("name")
            except Exception:
                rt_state.mic_device_name = None

        with sd.InputStream(
            callback=_callback,
            dtype=DTYPE,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            device=device_index,
        ):
            # Keep the stream alive while running flag is set
            while rt_state.running_event.is_set():
                sd.sleep(100)
    except Exception as e:
        rt_state.running_event.clear()
        print(f"Audio stream error: {e}")


def transcriber_worker(rt_state: RealTimeState, model: WhisperModel) -> None:
    """Background worker that consumes audio chunks and performs transcription."""

    full_audio_buffer = b""

    print("Transcriber started. Listening for audio chunks...")

    # Keep going while running, and drain any already-queued audio after stop.
    while rt_state.running_event.is_set() or not rt_state.audio_queue.empty():
        try:
            data = rt_state.audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        full_audio_buffer += data

        # Process when buffer reaches desired size
        if len(full_audio_buffer) >= BLOCK_SIZE * 2:  # int16 => 2 bytes per sample
            try:
                # Convert bytes to float32 numpy array in range [-1.0, 1.0]
                audio_np = np.frombuffer(full_audio_buffer, dtype=DTYPE).astype(np.float32) / 32768.0

                # Transcribe the chunk using the local Whisper model
                segments, info = model.transcribe(audio_np, beam_size=5)

                # Append the recognized text to the shared transcript
                chunk_text = ""
                for segment in segments:
                    line = f"{segment.text.strip()} "
                    print(f"[ {segment.start:.2f}s -> {segment.end:.2f}s ] {segment.text}")
                    chunk_text += line
                    rt_state.segment_count += 1

                if chunk_text:
                    with rt_state.transcript_lock:
                        rt_state.transcript += chunk_text
                    rt_state.last_update_ts = time.time()

            except Exception as e:
                rt_state.last_error = f"{type(e).__name__}: {e}"
                print(f"Transcription error: {rt_state.last_error}")

            # Clear the processed audio buffer (could keep tail for context if desired)
            full_audio_buffer = b""

    print("Transcriber stopped.")


def start_realtime_transcription(rt_state: RealTimeState, model: WhisperModel, device_index: int | None) -> None:
    """Start microphone capture and transcription if not already running."""
    if rt_state.running_event.is_set():
        return

    # Reset transcript and queue for a fresh session
    with rt_state.transcript_lock:
        rt_state.transcript = ""
    rt_state.segment_count = 0
    rt_state.last_update_ts = None
    rt_state.last_error = None
    rt_state.source = "mic"
    while not rt_state.audio_queue.empty():
        try:
            rt_state.audio_queue.get_nowait()
        except queue.Empty:
            break

    rt_state.running_event.set()

    # Start audio stream worker
    rt_state.audio_thread = threading.Thread(
        target=audio_stream_worker,
        args=(rt_state, device_index),
        daemon=True,
        name="audio_stream_worker",
    )
    rt_state.audio_thread.start()

    # Start transcriber worker
    rt_state.transcriber_thread = threading.Thread(
        target=transcriber_worker,
        args=(rt_state, model),
        daemon=True,
        name="transcriber_worker",
    )
    rt_state.transcriber_thread.start()


def wav_feeder_worker(rt_state: RealTimeState, wav_path: Path) -> None:
    """Feed a WAV file into the audio queue in (roughly) real-time chunks."""
    import scipy.io.wavfile as wavfile

    try:
        sr, audio = wavfile.read(str(wav_path))
        if sr != SAMPLE_RATE:
            audio = resample_to_sample_rate(audio, sr, SAMPLE_RATE)
            sr = SAMPLE_RATE

        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1).astype(audio.dtype)

        # Ensure int16 PCM
        if audio.dtype != np.int16:
            if np.issubdtype(audio.dtype, np.floating):
                audio = np.clip(audio, -1.0, 1.0)
                audio = (audio * 32767.0).astype(np.int16)
            else:
                audio = audio.astype(np.int16)

        chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
        for start in range(0, len(audio), chunk_samples):
            if not rt_state.running_event.is_set():
                break
            chunk = audio[start : start + chunk_samples]
            rt_state.audio_queue.put(chunk.tobytes())
            time.sleep(CHUNK_DURATION)
    except Exception as e:
        rt_state.last_error = f"{type(e).__name__}: {e}"
        print(f"WAV feeder error: {rt_state.last_error}")
    finally:
        # Let the transcriber drain the queue, then stop.
        rt_state.running_event.clear()


def start_simulated_live_from_wav(rt_state: RealTimeState, model: WhisperModel, wav_path: Path) -> None:
    """Start transcription by feeding a WAV file in chunks (no microphone required)."""
    if rt_state.running_event.is_set():
        return

    # Reset transcript and queue for a fresh session
    with rt_state.transcript_lock:
        rt_state.transcript = ""
    rt_state.segment_count = 0
    rt_state.last_update_ts = None
    rt_state.last_error = None
    rt_state.source = "wav"
    while not rt_state.audio_queue.empty():
        try:
            rt_state.audio_queue.get_nowait()
        except queue.Empty:
            break

    rt_state.running_event.set()

    # Start transcriber worker
    rt_state.transcriber_thread = threading.Thread(
        target=transcriber_worker,
        args=(rt_state, model),
        daemon=True,
        name="transcriber_worker",
    )
    rt_state.transcriber_thread.start()

    # Start WAV feeder worker
    rt_state.wav_feeder_thread = threading.Thread(
        target=wav_feeder_worker,
        args=(rt_state, wav_path),
        daemon=True,
        name="wav_feeder_worker",
    )
    rt_state.wav_feeder_thread.start()


def stop_realtime_transcription(rt_state: RealTimeState) -> None:
    """Signal background workers to stop and wait for them to finish."""
    if not rt_state.running_event.is_set():
        return

    rt_state.running_event.clear()
    # Drop queued audio so stop is fast
    while not rt_state.audio_queue.empty():
        try:
            rt_state.audio_queue.get_nowait()
        except queue.Empty:
            break

    # Give workers a moment to exit gracefully
    if rt_state.audio_thread is not None and rt_state.audio_thread.is_alive():
        rt_state.audio_thread.join(timeout=2.0)
    if rt_state.wav_feeder_thread is not None and rt_state.wav_feeder_thread.is_alive():
        rt_state.wav_feeder_thread.join(timeout=2.0)
    if rt_state.transcriber_thread is not None and rt_state.transcriber_thread.is_alive():
        rt_state.transcriber_thread.join(timeout=5.0)


def main():
    rt_state, model = get_rt_state_and_model()

    st.title("Real-time Whisper Transcription (Local Model)")
    st.write(
        "This app captures live audio from your microphone in chunks and transcribes it "
        "in real-time using a local Whisper model."
    )

    @st.fragment(run_every=0.5)
    def _controls_panel():
        st.subheader("Microphone")

        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
        default_name = None
        if default_in is not None and default_in >= 0:
            try:
                default_name = sd.query_devices(default_in).get("name")
            except Exception:
                default_name = None

        devices = list_input_devices()
        device_labels = ["(default input device)"] + [label for _, label in devices]
        label_to_index = {"(default input device)": None}
        for idx, label in devices:
            label_to_index[label] = idx

        selected_label = st.selectbox(
            "Input device",
            options=device_labels,
            index=0,
            key="mic_device_select",
            help="Choose which input device to use for live capture. Default uses sounddevice's default input.",
        )
        selected_index = label_to_index.get(selected_label)

        st.caption(
            f"default_input={default_in} ({default_name or 'unknown'})  "
            f"selected={selected_index if selected_index is not None else 'default'}"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Listening", disabled=rt_state.running_event.is_set()):
                start_realtime_transcription(rt_state, model, selected_index)
                st.success("Started real-time transcription. Speak into your microphone.")

        with col2:
            # Keep Stop always enabled during fragments; clicking when already stopped is a no-op.
            if st.button("Stop Listening", disabled=not rt_state.running_event.is_set()):
                stop_realtime_transcription(rt_state)
                st.info("Stopped transcription.")

    _controls_panel()

    @st.fragment(run_every=0.5)
    def _wav_panel():
        st.markdown("---")
        st.subheader("WAV Test / Simulation")

        ui_dir = Path(__file__).resolve().parent
        bundled_wav = ui_dir / "testaudio_16000_test01_20s.wav"
        wav_source = st.selectbox(
            "Audio source",
            options=[
                "Upload WAV…",
                f"Bundled: {bundled_wav.name}",
            ],
            index=1 if bundled_wav.exists() else 0,
            key="wav_source_select",
        )

        uploaded = None
        selected_path: Path | None = None
        if wav_source.startswith("Upload"):
            uploaded = st.file_uploader("Upload WAV", type=["wav"], accept_multiple_files=False, key="test_wav")
        else:
            selected_path = bundled_wav if bundled_wav.exists() else None

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Transcribe WAV (one-shot)", key="btn_transcribe_wav"):
                import scipy.io.wavfile as wavfile
                import io

                if uploaded is None and selected_path is None:
                    st.error("Please upload a WAV file or choose the bundled sample.")
                else:
                    if selected_path is not None:
                        sr, audio = wavfile.read(str(selected_path))
                    else:
                        data = uploaded.read()
                        sr, audio = wavfile.read(io.BytesIO(data))

                    if sr != SAMPLE_RATE:
                        audio = resample_to_sample_rate(audio, sr, SAMPLE_RATE)

                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1).astype(audio.dtype)

                    audio_np = audio.astype(np.float32) / 32768.0
                    segments, info = model.transcribe(audio_np, beam_size=5)
                    text = "".join(seg.text.strip() + " " for seg in segments)
                    with rt_state.transcript_lock:
                        rt_state.transcript = text
                    rt_state.segment_count = len(list(segments))
                    rt_state.last_update_ts = time.time()
                    rt_state.last_error = None
                    rt_state.source = "wav"
                    st.success("WAV transcription complete. See Live Transcript below.")

        with c2:
            if st.button("Simulate live from WAV", disabled=rt_state.running_event.is_set(), key="btn_sim_live"):
                if selected_path is None:
                    st.error("Simulation requires selecting the bundled WAV (no upload simulation yet).")
                else:
                    start_simulated_live_from_wav(rt_state, model, selected_path)
                    st.success("Started WAV-fed live simulation (no microphone required).")

    _wav_panel()

    st.subheader("Live Transcript")

    @st.fragment(run_every=0.5)
    def _live_panel():
        # Display the current transcript (updated by background threads).
        with rt_state.transcript_lock:
            current_text = rt_state.transcript or "(no transcript yet)"

        st.text(current_text)

        st.caption(
            f"running={rt_state.running_event.is_set()}  "
            f"source={rt_state.source}  "
            f"queue={rt_state.audio_queue.qsize()}  "
            f"segments={rt_state.segment_count}  "
            f"last_update={'-' if rt_state.last_update_ts is None else f'{time.time() - rt_state.last_update_ts:.1f}s ago'}  "
            f"err={rt_state.last_error or '-'}"
        )

    _live_panel()


if __name__ == "__main__":
    main()
