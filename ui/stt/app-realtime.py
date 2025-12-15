import streamlit as st
import torch
import numpy as np
from faster_whisper import WhisperModel
from scipy.signal import resample_poly
import queue
import threading
import time
from pathlib import Path
import io
import os
import json

try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None  # allows running in browser-only/container deployments without PortAudio

try:
    from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer, RTCConfiguration  # type: ignore
except Exception:  # pragma: no cover
    AudioProcessorBase = None  # type: ignore
    WebRtcMode = None  # type: ignore
    webrtc_streamer = None  # type: ignore
    RTCConfiguration = None  # type: ignore

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

# Duration (in seconds) of audio to accumulate before transcribing.
# Smaller values => lower latency, higher overhead.
CHUNK_DURATION = 5
BLOCK_SIZE = SAMPLE_RATE * CHUNK_DURATION

# WebRTC browser mic should feel responsive; keep its chunk smaller without affecting other modes.
BROWSER_MIC_CHUNK_DURATION = 0.75

# Browser-mic post-filtering to reduce hallucinations on noise.
# These thresholds only apply when rt_state.source == "browser_mic".
BROWSER_MIC_MIN_SEGMENT_DURATION_S = 0.10
BROWSER_MIC_MAX_NO_SPEECH_PROB = 0.95
BROWSER_MIC_MIN_AVG_LOGPROB = -1.20


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
        self.capture_sample_rate: int | None = None  # actual input stream sample rate (mic)
        # Minimal WebRTC debug (does not affect other modes)
        self.browser_mic_enqueued_bytes: int = 0
        self.browser_mic_enqueued_frames: int = 0
        self.browser_mic_last_pcm_shape: str | None = None
        self.browser_mic_last_pcm_dtype: str | None = None
        self.browser_mic_last_in_sr: int | None = None
        self.browser_mic_last_channels_meta: int | None = None
        self.browser_mic_last_mono_samples: int | None = None
        self.browser_mic_last_max_abs: float | None = None


def list_input_devices():
    """Return a list of (device_index, display_name) for input-capable devices."""
    if sd is None:
        return []
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
        st.session_state.whisper_model = WhisperModel("base.en", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")

    return st.session_state.rt_state, st.session_state.whisper_model


def audio_stream_worker(rt_state: RealTimeState, device_index: int | None) -> None:
    """Background worker that maintains a live input audio stream."""
    if sd is None:
        rt_state.last_error = "sounddevice is not available in this environment (no server-side mic)."
        rt_state.running_event.clear()
        return

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

        # Some devices/drivers reject 16kHz capture. Try 16k first, then fall back to the
        # device's default sample rate and resample to 16k before Whisper.
        def _open_stream(sr: int):
            blocksize = int(sr * CHUNK_DURATION)
            return sd.InputStream(
                callback=_callback,
                dtype=DTYPE,
                channels=CHANNELS,
                samplerate=sr,
                blocksize=blocksize,
                device=device_index,
            )

        try:
            rt_state.capture_sample_rate = SAMPLE_RATE
            stream = _open_stream(SAMPLE_RATE)
        except Exception as e_16k:
            # Determine device default sample rate
            default_sr = None
            try:
                dev_idx = rt_state.mic_device_index
                if dev_idx is not None:
                    default_sr = sd.query_devices(dev_idx).get("default_samplerate")
            except Exception:
                default_sr = None

            fallback_sr = int(default_sr) if default_sr else 48000
            rt_state.capture_sample_rate = fallback_sr
            rt_state.last_error = (
                f"16kHz capture unsupported ({type(e_16k).__name__}: {e_16k}); "
                f"falling back to {fallback_sr}Hz and resampling to 16k."
            )
            stream = _open_stream(fallback_sr)

        with stream:
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

    # Mic capture may not be 16k; we resample to SAMPLE_RATE for Whisper.
    capture_sr = int(rt_state.capture_sample_rate or SAMPLE_RATE)
    bytes_per_sample = np.dtype(DTYPE).itemsize
    chunk_duration = BROWSER_MIC_CHUNK_DURATION if rt_state.source == "browser_mic" else CHUNK_DURATION
    target_bytes = int(capture_sr * chunk_duration) * bytes_per_sample * CHANNELS

    # Keep going while running, and drain any already-queued audio after stop.
    while rt_state.running_event.is_set() or not rt_state.audio_queue.empty():
        try:
            data = rt_state.audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        full_audio_buffer += data

        # Process when buffer reaches desired size
        if len(full_audio_buffer) >= target_bytes:
            try:
                # Convert bytes to float32 numpy array in range [-1.0, 1.0]
                audio_np = np.frombuffer(full_audio_buffer, dtype=DTYPE).astype(np.float32) / 32768.0
                if capture_sr != SAMPLE_RATE:
                    audio_np = resample_to_sample_rate(audio_np, capture_sr, SAMPLE_RATE).astype(np.float32)

                # Transcribe the chunk using the local Whisper model
                transcribe_kwargs = {"beam_size": 5}
                # Keep browser-mic conservative to reduce hallucinations, without impacting other modes.
                if rt_state.source == "browser_mic":
                    transcribe_kwargs.update(
                        {
                            "language": "en",
                            "task": "transcribe",
                            "beam_size": 1,
                            "best_of": 1,
                            "temperature": 0.0,
                            "condition_on_previous_text": False,
                            "vad_filter": True,
                            "vad_parameters": {
                                "min_silence_duration_ms": 200,
                                "speech_pad_ms": 100,
                            },
                            # Slightly more permissive so quiet/short speech isn't dropped.
                            "no_speech_threshold": 0.6,
                            "log_prob_threshold": -1.0,
                            "compression_ratio_threshold": 2.2,
                            "hallucination_silence_threshold": 0.5,
                        }
                    )

                segments, info = model.transcribe(audio_np, **transcribe_kwargs)

                # Append the recognized text to the shared transcript
                chunk_text = ""
                for segment in segments:
                    # Browser-mic-only: drop low-confidence / likely-no-speech segments.
                    if rt_state.source == "browser_mic":
                        seg_dur = float(segment.end - segment.start)
                        if seg_dur < BROWSER_MIC_MIN_SEGMENT_DURATION_S:
                            continue
                        if float(segment.no_speech_prob) > BROWSER_MIC_MAX_NO_SPEECH_PROB:
                            continue
                        if float(segment.avg_logprob) < BROWSER_MIC_MIN_AVG_LOGPROB:
                            continue

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
    if sd is None:
        rt_state.last_error = "sounddevice is not available in this environment (no server-side mic)."
        return
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


def start_browser_mic_transcription(rt_state: RealTimeState, model: WhisperModel) -> None:
    """Start transcription for browser mic (audio is fed via streamlit-webrtc)."""
    if rt_state.running_event.is_set():
        return

    with rt_state.transcript_lock:
        rt_state.transcript = ""
    rt_state.segment_count = 0
    rt_state.last_update_ts = None
    rt_state.last_error = None
    rt_state.source = "browser_mic"
    rt_state.capture_sample_rate = SAMPLE_RATE
    rt_state.browser_mic_enqueued_bytes = 0
    rt_state.browser_mic_enqueued_frames = 0
    rt_state.browser_mic_last_pcm_shape = None
    rt_state.browser_mic_last_pcm_dtype = None
    rt_state.browser_mic_last_in_sr = None
    rt_state.browser_mic_last_channels_meta = None
    rt_state.browser_mic_last_mono_samples = None
    rt_state.browser_mic_last_max_abs = None

    while not rt_state.audio_queue.empty():
        try:
            rt_state.audio_queue.get_nowait()
        except queue.Empty:
            break

    rt_state.running_event.set()

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


def _decode_wav_bytes_to_mono_float32_16k(wav_bytes: bytes) -> np.ndarray:
    """
    Decode WAV bytes -> mono float32 array in [-1, 1] at SAMPLE_RATE.
    Uses scipy.io.wavfile, which supports WAV containers.
    """
    import scipy.io.wavfile as wavfile

    sr, audio = wavfile.read(io.BytesIO(wav_bytes))

    if sr != SAMPLE_RATE:
        audio = resample_to_sample_rate(audio, sr, SAMPLE_RATE)

    # Convert to mono if needed
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Normalize to float32 [-1, 1]
    if np.issubdtype(audio.dtype, np.integer):
        # assume int16-like PCM
        denom = float(np.iinfo(audio.dtype).max)
        audio_np = audio.astype(np.float32) / denom
    else:
        audio_np = audio.astype(np.float32)
        audio_np = np.clip(audio_np, -1.0, 1.0)

    return audio_np


def main():
    rt_state, model = get_rt_state_and_model()

    st.title("Real-time Whisper Transcription (Local Model)")
    st.write(
        "This app captures live audio from your microphone in chunks and transcribes it "
        "in real-time using a local Whisper model."
    )

    def _browser_webrtc_panel():
        st.subheader("Browser Mic (WebRTC - simple)")

        if webrtc_streamer is None or AudioProcessorBase is None or WebRtcMode is None or RTCConfiguration is None:
            st.info("Install `streamlit-webrtc` to enable browser mic streaming.")
            return

        # Deployed environments often need explicit STUN/TURN.
        # - Default: Google public STUN (works in many cases)
        # - Optional: provide TURN via env var WEBRTC_ICE_SERVERS (JSON) or WEBRTC_TURN_URL/USER/PASS
        #
        # WEBRTC_ICE_SERVERS example:
        #   export WEBRTC_ICE_SERVERS='[{"urls":["stun:stun.l.google.com:19302"]},{"urls":["turn:turn.example.com:3478"],"username":"u","credential":"p"}]'
        ice_servers = None
        ice_servers_json = os.getenv("WEBRTC_ICE_SERVERS")
        if ice_servers_json:
            try:
                ice_servers = json.loads(ice_servers_json)
            except Exception as e:
                st.warning(f"Invalid WEBRTC_ICE_SERVERS JSON: {type(e).__name__}: {e}")
                ice_servers = None

        if ice_servers is None:
            turn_url = os.getenv("WEBRTC_TURN_URL")
            turn_user = os.getenv("WEBRTC_TURN_USERNAME")
            turn_pass = os.getenv("WEBRTC_TURN_PASSWORD")
            if turn_url and turn_user and turn_pass:
                ice_servers = [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": [turn_url], "username": turn_user, "credential": turn_pass},
                ]
            else:
                ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

        rtc_config = RTCConfiguration(iceServers=ice_servers)

        def _coerce_int(v, default: int) -> int:
            if v is None:
                return default
            if isinstance(v, (int, np.integer)):
                return int(v)
            if isinstance(v, (tuple, list)):
                return len(v) if len(v) > 0 else default
            try:
                return int(v)
            except Exception:
                return default

        class _AudioProcessor(AudioProcessorBase):
            async def recv_queued(self, frames):
                # Process all frames in async mode (avoids dropped-frame warnings and improves throughput).
                for f in frames:
                    self.recv(f)
                return frames

            def recv(self, frame):
                # Minimal decoding: to mono -> float32 -> resample to 16k -> int16 bytes -> enqueue
                try:
                    pcm = frame.to_ndarray()
                    rt_state.browser_mic_last_pcm_shape = str(getattr(pcm, "shape", None))
                    rt_state.browser_mic_last_pcm_dtype = str(getattr(pcm, "dtype", None))

                    channels_meta = 1
                    try:
                        channels_meta = _coerce_int(getattr(frame.layout, "channels", None), 1)
                    except Exception:
                        channels_meta = 1
                    rt_state.browser_mic_last_channels_meta = int(channels_meta)

                    # Normalize FIRST (before mixing). Otherwise np.mean(int16) becomes float64 in "PCM counts"
                    # and Whisper sees near-silence.
                    if np.issubdtype(pcm.dtype, np.integer):
                        denom = float(np.iinfo(pcm.dtype).max)
                        pcm_f = pcm.astype(np.float32) / denom
                    else:
                        pcm_f = pcm.astype(np.float32)

                    if pcm_f.ndim == 2:
                        # Common cases:
                        # - (channels, samples)
                        # - (samples, channels)
                        if pcm_f.shape[0] <= 8 and pcm_f.shape[1] > pcm_f.shape[0]:
                            # channels-first
                            if pcm_f.shape[0] == 1 and channels_meta > 1 and (pcm_f.shape[1] % channels_meta) == 0:
                                interleaved = pcm_f.reshape(-1)
                                pcm2 = interleaved.reshape(-1, channels_meta)
                                mono = np.mean(pcm2, axis=1)
                            else:
                                mono = np.mean(pcm_f, axis=0)
                        else:
                            mono = np.mean(pcm_f, axis=1)
                    else:
                        mono = pcm_f.reshape(-1)

                    audio_f = mono.astype(np.float32)

                    in_sr = _coerce_int(getattr(frame, "sample_rate", None), SAMPLE_RATE)
                    rt_state.browser_mic_last_in_sr = int(in_sr)
                    if in_sr != SAMPLE_RATE:
                        audio_f = resample_to_sample_rate(audio_f, in_sr, SAMPLE_RATE).astype(np.float32)

                    # Clamp to sane range (resample can overshoot slightly)
                    audio_f = np.clip(audio_f, -1.0, 1.0)
                    rt_state.browser_mic_last_mono_samples = int(audio_f.size)
                    rt_state.browser_mic_last_max_abs = float(np.max(np.abs(audio_f))) if audio_f.size else 0.0

                    audio_i16 = (np.clip(audio_f, -1.0, 1.0) * 32767.0).astype(np.int16)
                    b = audio_i16.tobytes()
                    if b and rt_state.running_event.is_set() and rt_state.source == "browser_mic":
                        rt_state.audio_queue.put(b)
                        rt_state.browser_mic_enqueued_frames += 1
                        rt_state.browser_mic_enqueued_bytes += len(b)
                except Exception as e:
                    rt_state.last_error = f"WebRTC audio error: {type(e).__name__}: {e}"
                return frame

        webrtc_ctx = webrtc_streamer(
            key="browser_mic_webrtc",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=rtc_config,
            audio_processor_factory=_AudioProcessor,
            async_processing=True,
            media_stream_constraints={"audio": True, "video": False},
        )

        if webrtc_ctx.state.playing:
            if not rt_state.running_event.is_set() or rt_state.source != "browser_mic":
                start_browser_mic_transcription(rt_state, model)
                st.success("Browser mic streaming started.")
        else:
            if rt_state.running_event.is_set() and rt_state.source == "browser_mic":
                stop_realtime_transcription(rt_state)
                st.info("Browser mic streaming stopped.")

    _browser_webrtc_panel()

    @st.fragment(run_every=0.5)
    def _controls_panel():
        st.subheader("Microphone")

        if sd is None:
            st.warning(
                "Server-side microphone capture is unavailable here (sounddevice/PortAudio missing). "
                "Use **Browser Recorder** below instead."
            )
            return

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

    def _browser_audio_panel():
        st.markdown("---")
        st.subheader("Browser Recorder (st.audio_input)")

        audio_value = st.audio_input("Record a voice message", key="browser_audio_input")

        if audio_value:
            st.success("Audio recorded successfully!")
            st.audio(audio_value)

            if st.button("Transcribe recording (one-shot)", key="btn_transcribe_browser_audio"):
                try:
                    data = audio_value.getvalue() if hasattr(audio_value, "getvalue") else audio_value.read()
                    audio_np = _decode_wav_bytes_to_mono_float32_16k(data)
                    segs, info = model.transcribe(audio_np, beam_size=5)
                    seg_list = list(segs)
                    text = "".join(seg.text.strip() + " " for seg in seg_list)
                    with rt_state.transcript_lock:
                        rt_state.transcript = text
                    rt_state.segment_count = len(seg_list)
                    rt_state.last_update_ts = time.time()
                    rt_state.last_error = None
                    rt_state.source = "browser_audio"
                    st.success("Browser recording transcription complete. See Live Transcript below.")
                except Exception as e:
                    rt_state.last_error = f"{type(e).__name__}: {e}"
                    st.error(f"Failed to transcribe browser recording: {rt_state.last_error}")

    _browser_audio_panel()

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
                if uploaded is None and selected_path is None:
                    st.error("Please upload a WAV file or choose the bundled sample.")
                else:
                    if selected_path is not None:
                        with open(selected_path, "rb") as f:
                            data = f.read()
                    else:
                        data = uploaded.read()

                    audio_np = _decode_wav_bytes_to_mono_float32_16k(data)
                    segs, info = model.transcribe(audio_np, beam_size=5)
                    seg_list = list(segs)
                    text = "".join(seg.text.strip() + " " for seg in seg_list)
                    with rt_state.transcript_lock:
                        rt_state.transcript = text
                    rt_state.segment_count = len(seg_list)
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
            f"webrtc_frames={rt_state.browser_mic_enqueued_frames if rt_state.source == 'browser_mic' else '-'}  "
            f"webrtc_kb={rt_state.browser_mic_enqueued_bytes // 1024 if rt_state.source == 'browser_mic' else '-'}  "
            f"pcm_shape={(rt_state.browser_mic_last_pcm_shape or '-') if rt_state.source == 'browser_mic' else '-'}  "
            f"dtype={(rt_state.browser_mic_last_pcm_dtype or '-') if rt_state.source == 'browser_mic' else '-'}  "
            f"in_sr={(rt_state.browser_mic_last_in_sr or '-') if rt_state.source == 'browser_mic' else '-'}  "
            f"ch_meta={(rt_state.browser_mic_last_channels_meta or '-') if rt_state.source == 'browser_mic' else '-'}  "
            f"samples={(rt_state.browser_mic_last_mono_samples or '-') if rt_state.source == 'browser_mic' else '-'}  "
            f"max={(f'{rt_state.browser_mic_last_max_abs:.3f}' if rt_state.browser_mic_last_max_abs is not None else '-') if rt_state.source == 'browser_mic' else '-'}  "
            f"err={rt_state.last_error or '-'}"
        )

    _live_panel()


if __name__ == "__main__":
    main()
