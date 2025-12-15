"""
Streamlit real-time speech-to-text (STT) demo using faster-whisper.

This app supports multiple audio sources:
- Server microphone ("Microphone"): captured on the server via `sounddevice` (PortAudio).
  This only works when the Streamlit server has access to an input audio device.
- Browser microphone ("Browser Mic (WebSocket)"): captured in the browser via Web Audio and
  streamed to the server over a plain WebSocket as int16 mono PCM frames.
- WAV test/simulation: transcribe a WAV file in one-shot or simulate streaming from a WAV.

Why WebSocket browser mic?
WebRTC can be tricky in deployed environments (ICE/STUN/TURN). A simple WebSocket stream is
often easier to deploy as long as you can expose/proxy the WebSocket endpoint.

WebSocket protocol (browser -> server)
- Connect to: ws://<host>:<BROWSER_MIC_WS_PORT>/ws?token=<token>
- After connect, browser sends: {"type":"hello","token":"...","sample_rate": <AudioContext.sampleRate>}
- Then the browser sends binary frames: little-endian int16 mono PCM at `sample_rate`.

Deployment notes
- By default the WS server listens on 0.0.0.0:8765. Many platforms only expose one port.
  In that case you must proxy `/ws` to this WS server, or expose the WS port directly.
- You can override the browser WS URL with `BROWSER_MIC_WS_URL`. It supports `{token}`
  substitution, e.g.:
    export BROWSER_MIC_WS_URL="wss://your.domain/ws?token={token}"

Environment variables
- BROWSER_MIC_WS_HOST: host to bind the WS server (default "0.0.0.0")
- BROWSER_MIC_WS_PORT: port to bind the WS server (default "8765")
- BROWSER_MIC_WS_URL: override WS URL used by the browser (optional; useful behind proxies)
"""

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
import asyncio
import secrets
from urllib.parse import urlparse, parse_qs

import streamlit.components.v1 as components

import websockets

try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None  # allows running in browser-only/container deployments without PortAudio

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

# Duration (in seconds) of audio to accumulate before transcribing.
# Smaller values => lower latency, higher overhead.
CHUNK_DURATION = 5
BLOCK_SIZE = SAMPLE_RATE * CHUNK_DURATION

# Browser mic via WebSocket should feel responsive; keep its chunk smaller without affecting other modes.
BROWSER_WS_CHUNK_DURATION = 1.5
# Browser WebSocket post-filtering to reduce hallucinations on noise.
# These thresholds only apply when rt_state.source == "browser_ws".
# - BROWSER_WS_MIN_SEGMENT_DURATION_S:
#   Drop very short segments. Short "blips" are often noise or partial phonemes that Whisper can
#   turn into random words (hallucinations).
# - BROWSER_WS_MAX_NO_SPEECH_PROB:
#   Whisper outputs `no_speech_prob` per segment. If it's high, the model thinks it was probably
#   silence/noise; dropping those reduces hallucinations.
# - BROWSER_WS_MIN_AVG_LOGPROB:
#   Whisper outputs `avg_logprob` (average token log-probability). More negative means less confident.
#   Dropping low-confidence segments reduces garbage in the transcript.
BROWSER_WS_MIN_SEGMENT_DURATION_S = 0.10
BROWSER_WS_MAX_NO_SPEECH_PROB = 0.80
BROWSER_WS_MIN_AVG_LOGPROB = -0.80
# Extra browser_ws-only repetition suppression (common hallucination pattern: "Okay." repeated)
BROWSER_WS_MAX_REPEAT_SAME_TEXT = 2  # allow up to N consecutive identical segments

_WS_TOKEN_LOCK = threading.Lock()
_WS_TOKEN_TO_CTX: dict[str, tuple["RealTimeState", WhisperModel]] = {}
_WS_SERVER_STARTED = False
_WS_SERVER_LOCK = threading.Lock()


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
        # Browser mic via WebSocket stats (optional)
        self.browser_ws_connected: bool = False
        self.browser_ws_sample_rate: int | None = None
        self.browser_ws_frames: int = 0
        self.browser_ws_bytes: int = 0


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
    last_text_norm: str | None = None
    last_text_repeat: int = 0

    print("Transcriber started. Listening for audio chunks...")

    # Mic capture may not be 16k; we resample to SAMPLE_RATE for Whisper.
    capture_sr = int(rt_state.capture_sample_rate or SAMPLE_RATE)
    bytes_per_sample = np.dtype(DTYPE).itemsize
    chunk_duration = BROWSER_WS_CHUNK_DURATION if rt_state.source == "browser_ws" else CHUNK_DURATION
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

                # Transcribe the chunk using the local Whisper model.
                transcribe_kwargs = {"beam_size": 5}
                # Browser WS mic: be conservative to avoid hallucinations, without impacting other modes.
                if rt_state.source == "browser_ws":
                    transcribe_kwargs.update(
                        {
                            "language": "en",
                            "task": "transcribe",
                            "beam_size": 1,
                            "best_of": 1,
                            "temperature": 0.0,
                            "condition_on_previous_text": False,
                            "repetition_penalty": 1.2,
                            "no_repeat_ngram_size": 3,
                            "vad_filter": True,
                            "vad_parameters": {
                                "min_silence_duration_ms": 400,
                                "speech_pad_ms": 200,
                            },
                            "no_speech_threshold": 0.8,
                            "log_prob_threshold": -0.8,
                            "compression_ratio_threshold": 2.2,
                            "hallucination_silence_threshold": 0.5,
                        }
                    )

                segments, info = model.transcribe(audio_np, **transcribe_kwargs)

                # Append the recognized text to the shared transcript
                chunk_text = ""
                for segment in segments:
                    # Browser WS mic: drop low-confidence / likely-no-speech segments.
                    if rt_state.source == "browser_ws":
                        seg_dur = float(segment.end - segment.start)
                        if seg_dur < BROWSER_WS_MIN_SEGMENT_DURATION_S:
                            continue
                        if float(segment.no_speech_prob) > BROWSER_WS_MAX_NO_SPEECH_PROB:
                            continue
                        if float(segment.avg_logprob) < BROWSER_WS_MIN_AVG_LOGPROB:
                            continue

                    txt = segment.text.strip()
                    if rt_state.source == "browser_ws":
                        # Drop excessive repeats of identical text (e.g. "Okay." over and over).
                        norm = " ".join(txt.lower().split())
                        if norm and norm == last_text_norm:
                            last_text_repeat += 1
                            if last_text_repeat > BROWSER_WS_MAX_REPEAT_SAME_TEXT:
                                continue
                        else:
                            last_text_norm = norm
                            last_text_repeat = 0

                    line = f"{txt} "
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


def start_browser_ws_transcription(rt_state: RealTimeState, model: WhisperModel, capture_sr: int) -> None:
    """Start transcription for browser mic (audio is fed via a WebSocket)."""
    if rt_state.running_event.is_set():
        return

    with rt_state.transcript_lock:
        rt_state.transcript = ""
    rt_state.segment_count = 0
    rt_state.last_update_ts = None
    rt_state.last_error = None
    rt_state.source = "browser_ws"
    rt_state.capture_sample_rate = int(capture_sr)
    rt_state.browser_ws_frames = 0
    rt_state.browser_ws_bytes = 0

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


async def _ws_handler(websocket, path=None):
    """Handle one browser WS client connection.

    Expected client behavior:
    - token may be passed via query param `?token=...`
    - client should send a hello JSON message with sample_rate (AudioContext sample rate)
    - client then sends binary audio frames as int16 mono PCM
    """
    rt_state: RealTimeState | None = None
    model: WhisperModel | None = None
    token: str | None = None
    capture_sr: int = 48000

    try:
        # token can be passed as query param: ws://host:port/ws?token=...
        try:
            # websockets<11 passes `path` arg; websockets>=11 attaches it to the connection
            req_path = path or getattr(websocket, "path", None) or getattr(getattr(websocket, "request", None), "path", "")
            parsed = urlparse(req_path)
            qs = parse_qs(parsed.query or "")
            token = (qs.get("token", [None])[0]) if qs else None
        except Exception:
            token = None

        if token:
            with _WS_TOKEN_LOCK:
                ctx = _WS_TOKEN_TO_CTX.get(token)
                if ctx:
                    rt_state, model = ctx

        if rt_state is None:
            # Wait for a hello message with token
            msg = await websocket.recv()
            if isinstance(msg, (bytes, bytearray)):
                await websocket.close(code=1008, reason="Expected hello JSON first")
                return
            try:
                hello = json.loads(msg)
            except Exception:
                await websocket.close(code=1008, reason="Invalid hello JSON")
                return
            token = str(hello.get("token", "")).strip() or None
            capture_sr = int(hello.get("sample_rate", capture_sr) or capture_sr)
            if token:
                with _WS_TOKEN_LOCK:
                    ctx = _WS_TOKEN_TO_CTX.get(token)
                    if ctx:
                        rt_state, model = ctx

        if rt_state is None:
            await websocket.close(code=1008, reason="Unknown token")
            return

        rt_state.browser_ws_connected = True
        rt_state.browser_ws_sample_rate = capture_sr

        # Start transcriber when the first client connects (needs model registered for this token)
        if model is not None:
            if not rt_state.running_event.is_set() or rt_state.source != "browser_ws":
                start_browser_ws_transcription(rt_state, model, capture_sr=capture_sr)

        async for msg in websocket:
            if isinstance(msg, str):
                # Optional control messages
                try:
                    ctrl = json.loads(msg)
                except Exception:
                    continue
                if ctrl.get("type") == "hello":
                    capture_sr = int(ctrl.get("sample_rate", capture_sr) or capture_sr)
                    rt_state.browser_ws_sample_rate = capture_sr
                continue

            # Binary audio chunk
            if rt_state.running_event.is_set() and rt_state.source == "browser_ws":
                rt_state.audio_queue.put(bytes(msg))
                rt_state.browser_ws_frames += 1
                rt_state.browser_ws_bytes += len(msg)

    finally:
        if rt_state is not None:
            rt_state.browser_ws_connected = False
            # stop only if this session is in browser_ws mode
            if rt_state.running_event.is_set() and rt_state.source == "browser_ws":
                stop_realtime_transcription(rt_state)


def ensure_browser_ws_server() -> int:
    """Start the browser mic WebSocket server once per process; returns port."""
    global _WS_SERVER_STARTED
    with _WS_SERVER_LOCK:
        if _WS_SERVER_STARTED:
            return int(os.getenv("BROWSER_MIC_WS_PORT", "8765"))

        port = int(os.getenv("BROWSER_MIC_WS_PORT", "8765"))
        host = os.getenv("BROWSER_MIC_WS_HOST", "0.0.0.0")

        def _run():
            async def _main():
                async with websockets.serve(_ws_handler, host, port, max_size=2**22):
                    await asyncio.Future()  # run forever

            asyncio.run(_main())

        t = threading.Thread(target=_run, daemon=True, name="browser_mic_ws_server")
        t.start()
        _WS_SERVER_STARTED = True
        return port

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

    def _browser_ws_panel():
        st.subheader("Browser Mic (WebSocket)")

        port = ensure_browser_ws_server()
        if "browser_ws_token" not in st.session_state:
            st.session_state.browser_ws_token = secrets.token_urlsafe(16)
        token = st.session_state.browser_ws_token

        # Register this Streamlit session with the WS server
        with _WS_TOKEN_LOCK:
            _WS_TOKEN_TO_CTX[token] = (rt_state, model)

        st.caption(
            f"WebSocket server: ws://<this-host>:{port}/ws?token=...  "
            f"(deployment must expose/proxy this port)"
        )

        ws_url_env = os.getenv("BROWSER_MIC_WS_URL", "").strip()
        ws_url = ws_url_env if ws_url_env else ""  # JS will build from location + port if empty

        html = f"""
<!doctype html>
<html>
  <body>
    <div style="font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;">
      <button id="startBtn">Start Browser Mic</button>
      <button id="stopBtn" disabled>Stop</button>
      <span id="status" style="margin-left: 8px;">idle</span>
    </div>
    <script>
      const token = {json.dumps(token)};
      const wsPort = {port};
      const wsUrlOverride = {json.dumps(ws_url)};

      function buildWsUrl() {{
        if (wsUrlOverride) {{
          // allow {token} substitution
          return wsUrlOverride.replaceAll("{{token}}", encodeURIComponent(token));
        }}
        // We're running inside an iframe (often about:srcdoc), so window.location may not have a hostname.
        // Prefer parent window location, then referrer.
        let proto = "";
        let host = "";
        try {{
          proto = window.parent.location.protocol || "";
          host = window.parent.location.hostname || "";
        }} catch (e) {{}}
        if (!host) {{
          try {{
            proto = window.location.protocol || proto;
            host = window.location.hostname || host;
          }} catch (e) {{}}
        }}
        if (!host) {{
          try {{
            const u = new URL(document.referrer);
            proto = u.protocol || proto;
            host = u.hostname || host;
          }} catch (e) {{}}
        }}
        const scheme = (proto === "https:") ? "wss" : "ws";
        return `${{scheme}}://${{host}}:${{wsPort}}/ws?token=${{encodeURIComponent(token)}}`;
      }}

      const statusEl = document.getElementById("status");
      const startBtn = document.getElementById("startBtn");
      const stopBtn = document.getElementById("stopBtn");

      let ws = null;
      let audioCtx = null;
      let processor = null;
      let source = null;
      let stream = null;
      let zeroGain = null;

      function setStatus(s) {{
        statusEl.textContent = s;
      }}

      function floatTo16BitPCM(float32Array) {{
        const out = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {{
          let s = Math.max(-1, Math.min(1, float32Array[i]));
          out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }}
        return out;
      }}

      async function start() {{
        startBtn.disabled = true;
        stopBtn.disabled = false;
        setStatus("requesting mic...");

        const url = buildWsUrl();
        ws = new WebSocket(url);
        ws.binaryType = "arraybuffer";

        ws.onopen = async () => {{
          setStatus("ws connected, starting audio...");
          stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
          source = audioCtx.createMediaStreamSource(stream);

          // ScriptProcessor is deprecated but simplest for a first pass.
          const bufSize = 4096;
          processor = audioCtx.createScriptProcessor(bufSize, source.channelCount, 1);
          zeroGain = audioCtx.createGain();
          zeroGain.gain.value = 0;

          processor.onaudioprocess = (e) => {{
            if (!ws || ws.readyState !== 1) return;
            const inBuf = e.inputBuffer;
            const chs = inBuf.numberOfChannels;
            const frames = inBuf.length;
            const mono = new Float32Array(frames);
            for (let ch = 0; ch < chs; ch++) {{
              const data = inBuf.getChannelData(ch);
              for (let i = 0; i < frames; i++) mono[i] += data[i] / chs;
            }}
            const pcm16 = floatTo16BitPCM(mono);
            ws.send(pcm16.buffer);
          }};

          source.connect(processor);
          processor.connect(zeroGain);
          zeroGain.connect(audioCtx.destination);

          // Send hello so server can know the sample rate
          ws.send(JSON.stringify({{type: "hello", token, sample_rate: audioCtx.sampleRate}}));
          setStatus("streaming...");
        }};

        ws.onclose = () => {{
          setStatus("ws closed");
          startBtn.disabled = false;
          stopBtn.disabled = true;
        }};

        ws.onerror = (e) => {{
          setStatus("ws error");
        }};
      }}

      async function stop() {{
        stopBtn.disabled = true;
        startBtn.disabled = false;
        setStatus("stopping...");

        try {{
          if (processor) processor.disconnect();
          if (source) source.disconnect();
          if (zeroGain) zeroGain.disconnect();
          if (stream) stream.getTracks().forEach(t => t.stop());
          if (audioCtx) await audioCtx.close();
        }} catch (e) {{}}

        processor = null; source = null; zeroGain = null; stream = null; audioCtx = null;

        try {{
          if (ws) ws.close();
        }} catch (e) {{}}
        ws = null;
        setStatus("idle");
      }}

      startBtn.addEventListener("click", start);
      stopBtn.addEventListener("click", stop);
    </script>
  </body>
</html>
"""
        components.html(html, height=80)

    _browser_ws_panel()

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
            f"ws_connected={rt_state.browser_ws_connected if rt_state.source == 'browser_ws' else '-'}  "
            f"ws_sr={rt_state.browser_ws_sample_rate if rt_state.source == 'browser_ws' else '-'}  "
            f"ws_frames={rt_state.browser_ws_frames if rt_state.source == 'browser_ws' else '-'}  "
            f"ws_kb={rt_state.browser_ws_bytes // 1024 if rt_state.source == 'browser_ws' else '-'}  "
            f"err={rt_state.last_error or '-'}"
        )

    _live_panel()


if __name__ == "__main__":
    main()
