#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyaudio",
#     "faster-whisper",
# ]
# ///
"""
Continuous voice listener that talks to OpenClaw Gateway.

Usage:
    uv run voice_listener.py --list
    uv run voice_listener.py
    uv run voice_listener.py --session-id voice-local
"""

import argparse
import json
import os
import re
import select
import signal
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
import unicodedata
import uuid
import wave
from datetime import datetime

# Sound effect file paths (set to empty string to disable)
# Priority: env var > common system sounds > None
def find_default_sound():
    """Find a suitable system sound file."""
    common_sounds = [
        # Freedesktop sounds
        "/usr/share/sounds/freedesktop/stereo/message-new-instant.oga",
        "/usr/share/sounds/freedesktop/stereo/message.oga",
        "/usr/share/sounds/freedesktop/stereo/complete.oga",
        # Ubuntu/Debian sounds
        "/usr/share/sounds/gnome/default/alerts/glass.ogg",
        "/usr/share/sounds/gnome/default/alerts/drip.ogg",
        "/usr/share/sounds/gnome/default/alerts/click.ogg",
        "/usr/share/sounds/deepin/stereo/message.ogg",
        # KDE sounds
        "/usr/share/sounds/deepin/stereo/system-notification.ogg",
        # ALSA fallback
        "/usr/share/sounds/alsa/Noise.wav",
        # macOS sounds
        "/System/Library/Sounds/Glass.aiff",
        "/System/Library/Sounds/Tink.aiff",
        "/System/Library/Sounds/Pop.aiff",
    ]
    for path in common_sounds:
        if os.path.exists(path):
            return path
    return None

_DEFAULT_SOUND = find_default_sound()
SOUND_START_RECORDING = os.environ.get("VOICE_SOUND_START") or _DEFAULT_SOUND
SOUND_BEFORE_RESPONSE = os.environ.get("VOICE_SOUND_RESPONSE") or _DEFAULT_SOUND

import pyaudio
from faster_whisper import WhisperModel

# Audio recording parameters
DEFAULT_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16

# Voice activity detection
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5
MIN_RECORDING_DURATION = 0.5

# Whisper model
MODEL_SIZE = "base"
POST_TTS_COOLDOWN = 1.0

# Lock file for TTS synchronization
TTS_LOCK_FILE = "/tmp/tts-playing.lock"
DEFAULT_TTS_COMMAND = "~/.local/bin/tts"

# Minimum text length to be considered meaningful
MIN_TEXT_LENGTH = 3

# Noise patterns to filter out obvious hallucinations/fillers
NOISE_PATTERNS = [
    "嗯",
    "啊",
    "呃",
    "哦",
    "唔",
    "额",
    "嗯嗯",
    "啊啊",
    "谢谢观看",
    "感谢观看",
    "订阅",
    "点赞",
    "关注",
    "字幕",
    "翻译",
    "校对",
    "时间轴",
    "后期",
    "thank you",
    "thanks for watching",
    "subscribe",
    "please subscribe",
    "like and subscribe",
    "♪",
    "music",
    "音乐",
    "...",
    "。。。",
    "，，，",
]


def play_sound_effect(sound_path: str) -> bool:
    """Play a sound effect using available audio tools."""
    if not sound_path or not os.path.exists(sound_path):
        return False

    # Try multiple methods to play sound
    methods = [
        # Method 1: paplay (PulseAudio)
        (lambda p: subprocess.Popen(
            ["paplay", p],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ), ".wav"),
        # Method 2: ogg123 (for .oga files)
        (lambda p: subprocess.Popen(
            ["ogg123", "-q", p],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ), ".oga"),
        # Method 3: aplay (ALSA)
        (lambda p: subprocess.Popen(
            ["aplay", "-q", p],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ), ".wav"),
    ]

    ext = os.path.splitext(sound_path)[1].lower()

    for method, supported_ext in methods:
        if ext != supported_ext:
            continue
        try:
            proc = method(sound_path)
            threading.Thread(target=lambda: proc.wait(), daemon=True).start()
            return True
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return False


def has_repeated_chars(text: str, threshold: float = 0.7) -> bool:
    """Detect if text consists mostly of repeated single character."""
    if not text or len(text) < 4:
        return False

    # Count most common character
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1

    max_count = max(char_counts.values())
    return max_count / len(text) >= threshold


def is_meaningful_text(text: str) -> bool:
    if not text:
        return False

    text = text.strip()
    if len(text) < MIN_TEXT_LENGTH:
        return False

    if "简体中文" in text:
        return False

    # Filter repeated characters like "证证证证证证" or "啊啊啊啊啊啊啊啊啊啊啊啊啊"
    if has_repeated_chars(text):
        return False

    text_lower = text.lower()
    for pattern in NOISE_PATTERNS:
        if text_lower == pattern.lower() or text == pattern:
            return False
        if len(text) < 10 and pattern.lower() in text_lower:
            return False

    return True


def parse_json_output(stdout: str) -> dict | None:
    """Try to parse OpenClaw JSON output even if extra logs exist."""
    if not stdout:
        return None

    text = stdout.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def extract_replies(payload: dict) -> list[str]:
    """Extract all replies from OpenClaw response (supports multiple messages)."""
    replies = []
    if not isinstance(payload, dict):
        return replies

    # Direct fields
    for key in ["reply", "response", "output", "text"]:
        if key in payload and isinstance(payload[key], str) and payload[key].strip():
            replies.append(payload[key].strip())
            break

    # Nested result
    result = payload.get("result")
    if isinstance(result, dict):
        for key in ["reply", "response", "output", "text"]:
            if key in result and isinstance(result[key], str) and result[key].strip():
                replies.append(result[key].strip())
                break

        # Multiple payloads
        payloads = result.get("payloads")
        if isinstance(payloads, list):
            for entry in payloads:
                if isinstance(entry, dict):
                    for key in ["text", "message", "content"]:
                        if key in entry and isinstance(entry[key], str) and entry[key].strip():
                            replies.append(entry[key].strip())
                            break

    # Deduplicate while preserving order
    seen = set()
    unique_replies = []
    for r in replies:
        if r not in seen:
            seen.add(r)
            unique_replies.append(r)

    return unique_replies


def extract_run_status(payload: dict) -> tuple[str | None, str | None, str | None]:
    """Extract high-level run status fields for logging."""
    if not isinstance(payload, dict):
        return None, None, None
    run_id = payload.get("runId")
    status = payload.get("status")
    summary = payload.get("summary")
    return run_id, status, summary


def sanitize_tts_text(text: str) -> str:
    """Clean markdown/emojis/noise so TTS sounds natural."""
    if not text:
        return ""

    cleaned = text
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"~~(.*?)~~", r"\1", cleaned)
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)

    allowed_punct = set(" ,.!?;:-()[]{}'\"，。！？；：、（）")
    output = []
    for char in cleaned:
        if char.isspace() or char in allowed_punct:
            output.append(char)
            continue
        category = unicodedata.category(char)
        if category.startswith("L") or category.startswith("N"):
            output.append(char)

    cleaned = "".join(output)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class TTSProvider:
    """Base class for TTS providers."""

    def is_available(self) -> bool:
        raise NotImplementedError

    def playback(self, text: str, stop_checker=None) -> bool:
        raise NotImplementedError


class XiaoxiaoTTS(TTSProvider):
    """Xiaoxiao TTS using local command."""

    def __init__(self, command_path: str = DEFAULT_TTS_COMMAND):
        self.command_path = os.path.expanduser(command_path)

    def is_available(self) -> bool:
        return os.path.isfile(self.command_path) and os.access(self.command_path, os.X_OK)

    def playback(self, text: str, stop_checker=None) -> bool:
        if not text or not self.is_available():
            return False

        process = None
        try:
            with open(TTS_LOCK_FILE, "w", encoding="utf-8") as lock_file:
                lock_file.write(str(os.getpid()))

            process = subprocess.Popen(
                [self.command_path, text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )

            while True:
                return_code = process.poll()
                if return_code is not None:
                    return return_code == 0

                if stop_checker and stop_checker():
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=2)
                    return False

                time.sleep(0.05)
        except Exception:
            if process and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except Exception:
                    pass
            return False
        finally:
            try:
                os.unlink(TTS_LOCK_FILE)
            except FileNotFoundError:
                pass


class STTModel:
    """Base class for Speech-to-Text models."""

    def transcribe(self, audio_data: bytes, sample_rate: int) -> str:
        raise NotImplementedError

    def close(self):
        pass


class WhisperSTT(STTModel):
    """Whisper STT model using faster-whisper."""

    def __init__(self, model_size: str = MODEL_SIZE, language: str = "zh"):
        self.model_size = model_size
        self.language = language
        print(f"Loading Whisper model ({model_size})...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Model loaded")

    def transcribe(self, audio_data: bytes, sample_rate: int) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file_obj:
            temp_path = file_obj.name
            with wave.open(file_obj, "wb") as wav:
                wav.setnchannels(CHANNELS)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio_data)

        try:
            segments, _ = self.model.transcribe(
                temp_path,
                language=self.language,
                initial_prompt="以下是普通话的句子，使用简体中文。" if self.language == "zh" else None,
            )
            return "".join(segment.text for segment in segments).strip()
        finally:
            os.unlink(temp_path)


class KeyboardMonitor:
    def __init__(self):
        self.enabled = False
        self.fd = None
        self.old_settings = None

    def start(self):
        if not sys.stdin.isatty():
            return
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.enabled = True

    def stop(self):
        if self.enabled and self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        self.enabled = False

    def consume_space_pressed(self) -> bool:
        if not self.enabled:
            return False
        if self.fd is None:
            return False
        pressed = False
        try:
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0)
                if not ready:
                    break
                ch = os.read(self.fd, 1).decode("utf-8", errors="ignore")
                if ch == " ":
                    pressed = True
        except Exception:
            return False
        return pressed


class VoiceListener:
    def __init__(
        self,
        device_index=None,
        stt_model: STTModel | None = None,
        tts_provider: TTSProvider | None = None,
        session_id=None,
        channel="telegram",
        target=None,
        deliver=False,
        agent_id=None,
        timeout=120,
        gateway_port=None,
        gateway_token=None,
        tts_enabled=True,
        tts_command=DEFAULT_TTS_COMMAND,
        post_tts_cooldown=POST_TTS_COOLDOWN,
    ):
        self.device_index = device_index
        self.running = False
        self.channel = channel
        self.target = target
        self.deliver = deliver
        self.agent_id = agent_id
        self.timeout = timeout
        self.gateway_port = gateway_port
        self.gateway_token = gateway_token
        self.tts_enabled = tts_enabled
        self.post_tts_cooldown = post_tts_cooldown
        self.resume_listen_at = 0.0
        self.session_id = session_id or f"voice-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

        # Initialize TTS provider (once, reusable)
        self.tts = tts_provider if tts_provider else XiaoxiaoTTS(tts_command)

        # Initialize STT model (once, reusable) - only if not provided and needed
        self.stt = stt_model

        self.keyboard = KeyboardMonitor()

        # Only initialize audio if STT is needed
        if self.stt is not None:
            self.audio = pyaudio.PyAudio()
        else:
            self.audio = None
        self.rate = self._get_device_rate()
        print(f"Using sample rate: {self.rate} Hz")
        print(f"OpenClaw session: {self.session_id}")
        if self.tts_enabled and self.tts.is_available():
            print(f"TTS enabled: {getattr(self.tts, 'command_path', type(self.tts).__name__)}")
        elif self.tts_enabled:
            print(f"TTS not available")

    def _get_device_rate(self) -> int:
        if self.device_index is not None:
            info = self.audio.get_device_info_by_index(self.device_index)
            device_rate = int(info["defaultSampleRate"])
            for rate in [DEFAULT_RATE, device_rate]:
                try:
                    supported = self.audio.is_format_supported(
                        rate,
                        input_device=self.device_index,
                        input_channels=CHANNELS,
                        input_format=FORMAT,
                    )
                    if supported:
                        return rate
                except ValueError:
                    continue
            return device_rate
        return DEFAULT_RATE

    def list_devices(self):
        print("\nAvailable audio input devices:")
        print("-" * 50)
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if int(info["maxInputChannels"]) > 0:
                print(f"  [{i}] {info['name']}")
                print(
                    f"      Channels: {info['maxInputChannels']}, Rate: {info['defaultSampleRate']}"
                )
        print("-" * 50)

    def is_speech(self, audio_data: bytes) -> bool:
        import struct

        samples = struct.unpack(f"{len(audio_data) // 2}h", audio_data)
        amplitude = max(abs(s) for s in samples) if samples else 0
        return amplitude > SILENCE_THRESHOLD

    def record_until_silence(self, stream) -> bytes:
        frames = []
        silent_chunks = 0
        chunks_per_second = self.rate // CHUNK
        max_silent_chunks = int(SILENCE_DURATION * chunks_per_second)

        # Play sound effect when starting to record
        if SOUND_START_RECORDING:
            play_sound_effect(SOUND_START_RECORDING)

        print("Recording...", end="", flush=True)

        while self.running:
            if os.path.exists(TTS_LOCK_FILE):
                print("\nTTS active, stop recording")
                break

            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

                if self.is_speech(data):
                    silent_chunks = 0
                    print(".", end="", flush=True)
                else:
                    silent_chunks += 1

                if (
                    silent_chunks >= max_silent_chunks
                    and len(frames) > chunks_per_second * MIN_RECORDING_DURATION
                ):
                    break

            except Exception as err:
                print(f"\nRecording error: {err}")
                break

        print(" done")
        return b"".join(frames)

    def transcribe(self, audio_data: bytes) -> str:
        return self.stt.transcribe(audio_data, self.rate)

    def build_openclaw_command(self, text: str) -> list[str]:
        cmd = [
            "openclaw",
            "--no-color",
            "agent",
            "--session-id",
            self.session_id,
            "--message",
            text,
            "--json",
            "--timeout",
            str(self.timeout),
        ]

        if self.agent_id:
            cmd.extend(["--agent", self.agent_id])

        if self.deliver:
            cmd.append("--deliver")
            if self.channel:
                cmd.extend(["--reply-channel", self.channel])
            if self.target:
                cmd.extend(["--reply-to", self.target])

        return cmd

    def build_openclaw_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.gateway_port:
            env["OPENCLAW_GATEWAY_PORT"] = str(self.gateway_port)
        if self.gateway_token:
            env["OPENCLAW_GATEWAY_TOKEN"] = self.gateway_token
        return env

    def log_openclaw_status(self, payload: dict):
        run_id, status, summary = extract_run_status(payload)
        status_parts = []
        if run_id:
            status_parts.append(f"runId={run_id}")
        if status:
            status_parts.append(f"status={status}")
        if summary:
            status_parts.append(f"summary={summary}")
        if status_parts:
            print("OpenClaw status: " + ", ".join(status_parts))

    def ask_openclaw(self, text: str) -> list[str]:
        """Send text to OpenClaw and return list of replies (supports multiple messages)."""
        cmd = self.build_openclaw_command(text)

        try:
            import shutil

            start_at = time.monotonic()

            # Check if openclaw exists
            openclaw_path = shutil.which("openclaw")
            if not openclaw_path:
                print("openclaw command not found in PATH")
                return []

            print("Sending to OpenClaw...", end=" ", flush=True)

            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.build_openclaw_env(),
            )

            # Sending done, now waiting for reply
            print("Waiting for reply...")

            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.timeout + 10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                print("OpenClaw timeout")
                return []

            duration = time.monotonic() - start_at
            print(f"OpenClaw response received in {duration:.1f}s")

            # Create a result object similar to subprocess.run
            class Result:
                pass

            result = Result()
            result.returncode = process.returncode
            result.stdout = stdout
            result.stderr = stderr
        except FileNotFoundError:
            print("openclaw command not found in PATH")
            return []

        if result.returncode != 0:
            err = (result.stderr or result.stdout).strip()
            print(f"OpenClaw error: {err}")
            return []

        payload = parse_json_output(result.stdout)
        if not payload:
            print("OpenClaw status: parse_failed (non-JSON response)")
            return []
        self.log_openclaw_status(payload)

        replies = extract_replies(payload)
        if not replies:
            print("OpenClaw status: no_reply_extracted")
        return replies

    def speak(self, text: str):
        if not self.tts_enabled:
            return
        if not self.tts.is_available():
            print(f"TTS command not available: {self.tts.command_path}")
            return

        # Play sound effect before OpenClaw response
        if SOUND_BEFORE_RESPONSE:
            play_sound_effect(SOUND_BEFORE_RESPONSE)

        tts_text = sanitize_tts_text(text)
        if not tts_text:
            print("TTS skipped: no speakable text after sanitizing")
            return

        print("Playing TTS...")
        print("Press SPACE to stop TTS")
        if not self.tts.playback(tts_text, stop_checker=self.keyboard.consume_space_pressed):
            print("TTS playback stopped or failed")

    def flush_input(self, stream, seconds: float):
        end_time = time.monotonic() + max(0.0, seconds)
        while self.running and time.monotonic() < end_time:
            try:
                stream.read(CHUNK, exception_on_overflow=False)
            except Exception:
                pass

    def process_transcript(self, transcript: str, stream):
        if transcript and is_meaningful_text(transcript):
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] You: {transcript}")
            replies = self.ask_openclaw(transcript)
            if replies:
                for i, reply in enumerate(replies):
                    print(f"[{now}] OpenClaw ({i+1}/{len(replies)}): {reply}")
                    sys.stdout.flush()
                    self.speak(reply)
                    # Add small delay between multiple responses
                    if i < len(replies) - 1:
                        time.sleep(0.3)
                self.resume_listen_at = time.monotonic() + self.post_tts_cooldown
                self.flush_input(stream, self.post_tts_cooldown)
            else:
                print(f"[{now}] OpenClaw returned no parsable reply")
            return

        if transcript:
            print(f"Filtered: {transcript}")
        else:
            print("No speech detected")

    def start(self):
        self.running = True
        self.keyboard.start()

        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=CHUNK,
        )

        print("\nVoice listener started")
        print("=" * 50)
        print("Speak into your microphone. Press Ctrl+C to stop.")
        print("=" * 50)

        try:
            while self.running:
                if os.path.exists(TTS_LOCK_FILE):
                    time.sleep(0.1)
                    continue

                if time.monotonic() < self.resume_listen_at:
                    self.flush_input(stream, 0.1)
                    continue

                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                except Exception:
                    continue

                if self.is_speech(data):
                    while self.running and os.path.exists(TTS_LOCK_FILE):
                        time.sleep(0.1)
                    if not self.running:
                        break

                    audio_data = data + self.record_until_silence(stream)
                    print("Transcribing...")
                    text = self.transcribe(audio_data)
                    self.process_transcript(text, stream)

        except KeyboardInterrupt:
            print("\nStopping listener")
        finally:
            self.running = False
            self.keyboard.stop()
            stream.stop_stream()
            stream.close()

    def __del__(self):
        if hasattr(self, "stt") and self.stt:
            self.stt.close()
        if hasattr(self, "audio") and self.audio:
            self.audio.terminate()


def run_text_mode(args):
    """Run in text input mode - type messages instead of voice."""
    print("\nText mode - Type your message and press Enter")
    print("=" * 50)
    print("Commands:")
    print("  /quit or /q  - Exit")
    print("  /clear or /c - Clear screen")
    print("=" * 50)

    # Create TTS provider if enabled
    tts_provider = XiaoxiaoTTS(args.tts_command) if not args.no_tts else None

    # Create a simple text-based listener
    listener = VoiceListener(
        device_index=None,
        stt_model=None,  # No STT needed for text mode
        tts_provider=tts_provider,
        session_id=args.session_id,
        channel=args.channel,
        target=args.target,
        deliver=args.deliver,
        agent_id=args.agent,
        timeout=args.timeout,
        gateway_port=args.gateway_port,
        gateway_token=args.gateway_token,
        tts_enabled=not args.no_tts,
        post_tts_cooldown=0,  # No cooldown needed for text
    )

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/q"):
                print("Goodbye!")
                break

            if user_input.lower() in ("/clear", "/c"):
                os.system("clear" if os.name != "nt" else "cls")
                continue

            if user_input.lower() == "stt":
                print("Switching to voice mode...")
                return "SWITCH_TO_VOICE"

            now = datetime.now().strftime("%H:%M:%S")
            replies = listener.ask_openclaw(user_input)

            if replies:
                for i, reply in enumerate(replies):
                    print(f"[{now}] OpenClaw ({i+1}/{len(replies)}): {reply}")
                    if listener.tts_enabled and tts_provider and tts_provider.is_available():
                        listener.speak(reply)
                    # Small delay between multiple responses
                    if i < len(replies) - 1:
                        time.sleep(0.3)
            else:
                print(f"[{now}] OpenClaw returned no parsable reply")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def create_stt_model(model_name: str) -> STTModel:
    """Factory function to create STT model by name."""
    model_lower = model_name.lower()

    # Whisper models
    if model_lower.startswith("whisper-") or model_lower in ("tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"):
        whisper_size = model_lower.replace("whisper-", "") if model_lower.startswith("whisper-") else model_lower
        return WhisperSTT(model_size=whisper_size)

    # Default to whisper base
    return WhisperSTT(model_size=MODEL_SIZE)


def main():
    parser = argparse.ArgumentParser(description="Voice listener for OpenClaw Gateway")
    parser.add_argument("--list", action="store_true", help="List audio devices")
    parser.add_argument("--device", type=int, help="Audio device index")
    parser.add_argument("--model", default=f"whisper-{MODEL_SIZE}", help="STT model (whisper-tiny/base/small/medium/large)")
    parser.add_argument("--session-id", help="OpenClaw session id (fixed)")
    parser.add_argument("--agent", help="OpenClaw agent id")
    parser.add_argument("--timeout", type=int, default=120, help="OpenClaw timeout seconds")
    parser.add_argument("--gateway-port", type=int, help="Override OPENCLAW_GATEWAY_PORT")
    parser.add_argument("--gateway-token", help="Override OPENCLAW_GATEWAY_TOKEN")
    parser.add_argument("--tts-command", default=DEFAULT_TTS_COMMAND, help="Local TTS command path")
    parser.add_argument("--no-tts", action="store_true", help="Disable local TTS playback")
    parser.add_argument("--post-tts-cooldown", type=float, default=POST_TTS_COOLDOWN, help="Seconds to discard mic input after TTS")
    parser.add_argument(
        "--deliver",
        action="store_true",
        help="Deliver agent reply to channel via OpenClaw",
    )
    parser.add_argument(
        "--channel",
        default="telegram",
        help="reply channel when --deliver is enabled",
    )
    parser.add_argument(
        "--target",
        help="reply target when --deliver is enabled (chat id/user)",
    )
    parser.add_argument(
        "-t", "--text",
        action="store_true",
        help="Text input mode (type instead of voice)",
    )
    args = parser.parse_args()

    # Text input mode
    if args.text:
        result = run_text_mode(args)
        if result == "SWITCH_TO_VOICE":
            # Clear the --text flag and run voice mode
            args.text = False
        else:
            return

    # Create STT model (once, reusable)
    stt_model = create_stt_model(args.model)

    # Create TTS provider (once, reusable)
    tts_provider = XiaoxiaoTTS(args.tts_command) if not args.no_tts else None

    listener = VoiceListener(
        device_index=args.device,
        stt_model=stt_model,
        tts_provider=tts_provider,
        session_id=args.session_id,
        channel=args.channel,
        target=args.target,
        deliver=args.deliver,
        agent_id=args.agent,
        timeout=args.timeout,
        gateway_port=args.gateway_port,
        gateway_token=args.gateway_token,
        tts_enabled=not args.no_tts,
        post_tts_cooldown=args.post_tts_cooldown,
    )

    if args.list:
        listener.list_devices()
        return

    listener.start()


if __name__ == "__main__":
    main()
