# chat_gateway

Voice listener that transcribes microphone input and talks to OpenClaw through the Gateway, using a dedicated `session-id`.

## Prerequisites

- `openclaw` installed and configured
- Gateway running (`openclaw gateway start` or foreground `openclaw gateway run`)
- Microphone available

## Run

```bash
cd ~/chat_gateway
uv run voice_listener.py --list
uv run voice_listener.py
```

### Recommended on this machine

```bash
cd ~/chat_gateway
source ~/.zshrc
uv run voice_listener.py --device 6 --session-id voice-albert-001
```

`--device 6` is the `pulse` input device (usually the most stable choice).
By default, replies are spoken locally via `~/.local/bin/tts` (Xiaoxiao voice).

### TTS options

```bash
# Disable local playback
uv run voice_listener.py --device 6 --no-tts

# Use another TTS command path
uv run voice_listener.py --device 6 --tts-command ~/.local/bin/xiaoxiao-tts
```

### Fixed independent session

```bash
uv run voice_listener.py --session-id voice-albert-001
```

### Deliver reply back to channel (optional)

```bash
uv run voice_listener.py \
  --session-id voice-albert-001 \
  --deliver \
  --channel telegram \
  --target 1115213761
```

### If gateway port/token is custom

```bash
uv run voice_listener.py \
  --gateway-port 18790 \
  --gateway-token "your_gateway_token"
```

## Notes

- Without `--session-id`, the script auto-generates one session per run.
- The script prints both your transcript and parsed OpenClaw reply.
