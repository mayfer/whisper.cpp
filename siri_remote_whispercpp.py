#!/usr/bin/env python3
"""
siri_remote_whisper.py  –  Streams Siri-Remote audio to whisper-stream-stdin

Prerequisites
• whisper-stream-stdin must be built (see examples/stream_stdin)
• A FIFO (named pipe) is created automatically in /tmp/whisper_audio.fifo
• The script starts whisper-stream-stdin once and keeps it alive
  – the Whisper model stays resident on the GPU / CPU between sessions

Usage (same PacketLogger pipe as the stand-alone decoder):

    sudo PacketLogger … convert -s -f nhdr |
        ./siri_remote_whisper.py C08N1RGN2330

The MAC/device-name filter argument is optional.

The script prints incremental transcripts in real-time.
"""

from __future__ import annotations

import os
import platform
import subprocess
import pathlib
import sys
import re
import wave
import json
import threading
import time
from pathlib import Path
from typing import List, Optional


# ── Auto-locate Home-brew libopus on macOS (before importing opuslib) ──────────
if platform.system() == "Darwin" and "OPUS_LIB" not in os.environ:
    try:
        brew_prefix = subprocess.check_output(["brew", "--prefix", "opus"], text=True).strip()
        dylib = pathlib.Path(brew_prefix, "lib/libopus.dylib")
        if dylib.exists():
            os.environ["OPUS_LIB"] = str(dylib)
    except Exception:
        pass

try:
    import opuslib
except ImportError:
    sys.stderr.write("✖︎ opuslib not installed –  pip install opuslib\n")
    sys.exit(1)


# ── Config ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000
CHANNELS      = 1
HEADER_MASK   = 0xC0        # top-2-bits mask for Opus header heuristic

WHISPER_BIN   = pathlib.Path("/Users/murat/Code/AI/whisper.cpp/build/bin/whisper-stream-stdin")
# Update if you wish a different model or parameters
WHISPER_ARGS  = ["--model", "/Users/murat/Code/AI/whisper.cpp/models/ggml-large-v3-turbo.bin"]

# Named pipe path – will be created if missing
FIFO_PATH     = pathlib.Path("/tmp/whisper_audio.fifo")


# ── Helpers ────────────────────────────────────────────────────────────────────
class Opus16kDecoder:
    """Very thin wrapper around opuslib.Decoder returning 16-bit PCM bytes."""

    def __init__(self):
        self._dec = opuslib.Decoder(SAMPLE_RATE, CHANNELS)

    def decode(self, packet: bytes) -> bytes:
        return self._dec.decode(packet, 640)  # 40 ms frame @ 16 kHz


def looks_like_header(tokens: list[str]) -> bool:
    return (
        len(tokens) >= 2
        and (int(tokens[0], 16) & HEADER_MASK) == 0x40
        and tokens[1] == "20"
    )


def is_voice_start(tail: str) -> bool:
    return tail.endswith("1B 39 00 20 00")


def is_voice_end(tail: str) -> bool:
    return tail.endswith("1B 39 00 00 00") or tail.endswith("1B 39 00 00")


hex_tail_pat = re.compile(r"([0-9A-F]{2}(?:\s+[0-9A-F]{2})+)$")


# Voice-related packets always contain 1B 39 (start/stop) or 1B 35 (audio)
def is_voice_packet(tokens: list[str]) -> bool:
    return any(tokens[i] == "1B" and tokens[i + 1] in ("39", "35") for i in range(len(tokens) - 1))


# ── Whisper process wrapper ────────────────────────────────────────────────────
class WhisperStreamer:
    """Starts whisper-stream-stdin once and feeds it through a FIFO."""

    def __init__(self, fifo: Path, bin_path: Path = WHISPER_BIN, extra_args: list[str] | None = None):
        self.fifo = fifo
        self._ensure_fifo()

        if not bin_path.exists():
            sys.stderr.write(f"✖︎ Whisper binary not found at {bin_path}\n")
            sys.exit(1)

        cmd = [str(bin_path)] + (extra_args or [])
        # Open FIFO for reading inside Popen so whisper gets it as stdin.
        fifo_read = open(self.fifo, "rb", buffering=0)
        self.proc = subprocess.Popen(cmd, stdin=fifo_read, stdout=subprocess.PIPE, text=True)

        # Background thread prints transcripts
        self._reader_thread = threading.Thread(target=self._stdout_reader, daemon=True)
        self._reader_thread.start()

        self._current_writer: Optional[io.BufferedWriter] = None

    def _ensure_fifo(self):
        if self.fifo.exists():
            if not self.fifo.is_fifo():
                raise RuntimeError(f"{self.fifo} exists and is not a FIFO")
        else:
            os.mkfifo(self.fifo)

    # ── Output reader ────────────────────────────────────────────────────────
    def _stdout_reader(self):
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                kind = msg.get("type", "?")
                text = msg.get("text", "")
                prefix = "[FINAL]" if kind == "final" else "[PART ]"
                print(f"{prefix} {text}")
            except json.JSONDecodeError:
                # Whisper sometimes prints diagnostic lines; forward them
                print("WHISPER:", line)

    # ── Session control ─────────────────────────────────────────────────────
    def start_session(self):
        """Open FIFO for writing (blocks until reader ready)."""
        if self._current_writer and not self._current_writer.closed:
            self._current_writer.close()
        self._current_writer = open(self.fifo, "wb", buffering=0)

    def write(self, pcm_s16: bytes):
        if self._current_writer is None:
            return  # not in session
        try:
            self._current_writer.write(pcm_s16)
        except BrokenPipeError:
            # Whisper crashed? abort session silently
            self._current_writer = None

    def end_session(self):
        if self._current_writer and not self._current_writer.closed:
            self._current_writer.close()
        self._current_writer = None

    def terminate(self):
        self.end_session()
        if self.proc.poll() is None:
            self.proc.terminate()


# ── Main packet-decoder loop ─────────────────────────────────────────────────


def main():
    # Optional MAC or device name filter
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    mac_filt, name_filt = "", ""
    if filt:
        if re.fullmatch(r"[0-9A-F]{2}([:\-]?[0-9A-F]{2}){5}", filt, re.I):
            mac_filt = filt.replace("-", ":").lower()
        else:
            name_filt = filt.lower()
        print(f"Filtering on: {filt}")

    opus_dec = Opus16kDecoder()
    whisper = WhisperStreamer(FIFO_PATH, extra_args=WHISPER_ARGS)

    collecting = False
    frame_frag: List[str] = []
    frames: List[str] = []
    header_streak = 0      # consecutive Opus headers while not collecting
    audio_seen = False

    def flush_frames():
        nonlocal frames, frame_frag, audio_seen
        if not audio_seen:
            frames.clear(); frame_frag.clear();
            return
        for fh in frames:
            data = bytes.fromhex(fh.replace(" ", ""))
            if len(data) <= data[0]:
                continue
            try:
                whisper.write(opus_dec.decode(data[1 : 1 + data[0]]))
            except opuslib.OpusError as e:
                print(e, file=sys.stderr)
        frames.clear(); frame_frag.clear(); audio_seen = False

    for raw in sys.stdin:
        line = raw.rstrip()
        if (
            (mac_filt and mac_filt not in line.lower())
            or (name_filt and name_filt not in line.lower())
            or " RECV " not in line
        ):
            continue

        m = hex_tail_pat.search(line)
        if not m:
            continue
        hex_tail = m.group(1)
        tokens = hex_tail.split()

        if is_voice_start(hex_tail):
            if collecting:
                flush_frames()
                whisper.end_session()
            print("Voice started …")
            whisper.start_session()
            collecting = True
            frame_frag.clear(); frames.clear(); audio_seen = False
            continue

        if is_voice_end(hex_tail):
            if not collecting:
                continue  # stray end
            print("Voice ended …")
            # flush last frames to Whisper
            flush_frames()
            whisper.end_session()
            collecting = False
            continue

        if looks_like_header(tokens) and is_voice_packet(tokens):
            has_b8 = "B8" in tokens  # Opus packet marker

            if not collecting:
                if not has_b8:
                    header_streak = 0
                    continue

                header_streak += 1
                if header_streak < 3:
                    continue
                # auto-start after 3 consecutive headers
                print("Voice auto-started (3 B8 headers)")
                whisper.start_session()
                collecting = True
                frame_frag.clear(); frames.clear(); audio_seen = False
            else:
                header_streak = 0

            if not has_b8:
                if is_voice_packet(tokens):
                    audio_seen = True
                frame_frag.extend(tokens)
                continue

            audio_seen = True
            try:
                b8 = tokens.index("B8")
            except ValueError:
                b8 = -1
            if b8 > 0:
                if frame_frag:
                    frames.append(" ".join(frame_frag))
                frame_frag = tokens[b8 - 1 :]
            else:
                frame_frag.extend(tokens)

        elif collecting and len(tokens) > 3:
            # Append non-header packets belonging to current frame
            frame_frag.extend(tokens[3:])

    # Clean termination on ctrl-C / EOF
    whisper.terminate()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
