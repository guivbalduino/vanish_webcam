"""Preload the Naruto/Goku audio and play it without blocking video frames."""

from __future__ import annotations

import math
import os
import random
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parent / "assets"
SOUND_PATH = ASSET_DIR / "vanish.MP3"
TELEPORT_SOUND_PATH = ASSET_DIR / "teleport.MP3"
TELEPORT_FALLBACK_PATH = Path(tempfile.gettempdir()) / "vanish_teleport.wav"
TELEPORT_SAMPLE_RATE = 44100


def synthesize_teleport_wav(
    path: Path, duration: float = 1.1, sample_rate: int = TELEPORT_SAMPLE_RATE
) -> Path:
    """Write a short rising/falling "zap" used when teleport.MP3 is absent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = max(1, int(duration * sample_rate))
    rng = random.Random(5150)
    samples = bytearray()
    phase = 0.0
    for index in range(count):
        progress = index / count
        frequency = (
            240.0
            + 1500.0 * math.sin(math.pi * min(1.0, progress * 1.35))
            + 700.0 * progress
        )
        phase += 2.0 * math.pi * frequency / sample_rate
        envelope = min(1.0, progress * 18.0) * (1.0 - progress) ** 1.7
        tone = math.sin(phase) * 0.55 + math.sin(phase * 2.02) * 0.22
        noise = (rng.random() * 2.0 - 1.0) * 0.3 * (1.0 - progress)
        value = int(max(-1.0, min(1.0, (tone + noise) * envelope)) * 32767.0)
        samples += struct.pack("<h", value)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(bytes(samples))
    return path


def teleport_audio_path() -> Path:
    """Return teleport.MP3 when present, otherwise a synthesized WAV fallback."""
    if TELEPORT_SOUND_PATH.is_file():
        return TELEPORT_SOUND_PATH
    if (
        not TELEPORT_FALLBACK_PATH.is_file()
        or TELEPORT_FALLBACK_PATH.stat().st_size < 1024
    ):
        synthesize_teleport_wav(TELEPORT_FALLBACK_PATH)
    return TELEPORT_FALLBACK_PATH


class _SoundPlayer:
    """Shared mixer/ffplay plumbing for the bundled effects."""

    label = "effetto"

    def __init__(self, path: Path):
        path = Path(path)
        if not path.is_file():
            raise RuntimeError(f"Audio {self.label} non trovato: {path}")
        self.path = path

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        self.process = None
        self.ffplay = None
        try:
            from pygame import mixer
        except ImportError:
            self.ffplay = shutil.which("ffplay")
            if self.ffplay is None:
                raise RuntimeError(
                    f"Per riprodurre {path.name} installa pygame o ffplay."
                )
            return

        self.mixer = mixer
        self.owns_mixer = mixer.get_init() is None
        try:
            if self.owns_mixer:
                mixer.init(frequency=44100, size=-16, channels=2, buffer=256)
            # Decode once, before opening the webcam.
            self.sound = mixer.Sound(str(path))
        except Exception as error:
            if self.owns_mixer:
                mixer.quit()
            raise RuntimeError(f"Impossibile aprire l'audio {self.label}: {error}") from error

    def play(self) -> None:
        if self.ffplay is not None:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            try:
                self.process = subprocess.Popen(
                    [self.ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet",
                     "-i", str(self.path)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as error:
                raise RuntimeError(
                    f"Impossibile avviare l'audio {self.label}: {error}"
                ) from error
            return
        self.sound.stop()
        if self.sound.play() is None:
            raise RuntimeError(
                f"Impossibile avviare l'audio {self.label}: nessun canale libero."
            )

    def close(self) -> None:
        if self.ffplay is not None:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            return
        if self.owns_mixer:
            self.mixer.quit()
            self.owns_mixer = False


class VanishSound(_SoundPlayer):
    label = "Naruto"

    def __init__(self, path: Path = SOUND_PATH):
        super().__init__(path)


class TeleportSound(_SoundPlayer):
    label = "Goku"

    def __init__(self, path: Path | None = None):
        super().__init__(teleport_audio_path() if path is None else path)
