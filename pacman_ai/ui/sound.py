"""
Audio sound effects manager for Pac-Man using pygame.mixer.

Handles startup music, alternating waka-waka dot eating, power pellets,
ghost eating, and death sound effects with graceful fallback if audio devices are unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pygame

logger = logging.getLogger(__name__)


class SoundManager:
    """Manages playback of authentic retro sound effects."""

    def __init__(self, audio_dir: Optional[Path | str] = None):
        self.enabled = False
        if audio_dir is None:
            self.audio_dir = Path(__file__).resolve().parent.parent / "audio"
        else:
            self.audio_dir = Path(audio_dir)

        # Sounds dictionary
        self._sounds: dict[str, pygame.mixer.Sound] = {}
        self._chomp_index = 0
        self._start_channel: Optional[pygame.mixer.Channel] = None

        self._init_mixer()

    def _init_mixer(self) -> None:
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(8)
            self._start_channel = pygame.mixer.Channel(0)
            self._load_sounds()
            self.enabled = True
        except Exception as e:
            logger.warning("Could not initialize pygame.mixer (running headless or audio unavailable): %s", e)
            self.enabled = False

    def _load_sounds(self) -> None:
        sound_files = {
            "start": "start.wav",
            "chomp0": "eat_dot_0.wav",
            "chomp1": "eat_dot_1.wav",
            "power": "fright.wav",
            "death": "death_0.wav",
            "eat_ghost": "eat_ghost.wav",
            "fruit": "eat_fruit.wav",
        }

        for name, filename in sound_files.items():
            path = self.audio_dir / filename
            if path.exists():
                try:
                    snd = pygame.mixer.Sound(str(path))
                    snd.set_volume(0.65)
                    self._sounds[name] = snd
                except Exception as e:
                    logger.warning("Failed to load sound %s: %s", filename, e)

    def play_start(self) -> None:
        """Play authentic arcade game start music."""
        if not self.enabled or "start" not in self._sounds:
            return
        try:
            if self._start_channel:
                self._start_channel.play(self._sounds["start"])
            else:
                self._sounds["start"].play()
        except Exception:
            pass

    def is_start_playing(self) -> bool:
        """Check if startup music is currently playing."""
        if not self.enabled or self._start_channel is None:
            return False
        return self._start_channel.get_busy()

    def play_chomp(self) -> None:
        """Play alternating waka-waka chomp sound when eating pellets."""
        if not self.enabled:
            return
        key = f"chomp{self._chomp_index}"
        self._chomp_index = (self._chomp_index + 1) % 2
        snd = self._sounds.get(key) or self._sounds.get("chomp0")
        if snd:
            try:
                snd.play()
            except Exception:
                pass

    def play_power(self) -> None:
        """Play sound when eating a power pellet."""
        if not self.enabled or "power" not in self._sounds:
            return
        try:
            self._sounds["power"].play()
        except Exception:
            pass

    def play_death(self) -> None:
        """Play sound when Pac-Man loses a life."""
        if not self.enabled or "death" not in self._sounds:
            return
        try:
            self.stop_all()
            self._sounds["death"].play()
        except Exception:
            pass

    def play_eat_ghost(self) -> None:
        """Play sound when eating a frightened ghost."""
        if not self.enabled or "eat_ghost" not in self._sounds:
            return
        try:
            self._sounds["eat_ghost"].play()
        except Exception:
            pass

    def stop_all(self) -> None:
        """Stop all currently playing sound effects."""
        if not self.enabled:
            return
        try:
            pygame.mixer.stop()
        except Exception:
            pass
