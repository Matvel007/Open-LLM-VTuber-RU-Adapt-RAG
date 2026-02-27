"""Silero TTS engine - local Russian TTS using snakers4/silero-models via torch.hub."""

import os

import soundfile as sf
import torch
from loguru import logger

from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    """Local TTS engine using Silero models (Russian v5_1)."""

    def __init__(
        self,
        language: str = "ru",
        model_id: str = "v5_1_ru",
        speaker: str = "xenia",
        sample_rate: int = 48000,
        device: str = "cpu",
        put_accent: bool = True,
        put_yo: bool = True,
    ):
        """Initialize Silero TTS engine.

        Args:
            language: Language code (e.g. 'ru' for Russian).
            model_id: Silero model ID (e.g. 'v5_1_ru', 'v5_ru').
            speaker: Speaker/voice name within the model (e.g. 'xenia', 'baya').
            sample_rate: Output sample rate (8000, 24000, or 48000 for v5_1_ru).
            device: Device for inference ('cpu' or 'cuda').
            put_accent: Add Russian stress marks automatically.
            put_yo: Use letter ё where appropriate.
        """
        self.language = language
        self.model_id = model_id
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.put_accent = put_accent
        self.put_yo = put_yo

        self.file_extension = "wav"
        self.new_audio_dir = "cache"

        if not os.path.exists(self.new_audio_dir):
            os.makedirs(self.new_audio_dir)

        self.model = self._load_model()

    def _load_model(self):
        """Load Silero TTS model via torch.hub."""
        logger.info(
            f"Loading Silero TTS: language={self.language}, model={self.model_id}"
        )
        model, _example_text = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=self.language,
            speaker=self.model_id,
        )
        model.to(self.device)
        logger.info("Silero TTS model loaded successfully")
        return model

    def generate_audio(
        self, text: str, file_name_no_ext: str | None = None
    ) -> str | None:
        """Generate speech audio file using Silero TTS.

        Args:
            text: The text to convert to speech.
            file_name_no_ext: Optional filename without extension.

        Returns:
            Path to the generated audio file, or None on failure.
        """
        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)

        try:
            audio = self.model.apply_tts(
                text=text,
                speaker=self.speaker,
                sample_rate=self.sample_rate,
                put_accent=self.put_accent,
                put_yo=self.put_yo,
            )

            # apply_tts returns numpy array
            sf.write(
                file_name,
                audio,
                samplerate=self.sample_rate,
                subtype="PCM_16",
            )

            return file_name

        except Exception as e:
            logger.critical(f"Silero TTS failed to generate audio: {e}")
            return None
