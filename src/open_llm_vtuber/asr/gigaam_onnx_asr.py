"""GigaAM ONNX ASR engine using onnx-asr library.

Supports Russian speech recognition with GigaAM v3 E2E CTC model.
"""

import numpy as np
from loguru import logger

from .asr_interface import ASRInterface


class VoiceRecognition(ASRInterface):
    """GigaAM ONNX ASR using onnx-asr library."""

    def __init__(
        self,
        model_path: str,
        model_type: str = "gigaam-v3-e2e-ctc",
        quantization: str | None = None,
    ) -> None:
        """Initialize GigaAM ONNX ASR.

        Args:
            model_path: Path to directory containing v3_e2e_ctc.onnx, v3_e2e_ctc_vocab.txt
                (rename model.onnx and vocab.txt from GigaAM download).
            model_type: Model type for onnx-asr ('gigaam-v3-e2e-ctc' or 'gigaam-v3-ctc').
            quantization: Model quantization ('None' or 'int8').
        """
        import onnx_asr

        self.model_path = model_path
        self.model_type = model_type
        self.SAMPLE_RATE = 16000

        logger.info(f"Loading GigaAM ONNX ASR from {model_path}")
        self.model = onnx_asr.load_model(
            model_type,
            path=model_path,
            quantization=quantization,
        )
        logger.info("GigaAM ONNX ASR loaded successfully")

    def transcribe_np(self, audio: np.ndarray) -> str:
        """Transcribe audio numpy array to text.

        Args:
            audio: Float32 numpy array of audio at 16kHz.

        Returns:
            Transcribed text string.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        result = self.model.recognize(
            waveform=audio,
            sample_rate=self.SAMPLE_RATE,
        )
        return result if isinstance(result, str) else str(result)
