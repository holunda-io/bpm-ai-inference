import io

from bpm_ai_core.speech_recognition.asr import ASRModel, ASRResult
from bpm_ai_core.util.caching import cachable
from typing_extensions import override

try:
    from faster_whisper import WhisperModel
    has_faster_whisper = True
except ImportError:
    has_faster_whisper = False


@cachable()
class FasterWhisperASR(ASRModel):
    """
    Local `OpenAI` Whisper Automatic Speech Recognition (ASR) model for transcribing audio.

    To use, you should have the ``faster_whisper`` python package installed.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        if not has_faster_whisper:
            raise ImportError('faster_whisper is not installed')
        self.model_size = model_size
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    @override
    async def _do_transcribe(self, audio: io.BytesIO, language: str = None) -> ASRResult:
        segments, info = self.model.transcribe(audio, language=language)
        full_text = "".join([s.text for s in list(segments)])
        return ASRResult(text=full_text)
