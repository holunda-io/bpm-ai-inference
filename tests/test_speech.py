from bpm_ai_inference.speech_recognition.faster_whisper import FasterWhisperASR


async def test_faster_whisper():
    fw = FasterWhisperASR()
    result = await fw.transcribe("files/example.mp3")
    assert result.text.lower().strip() == "looking with a half-fantastic curiosity to see whether the tender grass of early spring"
