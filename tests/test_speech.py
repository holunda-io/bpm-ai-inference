from bpm_ai_inference.speech_recognition.faster_whisper import FasterWhisperASR


async def test_faster_whisper():
    fw = FasterWhisperASR()
    text = await fw.transcribe("example.mp3")
    assert text.lower().strip() == "looking with a half-fantastic curiosity to see whether the tender grass of early spring"


async def test_faster_whisper_url():
    fw = FasterWhisperASR()
    text = await fw.transcribe("https://upload.wikimedia.org/wikipedia/commons/d/dd/Armstrong_Small_Step.ogg")
    assert "giant leap for mankind" in text.lower()
