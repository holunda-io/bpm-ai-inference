from bpm_ai_inference.ocr.tesseract import TesseractOCR
from bpm_ai_inference.question_answering.transformers_qa import TransformersExtractiveQA


async def test_qa():
    context = "My name is John and I live in Hawaii"
    question = "Where does John live?"
    expected = "Hawaii"

    qa = TransformersExtractiveQA()
    actual = await qa.answer(context, question)

    assert actual.answer.strip() == expected


async def test_qa_ocr():
    ocr = TesseractOCR()
    ocr_result = await ocr.process("example-multipage.pdf")

    question = "What kind of PDF is this?"

    qa = TransformersExtractiveQA()
    actual = await qa.answer(ocr_result.full_text, question)

    assert actual.answer.strip() == "Dummy"


async def test_qa_unanswerable():
    context = "My name is John and I live in Hawaii"
    question = "How much is the fish?"

    qa = TransformersExtractiveQA()
    actual = await qa.answer(context, question, confidence_threshold=0.1)

    assert actual is None
