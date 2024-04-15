from bpm_ai_inference.ocr.tesseract import TesseractOCR


async def test_tesseract_image():
    ocr = TesseractOCR()

    result = await ocr.process("files/example-text.png")

    assert "example" in result.full_text

    #draw_boxes_on_image(Image.open("example.png"), result.pages[0].bboxes)


async def test_tesseract_pdf():
    ocr = TesseractOCR()

    result = await ocr.process("files/example-multipage.pdf")

    assert "Dummy PDF file" in result.full_text
