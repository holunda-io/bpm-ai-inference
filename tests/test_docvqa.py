from bpm_ai_core.llm.common.blob import Blob

from bpm_ai_inference.question_answering.transformers_docvqa import TransformersDocVQA


async def test_docvqa():
    model = TransformersDocVQA()

    question = "What is the total?"

    result = await model.answer(
        context_str_or_blob=Blob.from_path_or_url("invoice-simple.webp"),
        question=question
    )

    assert "300" in result.answer