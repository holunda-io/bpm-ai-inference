import logging

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.question_answering.question_answering import QuestionAnswering, QAResult
from bpm_ai_core.util.caching import cachable
from typing_extensions import override

from bpm_ai_inference.util.optimum import get_optimized_model

try:
    from transformers import pipeline, AutoTokenizer
    has_transformers = True
except ImportError:
    has_transformers = False

logger = logging.getLogger(__name__)


@cachable()
class TransformersExtractiveQA(QuestionAnswering):
    """
    Local extractive question answering model based on Huggingface transformers library.

    To use, you should have the ``transformers`` and ``optimum`` python packages installed.
    """

    def __init__(self, model: str = "deepset/deberta-v3-large-squad2"):
        if not has_transformers:
            raise ImportError('transformers is not installed')

        model, self.tokenizer = get_optimized_model(model, "question-answering")

        self.qa_model = pipeline(
            "question-answering",
            model=model,
            tokenizer=self.tokenizer
        )

    @override
    async def _do_answer(
            self,
            context_str_or_blob: str | Blob,
            question: str
    ) -> QAResult:
        if not isinstance(context_str_or_blob, str):
            raise Exception('TransformersExtractiveQA only supports string input')
        else:
            context = context_str_or_blob

        tokens = self.tokenizer.encode(context + question)
        logger.debug(f"Input tokens: {len(tokens)}")

        prediction = self.qa_model(
            question=question,
            context=context
        )
        logger.debug(f"prediction: {prediction}")

        return QAResult(
            answer=prediction['answer'],
            score=prediction['score'],
            start_index=prediction['start'],
            end_index=prediction['end'],
        )
