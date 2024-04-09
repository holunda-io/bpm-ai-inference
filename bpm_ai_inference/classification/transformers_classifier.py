import logging

from bpm_ai_core.classification.zero_shot_classifier import ZeroShotClassifier, ClassificationResult
from typing_extensions import override

from bpm_ai_inference.util.optimum import get_optimized_model

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer, ORTQuantizer

    has_transformers = True
except ImportError:
    has_transformers = False

logger = logging.getLogger(__name__)

DEFAULT_MODEL_EN = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
DEFAULT_MODEL_MULTI = "MoritzLaurer/bge-m3-zeroshot-v2.0"


class TransformersClassifier(ZeroShotClassifier):
    """
    Local zero-shot classification model based on Huggingface transformers library.

    To use, you should have the ``transformers`` and ``optimum`` python packages installed.
    """

    def __init__(self, model: str = DEFAULT_MODEL_EN):
        if not has_transformers:
            raise ImportError('transformers or optimum not installed')

        model, self.tokenizer = get_optimized_model(model, "zero-shot-classification")

        self.zeroshot_classifier = pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=self.tokenizer
        )

    @override
    async def _do_classify(
            self,
            text: str,
            classes: list[str],
            hypothesis_template: str | None = None
    ) -> ClassificationResult:
        input_tokens = len(self.tokenizer.encode(text))
        max_tokens = self.tokenizer.model_max_length
        logger.debug(f"Input tokens: {input_tokens}")
        if input_tokens > max_tokens:
            logger.warning(
                f"Input tokens exceed max model context size: {input_tokens} > {max_tokens}. Input will be truncated."
            )

        prediction = self.zeroshot_classifier(
            text,
            candidate_labels=classes,
            hypothesis_template=hypothesis_template or "This example is about {}",
            multi_label=False
        )
        # Zip the labels and scores together and find the label with the max score
        labels_scores = list(zip(prediction['labels'], prediction['scores']))
        max_label, max_score = max(labels_scores, key=lambda x: x[1])

        return ClassificationResult(
            max_label=max_label,
            max_score=max_score,
            labels_scores=labels_scores
        )
