import logging

from bpm_ai_core.text_classification.text_classifier import TextClassifier, ClassificationResult
from bpm_ai_core.util.caching import cachable
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


@cachable()
class TransformersClassifier(TextClassifier):
    """
    Local (zero-shot) text classification model based on Huggingface transformers library.

    To use, you should have the ``transformers`` and ``optimum`` python packages installed.
    """

    def __init__(self, model: str = DEFAULT_MODEL_EN, zero_shot: bool = True):
        if not has_transformers:
            raise ImportError('transformers or optimum not installed')

        task = "zero-shot-classification" if zero_shot else "text-classification"

        model, self.tokenizer = get_optimized_model(model, task)

        self.text_classifier = pipeline(
            task,
            model=model,
            tokenizer=self.tokenizer
        )

    @override
    async def _do_classify(
            self,
            text: str,
            classes: list[str] = None,
            hypothesis_template: str | None = None,
            multi_label: bool = False
    ) -> ClassificationResult:
        input_tokens = len(self.tokenizer.encode(text))
        max_tokens = self.tokenizer.model_max_length
        logger.debug(f"Input tokens: {input_tokens}")
        if input_tokens > max_tokens:
            logger.warning(
                f"Input tokens exceed max model context size: {input_tokens} > {max_tokens}. Input will be truncated."
            )

        prediction = self.text_classifier(
            text,
            **({"candidate_labels": classes} if classes else {}),
            **({"hypothesis_template": hypothesis_template or "This example is about {}"} if classes else {}),
            **({"multi_label": multi_label} if classes else {}),
        )
        # Zip the labels and scores together and find the label with the max score
        if 'labels' in prediction:
            labels_scores = list(zip(prediction['labels'], prediction['scores']))
        else:
            labels_scores = [(p['label'], p['score']) for p in prediction]
        max_label, max_score = max(labels_scores, key=lambda x: x[1])

        return ClassificationResult(
            max_label=max_label,
            max_score=max_score,
            labels_scores=labels_scores
        )
