import logging

from bpm_ai_core.token_classification.zero_shot_token_classifier import ZeroShotTokenClassifier, \
    TokenClassificationResult, TokenSpan
from bpm_ai_core.util.caching import cachable
from typing_extensions import override

try:
    from gliner import GLiNER
    has_gliner = True
except ImportError:
    has_gliner = False

logger = logging.getLogger(__name__)


@cachable()
class GlinerTokenClassifier(ZeroShotTokenClassifier):
    """
    Zero Shot Token Classifier based on GLiNER.

    To use, you should have the ``gliner`` python package installed.
    """

    def __init__(self, model: str = "urchade/gliner_large-v2.1"):
        self.model = GLiNER.from_pretrained(model)

    @override
    async def _do_classify(
            self,
            text: str,
            classes: list[str],
            confidence_threshold: float | None = None
    ) -> TokenClassificationResult:
        entities = self.model.predict_entities(
            text,
            classes,
            threshold=confidence_threshold or 0.5
        )

        return TokenClassificationResult(
            spans=[
                TokenSpan(
                    word=e["text"],
                    label=e["label"],
                    score=e["score"],
                    start=e["start"],
                    end=e["end"]
                )
                for e in entities
            ]
        )
