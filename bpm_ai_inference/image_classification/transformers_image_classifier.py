import logging

from PIL import Image
from bpm_ai_core.image_classification.image_classifier import ImageClassifier, ClassificationResult
from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.util.caching import cachable
from bpm_ai_core.util.image import pdf_to_images
from typing_extensions import override

try:
    from transformers import pipeline

    has_transformers = True
except ImportError:
    has_transformers = False

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/siglip-so400m-patch14-384"


@cachable()
class TransformersImageClassifier(ImageClassifier):
    """
    Local (zero-shot) image classification model based on Huggingface transformers library.

    To use, you should have the ``transformers`` and ``optimum`` python packages installed.
    """

    def __init__(self, model: str = DEFAULT_MODEL, zero_shot: bool = True):
        if not has_transformers:
            raise ImportError('transformers or optimum not installed')

        task = "zero-shot-image-classification" if zero_shot else "image-classification"

        self.image_classifier = pipeline(
            task,
            model=model
        )

    @override
    async def _do_classify(
            self,
            blob: Blob,
            classes: list[str] = None,
            hypothesis_template: str | None = None,
    ) -> ClassificationResult:
        if blob.is_pdf():
            images = pdf_to_images(await blob.as_bytes())
        elif blob.is_image():
            images = [Image.open(await blob.as_bytes_io())]
        else:
            raise ValueError("Blob must be a PDF or an image")

        if len(images) > 1:
            logger.warning('Multiple images provided, using only first image.')

        prediction = self.image_classifier(
            images[0],
            **({"candidate_labels": classes} if classes else {}),
            **({"hypothesis_template": hypothesis_template or "This is a photo of {}."} if classes else {})
        )
        # Zip the labels and scores together and find the label with the max score
        labels_scores = [(p['label'], p['score']) for p in prediction]
        total_score = sum(score for _, score in labels_scores)
        # Normalize the scores to 1
        labels_scores = [(label, score / total_score) for label, score in labels_scores]
        max_label, max_score = max(labels_scores, key=lambda x: x[1])

        return ClassificationResult(
            max_label=max_label,
            max_score=max_score,
            labels_scores=labels_scores
        )
