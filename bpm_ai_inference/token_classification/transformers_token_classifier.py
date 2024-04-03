import logging

from bpm_ai_core.token_classification.zero_shot_token_classifier import ZeroShotTokenClassifier, \
    TokenClassificationResult, TokenSpan
from typing_extensions import override

from bpm_ai_inference.classification.transformers_classifier import TransformersClassifier, DEFAULT_MODEL_EN, \
    DEFAULT_MODEL_MULTI
from bpm_ai_inference.pos.spacy_pos_tagger import SpacyPOSTagger
from bpm_ai_inference.util.language import indentify_language

logger = logging.getLogger(__name__)


class TransformersTokenClassifier(ZeroShotTokenClassifier):
    """
    Zero Shot Token Classifier based on a POS tagger and a zero shot classifier.

    To use, you should have the ``spacy`` and ``transformers`` python packages installed.
    """

    def __init__(self, classifier_model: str = None):
        self.classifier_model = classifier_model

    @override
    async def _do_classify(
            self,
            text: str,
            classes: list[str],
            confidence_threshold: float | None = None
    ) -> TokenClassificationResult:
        language = indentify_language(text)
        tagger = SpacyPOSTagger(language=language)
        classifier = TransformersClassifier(
            model=self.classifier_model or (DEFAULT_MODEL_EN if language == "en" else DEFAULT_MODEL_MULTI)
        )

        pos_tags = (await tagger.tag(text)).tags
        candidates = filter_and_join(pos_tags)

        entities = []
        for candidate in candidates:
            results = []
            for c in classes:
                true_label = c.lower()
                false_label = f"not {true_label}"
                result = await classifier.classify(candidate, [true_label, false_label], confidence_threshold=0.75)
                if result and result.max_label == true_label:
                    results.append((candidate, result.max_label, result.max_score))
            best_result = max(results, key=lambda r: r[2], default=None)
            if best_result:
                entities.append(best_result)

        return TokenClassificationResult(
            spans=[
                TokenSpan(
                    word=e[0],
                    label=e[1],
                    score=e[2],
                    start=-1,  # todo extract span indices and multiple occurrences
                    end=-1
                )
                for e in entities
            ]
        )


DEFAULT_TAGS_TO_JOIN = ['NOUN', 'PROPN', 'NUM', 'SYM', 'X']


def filter_and_join(tags, tags_to_join=None):
    if tags_to_join is None:
        tags_to_join = DEFAULT_TAGS_TO_JOIN
    result = []
    current_word = ""
    prev_tag = None

    for token, tag in tags:
        if tag in tags_to_join:
            if prev_tag not in tags_to_join and current_word:
                result.append(current_word.strip())
                current_word = ""
            current_word += token
        else:
            if current_word:
                result.append(current_word.strip())
                current_word = ""
        prev_tag = tag

    if current_word:
        result.append(current_word.strip())
    return result
