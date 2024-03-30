from bpm_ai_inference.classification.transformers_classifier import TransformersClassifier


async def test_classify():
    text = "I am so sleepy today."
    classes = ["tired", "energized", "unknown"]
    expected = "tired"

    classifier = TransformersClassifier()
    result = await classifier.classify(text, classes, confidence_threshold=0.8)

    assert result.max_label == expected


async def test_classify_threshold():
    text = "I am ok."
    classes = ["tired", "energized"]
    expected = None

    classifier = TransformersClassifier()
    result = await classifier.classify(text, classes, confidence_threshold=0.9)

    assert result == expected
