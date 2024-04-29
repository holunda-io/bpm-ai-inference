from bpm_ai_inference.classification.transformers_text_classifier import TransformersClassifier


async def test_classify_zero_shot():
    text = "I am so sleepy today."
    classes = ["tired", "energized", "unknown"]
    expected = "tired"

    classifier = TransformersClassifier()
    result = await classifier.classify(text, classes, confidence_threshold=0.8)

    assert result.max_label == expected


async def test_classify():
    text = "I am so happy today."
    expected = "Positive"

    classifier = TransformersClassifier(model="RashidNLP/Finance-Sentiment-Classification", zero_shot=False)
    result = await classifier.classify(text, confidence_threshold=0.8)

    assert result.max_label == expected


async def test_classify_threshold():
    text = "I am ok."
    classes = ["tired", "energized"]
    expected = None

    classifier = TransformersClassifier()
    result = await classifier.classify(text, classes, confidence_threshold=0.9)

    assert result == expected


async def test_classify_multi():
    text = "Poetry is a tool for dependency management and packaging in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Poetry offers a lockfile to ensure repeatable installs, and can build your project for distribution."
    classes = ["python", "news", "programming", "javascript", "law", "history"]
    expected = {"python", "programming"}

    classifier = TransformersClassifier()
    result = await classifier.classify(text, classes, confidence_threshold=0.8, multi_label=True)

    assert set([l for l, s in result.labels_scores]) == expected