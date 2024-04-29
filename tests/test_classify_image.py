from bpm_ai_inference.image_classification.transformers_image_classifier import TransformersImageClassifier


async def test_classify_zero_shot():
    image = "files/example.jpg"
    classes = ["golden retriever", "border collie", "labrador", "cat", "human"]
    expected = "labrador"

    classifier = TransformersImageClassifier()
    result = await classifier.classify(image, classes, confidence_threshold=0.8)

    assert result.max_label == expected


async def test_classify():
    image = "files/example.jpg"
    expected = "normal"

    classifier = TransformersImageClassifier(model="Falconsai/nsfw_image_detection", zero_shot=False)
    result = await classifier.classify(image, confidence_threshold=0.8)

    assert result.max_label == expected


async def test_classify_threshold():
    image = "files/example.jpg"
    classes = ["cat", "mouse", "human", "house"]
    expected = None

    classifier = TransformersImageClassifier()
    result = await classifier.classify(image, classes, confidence_threshold=0.8)

    assert result == expected

