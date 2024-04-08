from bpm_ai_inference.token_classification.gliner_token_classifier import GlinerTokenClassifier
from bpm_ai_inference.token_classification.transformers_token_classifier import TransformersTokenClassifier


async def test_classify():
    text = "My name is John and I am 20 years old."
    classes = ["firstname", "age"]
    expected_firstname = "John"
    expected_age = "20 years"

    classifier = TransformersTokenClassifier()
    result = await classifier.classify(text, classes)

    print(result)
    assert result.spans[0].word == expected_firstname
    assert result.spans[1].word == expected_age


async def test_classify_threshold():
    text = "My name is John and I am 20 years old."
    classes = ["occupation"]
    expected = []

    classifier = TransformersTokenClassifier()
    result = await classifier.classify(text, classes, confidence_threshold=0.8)

    assert result.spans == expected


async def test_gliner():
    text = """
    We received the following orders: Pizza (10.99€), Steak (28.89€). Please deliver asap.
    """
    classes = ["meal order", "price"]
    expected_meal = ["Pizza", "Steak"]
    expected_price = ["10.99€", "28.89€"]

    classifier = GlinerTokenClassifier(model="urchade/gliner_small")
    result = await classifier.classify(text, classes)

    meal_spans = [m.word for m in result.spans if m.label == "meal order"]
    price_spans = [m.word for m in result.spans if m.label == "price"]

    assert meal_spans == expected_meal
    assert price_spans == expected_price