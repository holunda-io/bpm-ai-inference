from bpm_ai_inference.pos.spacy_pos_tagger import SpacyPOSTagger


async def test_pos():
    text = "It's me, John Meier."

    tagger = SpacyPOSTagger()
    result = await tagger.tag(text)

    assert result.tags == [
        ('It', 'PRON'), ("'s ", 'AUX'), ('me', 'PRON'), (', ', 'PUNCT'), ('John ', 'PROPN'), ('Meier', 'PROPN'), ('.', 'PUNCT')
    ]
