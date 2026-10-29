import pytest
from src.recipe_ner import MarathiRecipeNER
from src.chunking import MarathiChunker

def test_ner_entity_extraction():
    ner = MarathiRecipeNER()
    sent = "एका कढईत दोन चमचे तेल गरम करा."
    entities = ner.extract_entities(sent)
    labels = {e["label"] for e in entities}
    assert "UTENSIL" in labels
    assert "QUANTITY" in labels
    assert "UNIT" in labels
    assert "INGREDIENT" in labels
    assert "ACTION" in labels

def test_chunking_phrases():
    chunker = MarathiChunker()
    sent = "कढईत दोन चमचे तेल गरम करा."
    chunks = chunker.chunk_sentence(sent)
    labels = {c["label"] for c in chunks}
    assert "PP" in labels or "NP" in labels
    assert "VP" in labels

def test_annotated_html():
    ner = MarathiRecipeNER()
    html = ner.render_annotated_html("दोन चमचे तेल घाला.")
    assert "span" in html
    assert "QUANTITY" in html
    assert "INGREDIENT" in html
