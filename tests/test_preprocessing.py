import pytest
from src.preprocessing import MarathiPreprocessor

def test_marathi_tokenization():
    p = MarathiPreprocessor()
    text = "दोन चमचे तेल गरम करा."
    tokens = p.word_tokenize(text)
    assert "दोन" in tokens
    assert "चमचे" in tokens
    assert "तेल" in tokens
    assert "गरम" in tokens
    assert "करा" in tokens

def test_script_validation():
    p = MarathiPreprocessor()
    dev_text = "एका कढईत दोन चमचे तेल गरम करा."
    val_dev = p.validate_script(dev_text)
    assert val_dev["is_devanagari"] is True
    assert val_dev["percentage"] > 70.0

    eng_text = "Heat two spoons of oil in a pan."
    val_eng = p.validate_script(eng_text)
    assert val_eng["is_devanagari"] is False

def test_numeral_normalization():
    p = MarathiPreprocessor()
    dev_num = "५० ग्रॅम आणि ०.५ लिटर"
    arabic = p.normalize_numerals(dev_num, to_arabic=True)
    assert "50" in arabic
    assert "0.5" in arabic

    back_to_dev = p.normalize_numerals(arabic, to_arabic=False)
    assert "५०" in back_to_dev
    assert "०.५" in back_to_dev

def test_stopword_removal():
    p = MarathiPreprocessor()
    tokens = ["त्यात", "दोन", "चमचे", "तेल", "आणि", "मोहरी", "घाला"]
    filtered, removed = p.remove_stopwords(tokens)
    assert "त्यात" in removed
    assert "आणि" in removed
    assert "तेल" in filtered
    assert "मोहरी" in filtered
    assert "दोन" in filtered

def test_marathi_stemmer_and_lemmatizer():
    p = MarathiPreprocessor()
    assert p.stem_word("तेलात") == "तेल"
    assert p.lemmatize_word("कांद्याला") == "कांदा"
    assert p.lemmatize_word("चिरलेला") == "चिरणे"
