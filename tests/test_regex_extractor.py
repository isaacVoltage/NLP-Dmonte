import pytest
from src.regex_extractor import MarathiRegexExtractor

def test_extract_quantities_and_units():
    ext = MarathiRegexExtractor()
    sent = "कढईत दोन चमचे तेल आणि ५० ग्रॅम रवा घाला."
    qu = ext.extract_quantities_and_units(sent)
    quantities = [item["quantity"] for item in qu]
    units = [item["unit"] for item in qu]
    assert "दोन" in quantities or "2" in quantities
    assert "चमचे" in units
    assert "५०" in quantities or "50" in quantities
    assert "ग्रॅम" in units

def test_extract_durations():
    ext = MarathiRegexExtractor()
    sent = "बारीक कांदा घालून पाच मिनिटे परता आणि नंतर अर्धा तास ठेवा."
    durations = ext.extract_durations(sent)
    extracted = [d["value"] for d in durations]
    assert any("पाच मिनिटे" in e for e in extracted)
    assert any("अर्धा तास" in e for e in extracted)

def test_extract_temperatures():
    ext = MarathiRegexExtractor()
    sent = "मंद आचेवर दहा मिनिटे रवा भाजा आणि मध्यम आचेवर ठेवा."
    temps = ext.extract_temperatures(sent)
    vals = [t["value"] for t in temps]
    assert any("मंद आच" in v for v in vals)
    assert any("मध्यम आच" in v for v in vals)

def test_extract_cooking_actions():
    ext = MarathiRegexExtractor()
    sent = "तेलात मोहरी घाला आणि कांदा परता."
    actions = ext.extract_actions(sent)
    vals = [a["value"] for a in actions]
    assert "घाला" in vals
    assert "परता" in vals

def test_extract_ingredients():
    ext = MarathiRegexExtractor()
    sent = "एका पातेल्यात दोन चमचे तेल गरम करा. त्यात एक चमचा मोहरी घाला."
    ings = ext.extract_ingredients(sent)
    terms = [i["value"] for i in ings]
    assert "तेल" in terms
    assert "मोहरी" in terms
