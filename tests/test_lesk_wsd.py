import pytest
from src.lesk_wsd import MarathiLeskWSD

def test_lesk_supported_words():
    wsd = MarathiLeskWSD()
    words = wsd.get_supported_target_words()
    assert "तेल" in words
    assert "तूप" in words
    assert "पान" in words
    assert "बारीक" in words

def test_lesk_cooking_disambiguation():
    wsd = MarathiLeskWSD()
    sent = "कढईत दोन चमचे तेल गरम करून मोहरीची फोडणी द्या."
    res = wsd.disambiguate(sent, "तेल")
    assert res["selected_sense_id"] == "तेल_01"
    assert res["best_overlap_count"] >= 1

def test_lesk_non_cooking_disambiguation():
    wsd = MarathiLeskWSD()
    sent = "देवापुढे समईत तेल ओतून दिवा प्रज्वलित करा."
    res = wsd.disambiguate(sent, "तेल")
    assert res["selected_sense_id"] == "तेल_02"
    assert res["best_overlap_count"] >= 1
