import pytest
from src.hmm_pos_tagger import MarathiHMMPosTagger
from src.viterbi import ViterbiDecoder

def test_hmm_initialization_and_probabilities():
    tagger = MarathiHMMPosTagger()
    assert tagger.is_trained is True
    assert len(tagger.initial_probs) == len(tagger.tagset)
    assert len(tagger.transition_probs) == len(tagger.tagset)
    # Check probability sum approx 1
    assert abs(sum(tagger.initial_probs.values()) - 1.0) < 0.05

def test_viterbi_tagging_output():
    tagger = MarathiHMMPosTagger()
    sent = "दोन चमचे तेल गरम करा ."
    tags, trellis = tagger.tag_sentence(sent)
    assert len(tags) > 0
    tokens = [t["token"] for t in tags]
    predicted = [t["predicted_tag"] for t in tags]
    assert "दोन" in tokens
    assert "करा" in tokens
    assert len(trellis) == len(tags)

def test_unknown_token_fallback():
    tagger = MarathiHMMPosTagger()
    prob_num = tagger.get_unknown_emission("999", "NUM")
    prob_noun = tagger.get_unknown_emission("999", "NOUN")
    assert prob_num > prob_noun
