import pytest
from src.ngram_model import MarathiNGramModel

def test_ngram_training_and_counts():
    model = MarathiNGramModel()
    sample_texts = [
        "दोन चमचे तेल गरम करा .",
        "त्यात एक चमचा मोहरी घाला .",
        "दोन कप पोहे स्वच्छ धुवा ."
    ]
    model.train(sample_texts)
    assert model.is_trained is True
    assert model.total_tokens > 0
    assert len(model.vocab) > 5

    unigrams = dict(model.get_unigram_frequency(10))
    assert "दोन" in unigrams

def test_next_word_prediction():
    model = MarathiNGramModel()
    model.train()
    # Predict after 'दोन'
    preds = model.predict_next_word_bigram("दोन", top_k=3)
    assert len(preds) > 0
    words = [p["predicted_word"] for p in preds]
    assert any(w in ["चमचे", "कप", "तास", "मिनिटे", "लिटर"] for w in words)

def test_compare_predictions():
    model = MarathiNGramModel()
    model.train()
    res = model.compare_predictions("दोन चमचे", top_k=3)
    assert "bigram_predictions" in res
    assert "trigram_predictions" in res
    assert len(res["bigram_predictions"]) > 0
