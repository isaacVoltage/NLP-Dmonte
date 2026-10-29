"""
Experiment 4: N-Gram Language Model for Marathi Recipe Corpus
Implements:
- Corpus loading and tokenization
- Unigram, Bigram, Trigram frequency distributions
- Maximum Likelihood Estimation (MLE) and Add-k (Laplace) smoothed probabilities
- Next-word prediction using Bigram context (1 preceding word)
- Next-word prediction using Trigram context (2 preceding words)
- Comparison of predictions across N-gram orders with fallback mechanism
"""
import os
import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any, Optional
from src.utils import load_recipes, BASE_DIR

class MarathiNGramModel:
    def __init__(self, k_smoothing: float = 0.5):
        self.k = k_smoothing
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.vocab = set()
        self.total_tokens = 0
        self.is_trained = False

    def load_corpus_from_recipes(self, recipes: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Loads Marathi recipe texts and concatenates them into training corpus."""
        if recipes is None:
            recipes = load_recipes()
        texts = [r.get("text", "") for r in recipes if r.get("text")]
        return texts

    def tokenize_text(self, text: str) -> List[str]:
        """Tokenizes text preserving Marathi Devanagari characters and numbers."""
        import re
        tokens = re.findall(r'[\u0900-\u097F0-9a-zA-Z]+', text)
        return [t.strip() for t in tokens if t.strip()]

    def train(self, texts: Optional[List[str]] = None):
        """Trains unigram, bigram, and trigram frequency counts on the corpus."""
        if texts is None:
            texts = self.load_corpus_from_recipes()

        self.unigrams.clear()
        self.bigrams.clear()
        self.trigrams.clear()
        self.vocab.clear()
        self.total_tokens = 0

        for text in texts:
            tokens = self.tokenize_text(text)
            if not tokens:
                continue

            # Add start/end sentence markers for N-gram stability
            padded_tokens = ["<s>", "<s>"] + tokens + ["</s>"]
            self.total_tokens += len(tokens)

            for i in range(len(padded_tokens)):
                w = padded_tokens[i]
                if w not in ("<s>", "</s>"):
                    self.unigrams[w] += 1
                    self.vocab.add(w)

                if i >= 1:
                    bg = (padded_tokens[i-1], padded_tokens[i])
                    self.bigrams[bg] += 1

                if i >= 2:
                    tg = (padded_tokens[i-2], padded_tokens[i-1], padded_tokens[i])
                    self.trigrams[tg] += 1

        self.is_trained = True

    def get_unigram_frequency(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """Returns top N most frequent unigram words."""
        return self.unigrams.most_common(top_n)

    def get_bigram_frequency(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """Returns top N most frequent bigrams formatted as string."""
        return [(f"{w1} {w2}", count) for (w1, w2), count in self.bigrams.most_common(top_n) if w1 != "<s>" and w2 != "</s>"]

    def get_trigram_frequency(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """Returns top N most frequent trigrams formatted as string."""
        return [(f"{w1} {w2} {w3}", count) for (w1, w2, w3), count in self.trigrams.most_common(top_n) if "<s>" not in (w1, w2) and w3 != "</s>"]

    def predict_next_word_bigram(self, previous_word: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Predicts next word using bigram model: P(w | previous_word).
        Uses Add-k smoothing over vocabulary.
        """
        if not self.is_trained:
            self.train()

        candidates = []
        prev_count = self.unigrams.get(previous_word, 0)
        vocab_size = len(self.vocab)

        # Find words following previous_word
        matching_next_words = {w2: cnt for (w1, w2), cnt in self.bigrams.items() if w1 == previous_word and w2 != "</s>"}

        if matching_next_words:
            for w2, count in matching_next_words.items():
                # Add-k smoothed probability
                prob = (count + self.k) / (prev_count + self.k * vocab_size) if (prev_count + self.k * vocab_size) > 0 else 0
                candidates.append({
                    "predicted_word": w2,
                    "frequency": count,
                    "probability": round(prob, 4),
                    "model": "Bigram (P(w|w_{i-1}))"
                })
        else:
            # Fallback to unigram top frequencies with smoothing
            for w, count in self.unigrams.most_common(top_k):
                prob = (count + self.k) / (self.total_tokens + self.k * vocab_size)
                candidates.append({
                    "predicted_word": w,
                    "frequency": count,
                    "probability": round(prob, 4),
                    "model": "Unigram Fallback (Unseen Context)"
                })

        candidates.sort(key=lambda x: x["probability"], reverse=True)
        return candidates[:top_k]

    def predict_next_word_trigram(self, word1: str, word2: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Predicts next word using trigram model: P(w | word1, word2).
        If unseen, falls back to bigram model.
        """
        if not self.is_trained:
            self.train()

        candidates = []
        bg_count = self.bigrams.get((word1, word2), 0)
        vocab_size = len(self.vocab)

        matching_next_words = {w3: cnt for (w_a, w_b, w3), cnt in self.trigrams.items() if w_a == word1 and w_b == word2 and w3 != "</s>"}

        if matching_next_words:
            for w3, count in matching_next_words.items():
                prob = (count + self.k) / (bg_count + self.k * vocab_size) if (bg_count + self.k * vocab_size) > 0 else 0
                candidates.append({
                    "predicted_word": w3,
                    "frequency": count,
                    "probability": round(prob, 4),
                    "model": "Trigram (P(w|w_{i-2}, w_{i-1}))"
                })
        else:
            # Fallback to bigram using word2
            bg_predictions = self.predict_next_word_bigram(word2, top_k=top_k)
            for item in bg_predictions:
                item_copy = dict(item)
                item_copy["model"] = "Bigram Backoff (Trigram Unseen)"
                candidates.append(item_copy)

        candidates.sort(key=lambda x: x["probability"], reverse=True)
        return candidates[:top_k]

    def compare_predictions(self, input_phrase: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Compares next-word predictions between Bigram and Trigram models for input text.
        """
        tokens = self.tokenize_text(input_phrase)
        if not tokens:
            return {
                "error": "कृपया किमान एक वैध मराठी शब्द प्रविष्ट करा (Please enter at least one word)."
            }

        if len(tokens) == 1:
            bigram_preds = self.predict_next_word_bigram(tokens[-1], top_k)
            trigram_preds = [
                {
                    "predicted_word": "कमी शब्द",
                    "frequency": 0,
                    "probability": 0.0,
                    "model": "Trigram requires at least 2 context words"
                }
            ]
        else:
            bigram_preds = self.predict_next_word_bigram(tokens[-1], top_k)
            trigram_preds = self.predict_next_word_trigram(tokens[-2], tokens[-1], top_k)

        return {
            "input_phrase": input_phrase,
            "tokens": tokens,
            "bigram_predictions": bigram_preds,
            "trigram_predictions": trigram_preds,
            "total_vocab_size": len(self.vocab),
            "total_corpus_tokens": self.total_tokens,
            "academic_disclaimer": "ही भाकिते (predictions) उपलब्ध पाककृती संग्रहावर आधारित संख्याशास्त्रीय संभाव्यता दर्शवतात. ती प्रत्यक्षात पाककृती म्हणून स्वयंपाकासाठी तपासलेली किंवा खात्रीशीर सूचना नाहीत."
        }
