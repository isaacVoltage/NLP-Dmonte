"""
Experiment 6: Hidden Markov Model (HMM) Part-of-Speech Tagger for Marathi
Trained on data/pos_tagged_corpus.csv.
Tagset: NOUN, VERB, ADJ, ADV, NUM, PRONOUN, POSTPOSITION, PARTICLE, OTHER.
Implements:
- Initial-state probability distribution Pi
- State transition matrix A
- Emission probability matrix B with smoothing
- Morphological heuristic for unknown Marathi words
- Viterbi decoding
"""
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
from src.utils import load_pos_tagged_corpus
from src.viterbi import ViterbiDecoder

class MarathiHMMPosTagger:
    def __init__(self, smoothing_alpha: float = 0.01):
        self.alpha = smoothing_alpha
        self.tagset = ["NOUN", "VERB", "ADJ", "ADV", "NUM", "PRONOUN", "POSTPOSITION", "PARTICLE", "OTHER"]
        self.initial_counts = Counter()
        self.transition_counts = defaultdict(Counter)
        self.emission_counts = defaultdict(Counter)
        self.tag_counts = Counter()
        self.vocab = set()

        self.initial_probs = {}
        self.transition_probs = {}
        self.emission_probs = {}
        self.is_trained = False

        # Train on default annotated dataset
        self.train()

    def train(self, df: Optional[pd.DataFrame] = None):
        """Trains HMM parameters on annotated dataset."""
        if df is None:
            df = load_pos_tagged_corpus()

        if df.empty:
            return

        self.initial_counts.clear()
        self.transition_counts.clear()
        self.emission_counts.clear()
        self.tag_counts.clear()
        self.vocab.clear()

        # Group by sentence
        sentences = df.groupby('sentence_id')
        num_sentences = len(sentences)

        for _, sent in sentences:
            tokens = sent['token'].tolist()
            tags = sent['tag'].tolist()

            if not tags:
                continue

            # Initial state
            first_tag = tags[0]
            self.initial_counts[first_tag] += 1

            for i in range(len(tags)):
                curr_tag = tags[i]
                curr_tok = str(tokens[i]).strip()

                self.tag_counts[curr_tag] += 1
                self.emission_counts[curr_tag][curr_tok] += 1
                self.vocab.add(curr_tok)

                if i > 0:
                    prev_tag = tags[i-1]
                    self.transition_counts[prev_tag][curr_tag] += 1

        # 1. Compute Initial State Probabilities Pi(s) with Laplace smoothing
        total_inits = sum(self.initial_counts.values())
        num_tags = len(self.tagset)
        for t in self.tagset:
            self.initial_probs[t] = (self.initial_counts[t] + self.alpha) / (total_inits + self.alpha * num_tags)

        # 2. Compute Transition Probabilities A(s_i -> s_j) with Add-alpha smoothing
        for t_prev in self.tagset:
            self.transition_probs[t_prev] = {}
            prev_total = sum(self.transition_counts[t_prev].values())
            for t_curr in self.tagset:
                count = self.transition_counts[t_prev][t_curr]
                self.transition_probs[t_prev][t_curr] = (count + self.alpha) / (prev_total + self.alpha * num_tags)

        # 3. Compute Emission Probabilities B(s -> w) with Add-alpha smoothing
        vocab_size = len(self.vocab)
        for t in self.tagset:
            self.emission_probs[t] = {}
            tag_total = self.tag_counts[t]
            for w in self.vocab:
                w_count = self.emission_counts[t][w]
                if w_count > 0:
                    self.emission_probs[t][w] = (w_count + self.alpha) / (tag_total + self.alpha * vocab_size)

        self.is_trained = True

    def get_unknown_emission(self, token: str, tag: str) -> float:
        """
        Morphological fallback probability for out-of-vocabulary (OOV) Marathi words.
        """
        # Numerals
        if re.match(r'^[\d०-९]+(?:[./][\d०-९]+)?$', token):
            return 0.85 if tag == "NUM" else 0.01

        # Verbs commonly end in -ून, -ताना, -णे, -वे, -ला, -ली, -ले, -वा
        if any(token.endswith(s) for s in ["ून", "ताना", "णे", "वावे", "तात"]):
            return 0.70 if tag == "VERB" else 0.03

        # Nouns with locative or instrumental case suffixes
        if any(token.endswith(s) for s in ["ात", "ामध्ये", "ावर", "ाने", "ाला", "ाचे", "ाची"]):
            return 0.70 if tag == "NOUN" else 0.04

        # Adverbs ending in -णे, -पणे, -सर
        if any(token.endswith(s) for s in ["पणे", "सर", "वार"]):
            return 0.60 if tag == "ADV" else 0.04

        # Punctuation
        if token in [".", ",", "!", "?", ";", ":", "-", '"', "'"]:
            return 0.95 if tag == "OTHER" else 0.001

        # Prior distribution over tags as fallback
        tag_prior = self.tag_counts.get(tag, 1) / max(sum(self.tag_counts.values()), 1)
        return max(tag_prior * 0.1, 1e-5)

    def tag_sentence(self, sentence: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Tags input Marathi sentence using HMM and Viterbi algorithm.
        Returns:
            - List of tag records {token, tag, confidence, explanation}
            - Trellis steps matrix for visual demonstration
        """
        if not self.is_trained:
            self.train()

        tokens = re.findall(r'[\u0900-\u097F0-9a-zA-Z]+|[.,!?;]', sentence)
        if not tokens:
            return [], []

        # Prepare emissions with OOV estimation
        custom_emissions = {}
        for s in self.tagset:
            custom_emissions[s] = dict(self.emission_probs.get(s, {}))
            for tok in tokens:
                if tok not in custom_emissions[s]:
                    custom_emissions[s][tok] = self.get_unknown_emission(tok, s)

        decoder = ViterbiDecoder(
            states=self.tagset,
            initial_probs=self.initial_probs,
            transition_probs=self.transition_probs,
            emission_probs=custom_emissions
        )

        viterbi_results, trellis_steps = decoder.decode(tokens)

        tag_explanations = {
            "NOUN": "नाम (Noun: घटक, पात्र, स्थळ किंवा वस्तू)",
            "VERB": "क्रियापद (Verb: स्वयंपाक कृती किंवा निर्देश)",
            "ADJ": "विशेषण (Adjective: गुण, रंग किंवा स्थिती)",
            "ADV": "क्रियाविशेषण (Adverb: कृतीची रीत किंवा काल)",
            "NUM": "संख्याविशेषण / परिमाण (Numeral / Quantity)",
            "PRONOUN": "सर्वनाम (Pronoun)",
            "POSTPOSITION": "शब्दयोगी अव्यय (Postposition)",
            "PARTICLE": "केवलप्रयोगी / अव्यय (Particle)",
            "OTHER": "विरामचिन्ह किंवा इतर (Punctuation / Other)"
        }

        output = []
        for tok, tag, conf in viterbi_results:
            output.append({
                "token": tok,
                "predicted_tag": tag,
                "confidence": conf,
                "explanation": tag_explanations.get(tag, "अज्ञात प्रवर्ग")
            })

        return output, trellis_steps
