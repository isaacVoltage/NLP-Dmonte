"""
Experiment 7: Word Sense Disambiguation (WSD) using the Simplified Lesk Algorithm for Marathi
Genuine implementation of the Lesk Overlap Algorithm:
1. Extract context words from input sentence.
2. Normalize and remove functional stopwords.
3. For each candidate sense of the ambiguous Marathi target word:
   - Tokenize the sense definition/gloss and signature examples.
   - Remove stopwords.
   - Compute vocabulary intersection (word overlap).
4. Score = |Context_Words ∩ Sense_Definition_Words|.
5. Return winning sense with full step-by-step overlap details.
"""
import re
from typing import List, Dict, Any, Optional, Set
from src.utils import load_wsd_lexicon, load_stopwords

class MarathiLeskWSD:
    def __init__(self):
        self.lexicon = load_wsd_lexicon()
        self.stopwords = set(load_stopwords())
        self.target_words_dict = {
            item["word"]: item for item in self.lexicon.get("target_words", [])
        }

    def get_supported_target_words(self) -> List[str]:
        """Returns the list of polysemous Marathi target words in the lexicon."""
        return list(self.target_words_dict.keys())

    def preprocess_text(self, text: str) -> Set[str]:
        """Tokenizes text and removes Marathi stopwords, returning a set of content words."""
        tokens = re.findall(r'[\u0900-\u097F0-9a-zA-Z]+', text)
        content_words = {
            t.strip() for t in tokens
            if t.strip() and t.strip() not in self.stopwords and len(t.strip()) > 1
        }
        return content_words

    def disambiguate(self, sentence: str, target_word: str) -> Dict[str, Any]:
        """
        Runs the genuine Lesk algorithm to disambiguate target_word in the sentence context.
        """
        if target_word not in self.target_words_dict:
            return {
                "error": f"'{target_word}' हा शब्द प्रात्यक्षिक लेक्सिकॉनमध्ये उपलब्ध नाही. (Word not in demonstration lexicon)."
            }

        word_entry = self.target_words_dict[target_word]
        candidate_senses = word_entry.get("senses", [])

        # Step 1: Preprocess sentence context (excluding the target word itself)
        context_words = self.preprocess_text(sentence)
        context_words_without_target = {w for w in context_words if w != target_word}

        sense_scores = []
        highest_score = -1
        best_sense = None

        for sense in candidate_senses:
            sense_id = sense["sense_id"]
            gloss = sense["gloss"]
            def_words = set(sense.get("definition_words", []))
            examples = sense.get("examples", [])

            # Combine gloss and examples tokens into signature set
            signature_words = set(def_words)
            signature_words.update(self.preprocess_text(gloss))
            for ex in examples:
                signature_words.update(self.preprocess_text(ex))

            # Exclude target word from signature
            signature_words.discard(target_word)

            # Compute actual word overlap
            overlap_words = sorted(list(context_words_without_target.intersection(signature_words)))
            overlap_count = len(overlap_words)

            score_record = {
                "sense_id": sense_id,
                "gloss": gloss,
                "overlap_count": overlap_count,
                "overlapping_words": overlap_words,
                "signature_size": len(signature_words)
            }
            sense_scores.append(score_record)

            if overlap_count > highest_score:
                highest_score = overlap_count
                best_sense = score_record

        # Handle tie or zero overlap gracefully
        is_tie = sum(1 for s in sense_scores if s["overlap_count"] == highest_score) > 1 and highest_score > 0
        if highest_score <= 0:
            # Default to first sense (most frequent / default sense) with explanation
            best_sense = sense_scores[0]
            resolution_note = "संदर्भात कोणताही थेट शब्द जुळला नाही (Zero overlap). पूर्वनिर्धारित मुख्य अर्थ (Sense 1) निवडला गेला आहे."
        elif is_tie:
            resolution_note = "दोन अर्थांमध्ये समान ओव्हरलॅप आढळला (Tie in overlap count). प्रथम प्राधान्य असलेला अर्थ निवडला गेला."
        else:
            resolution_note = f"कमाल शब्द ओव्हरलॅप ({highest_score} शब्द: {', '.join(best_sense['overlapping_words'])}) आधारे निवडला गेला."

        return {
            "sentence": sentence,
            "target_word": target_word,
            "context_content_words": sorted(list(context_words_without_target)),
            "candidate_senses": sense_scores,
            "selected_sense_id": best_sense["sense_id"],
            "selected_gloss": best_sense["gloss"],
            "best_overlap_count": highest_score,
            "winning_overlapping_words": best_sense["overlapping_words"],
            "explanation": resolution_note,
            "lexicon_provenance": "Manually Curated Marathi Demonstration Lexicon (indo-WordNet adaptation for college practicals)",
            "algorithm_type": "Simplified Lesk Overlap Algorithm"
        }
