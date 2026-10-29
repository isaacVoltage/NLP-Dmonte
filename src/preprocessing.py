"""
Experiment 2: Text Preprocessing Module for Marathi Recipe Text
Includes:
- Tokenization (Word & Sentence)
- Rule-based Marathi Stemmer
- Domain-adapted Marathi Lemmatizer
- Stopword Removal (preserving culinary negations and quantities)
- Filtration & Noise Removal
- Devanagari Script Validation
- Additional techniques: Unicode normalization (NFC/NFD), whitespace normalization,
  punctuation normalization, numeral normalization, recipe step normalization,
  repeated punctuation handling, and input validation.
"""
import re
import unicodedata
from typing import List, Dict, Tuple, Any
from src.utils import (
    load_stopwords,
    devanagari_to_arabic_num,
    arabic_to_devanagari_num,
    is_devanagari_char
)

class MarathiPreprocessor:
    def __init__(self, stopwords: List[str] = None):
        self.stopwords = set(stopwords) if stopwords else set(load_stopwords())
        # Suffix rules for rule-based Marathi stemmer (longest suffixes first)
        self.marathi_suffixes = [
            "ांसारखा", "ांसारखी", "ांसारखे", "ांच्या", "ांसाठी", "ांसमोर", "ांमधील",
            "ाप्रमाणे", "ांनुसार", "ावरून", "ातून", "ाला", "ांना", "ांवर", "ामध्ये",
            "ावर", "ातील", "ात", "ाचे", "ाची", "ाचा", "ांचे", "ांची", "ांचा",
            "्याने", "्यांनी", "्याला", "्यांना", "्याचे", "्याची", "्याचा", "्यांच्या",
            "ून", "त", "ता", "ताना", "लेला", "लेली", "लेले", "लेल्या", "णाऱ्या",
            "णार", "णारा", "णारी", "णारे", "वे", "वा", "वी", "तात", "तो", "ते"
        ]
        # Lexical lemmatization mapping for common Marathi recipe words
        self.lemma_dict = {
            "कांद्याला": "कांदा", "कांद्याची": "कांदा", "कांद्यात": "कांदा", "कांदे": "कांदा", "कांद्याने": "कांदा",
            "तेलात": "तेल", "तेलामध्ये": "तेल", "तेलाचा": "तेल", "तेलाची": "तेल", "तले": "तेल",
            "तुपात": "तूप", "तुपामध्ये": "तूप", "तुपाचा": "तूप", "तुपाची": "तूप",
            "पाण्यात": "पाणी", "पाण्याने": "पाणी", "पाण्याची": "पाणी",
            "बटाट्याची": "बटाटा", "बटाट्यात": "बटाटा", "बटाटे": "बटाटा", "बटाट्याला": "बटाटा",
            "वांग्याची": "वांगी", "वांग्यात": "वांगी", "वांगे": "वांगी",
            "मिरच्या": "हिरवी मिरची", "मिरच्यांची": "हिरवी मिरची", "मिरचीत": "हिरवी मिरची",
            "पोह्यांचे": "पोहे", "पोह्यांना": "पोहे", "पोह्यात": "पोहे",
            "रव्यात": "रवा", "रव्याचा": "रवा", "रव्याची": "रवा",
            "साखरेत": "साखर", "साखरेचा": "साखर", "साखरेची": "साखर",
            "गुळात": "गूळ", "गुळाचा": "गूळ", "गुळाची": "गूळ",
            "दुधात": "दूध", "दुधाचा": "दूध", "दुधामध्ये": "दूध",
            "दह्यात": "दही", "दह्याचा": "दही", "दह्यामध्ये": "दही",
            "भातात": "भात", "भाताचा": "भात", "भातावर": "भात",
            "कढईत": "कढई", "कढईमध्ये": "कढई", "कढईला": "कढई",
            "पातेल्यात": "पातेले", "पातेल्यामध्ये": "पातेले",
            "कुकरमध्ये": "कुकर", "कुकरच्या": "कुकर",
            "चिरलेला": "चिरणे", "चिरलेली": "चिरणे", "चिरलेले": "चिरणे", "चिरून": "चिरणे", "चिरा": "चिरणे",
            "भाजलेला": "भाजणे", "भाजलेली": "भाजणे", "भाजलेले": "भाजणे", "भाजून": "भाजणे", "भाजा": "भाजणे",
            "शिजलेला": "शिजवणे", "शिजलेली": "शिजवणे", "शिजलेले": "शिजवणे", "शिजवून": "शिजवणे", "शिजवा": "शिजवणे",
            "तळलेला": "तळणे", "तळलेली": "तळणे", "तळलेले": "तळणे", "तळून": "तळणे", "तळा": "तळणे",
            "उकळलेला": "उकळणे", "उकळलेली": "उकळणे", "उकळलेले": "उकळणे", "उकळून": "उकळणे", "उकळा": "उकळणे",
            "घाला": "घालणे", "घालून": "घालणे", "घालतात": "घालणे", "घालत": "घालणे",
            "परता": "परतणे", "परतून": "परतणे", "परतत": "परतणे",
            "मिसळा": "मिसळणे", "मिसळून": "मिसळणे", "मिसळत": "मिसळणे",
            "वाटा": "वाटणे", "वाटून": "वाटणे", "वाटलेले": "वाटणे",
            "किसा": "किसणे", "किसून": "किसणे", "किसलेला": "किसणे", "किसलेली": "किसणे", "किसलेले": "किसणे"
        }
        # Critical cooking negation tokens to preserve
        self.critical_negations = {"नाही", "नका", "नको", "नये", "नव्हते", "नाहीत", "नसल्यास", "नसता"}

    def validate_input(self, text: str) -> Dict[str, Any]:
        """Validates if input is non-empty and reasonably sized."""
        if not text or not text.strip():
            return {"valid": False, "message": "इनपुट रिकामे आहे (Input is empty)."}
        if len(text.strip()) < 3:
            return {"valid": False, "message": "मजकूर खूप लहान आहे (Text is too short)."}
        return {"valid": True, "message": "वैध मजकूर (Valid input).", "char_count": len(text)}

    def validate_script(self, text: str) -> Dict[str, Any]:
        """
        Validates whether the text contains Devanagari script.
        Calculates Devanagari character percentage.
        """
        if not text:
            return {"is_devanagari": False, "percentage": 0.0, "details": "खालील मजकूर उपलब्ध नाही."}
        
        non_space_chars = [ch for ch in text if not ch.isspace()]
        if not non_space_chars:
            return {"is_devanagari": False, "percentage": 0.0, "details": "केवळ रिक्त जागा आढळली."}
        
        dev_count = sum(1 for ch in non_space_chars if is_devanagari_char(ch))
        percentage = (dev_count / len(non_space_chars)) * 100.0
        is_devanagari = percentage >= 40.0
        
        return {
            "is_devanagari": is_devanagari,
            "devanagari_chars": dev_count,
            "total_chars": len(non_space_chars),
            "percentage": round(percentage, 2),
            "status": "मान्य देवनागरी लिपी (Valid Devanagari)" if is_devanagari else "अमान्य किंवा कमी देवनागरी वर्ण (Low Devanagari ratio)"
        }

    def normalize_unicode(self, text: str, form: str = "NFC") -> str:
        """Applies Unicode normalization (NFC or NFD) for consistent character encoding."""
        return unicodedata.normalize(form, text)

    def remove_extra_whitespace(self, text: str) -> str:
        """Removes duplicate spaces, leading/trailing whitespace, and blank lines."""
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()

    def normalize_punctuation(self, text: str) -> str:
        """Normalizes traditional Marathi danda '।' to standard period and cleans quotes."""
        text = text.replace("।", ".")
        text = text.replace("॥", ".")
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("–", "-").replace("—", "-")
        return text

    def handle_repeated_punctuation(self, text: str) -> str:
        """Reduces repeated punctuation marks like '??', '!!', '..' to a single mark."""
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r',{2,}', ',', text)
        return text

    def normalize_numerals(self, text: str, to_arabic: bool = True) -> str:
        """Converts Devanagari numerals to Arabic digits or vice versa."""
        if to_arabic:
            return devanagari_to_arabic_num(text)
        return arabic_to_devanagari_num(text)

    def normalize_recipe_steps(self, text: str) -> List[str]:
        """Segments text into sequential recipe steps based on punctuation and numbering."""
        text = self.normalize_punctuation(text)
        # Split by periods, semicolons, newlines, or step markers
        raw_steps = re.split(r'[\n\r]+|[.!?]\s+|\d+\.\s*', text)
        cleaned_steps = [s.strip() for s in raw_steps if s and len(s.strip()) > 3]
        return cleaned_steps

    def sentence_tokenize(self, text: str) -> List[str]:
        """Segments Marathi text into sentences."""
        norm_text = self.normalize_punctuation(text)
        sentences = re.split(r'(?<=[.!?])\s+', norm_text)
        return [s.strip() for s in sentences if s.strip()]

    def word_tokenize(self, text: str, preserve_punctuation: bool = False) -> List[str]:
        """
        Tokenizes Marathi text into word tokens.
        Preserves Devanagari conjuncts, halant, and matras accurately.
        """
        text = self.normalize_unicode(text, "NFC")
        if preserve_punctuation:
            tokens = re.findall(r'[\u0900-\u097F0-9a-zA-Z]+|[.,!?;:()"-]', text)
        else:
            tokens = re.findall(r'[\u0900-\u097F0-9a-zA-Z]+', text)
        return [t.strip() for t in tokens if t.strip()]

    def remove_stopwords(self, tokens: List[str], preserve_negations: bool = True) -> Tuple[List[str], List[str]]:
        """
        Removes stopwords from token list while strictly preserving
        cooking ingredients, units, quantities, and important negations.
        Returns (filtered_tokens, removed_stopwords).
        """
        filtered = []
        removed = []
        for t in tokens:
            t_lower = t.strip()
            if preserve_negations and t_lower in self.critical_negations:
                filtered.append(t)
            elif t_lower in self.stopwords:
                removed.append(t)
            else:
                filtered.append(t)
        return filtered, removed

    def stem_word(self, word: str) -> str:
        """
        Rule-based Marathi suffix-stripping stemmer.
        Strips inflectional case suffixes (-ात, -ाचे, -ाला, -ून, etc.).
        Note: Marathi is a fusional language; suffix stripping has known linguistic limitations,
        which are documented for academic clarity.
        """
        if len(word) <= 3:
            return word
        for suffix in self.marathi_suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if len(stem) >= 2:
                    return stem
        return word

    def lemmatize_word(self, word: str) -> str:
        """
        Dictionary-assisted Marathi lemmatizer with rule-based fallback.
        Maps inflected forms to standard base lemma.
        """
        # Exact dictionary lookup
        if word in self.lemma_dict:
            return self.lemma_dict[word]
        # Check normalized form
        clean_word = self.normalize_unicode(word, "NFC")
        if clean_word in self.lemma_dict:
            return self.lemma_dict[clean_word]
        # Fallback to rule-based stemmer
        return self.stem_word(clean_word)

    def filter_tokens(self, tokens: List[str], remove_numbers: bool = False) -> List[str]:
        """Filters out non-Devanagari/punctuation noise from token list."""
        filtered = []
        for t in tokens:
            # Check if token contains Devanagari characters or digits
            has_dev = any(is_devanagari_char(c) for c in t)
            has_digit = any(c.isdigit() or c in "०१२३४५६७८९" for c in t)
            if remove_numbers and has_digit and not has_dev:
                continue
            if has_dev or has_digit:
                filtered.append(t)
        return filtered

    def run_full_pipeline(self, text: str) -> Dict[str, Any]:
        """
        Runs the full text preprocessing pipeline for Experiment 2,
        returning structured outputs for comparison and analysis.
        """
        validation = self.validate_input(text)
        if not validation["valid"]:
            return {"error": validation["message"]}

        script_val = self.validate_script(text)
        nfc_text = self.normalize_unicode(text, "NFC")
        no_extra_ws = self.remove_extra_whitespace(nfc_text)
        punct_norm = self.normalize_punctuation(no_extra_ws)
        clean_punct = self.handle_repeated_punctuation(punct_norm)
        arabic_num_text = self.normalize_numerals(clean_punct, to_arabic=True)
        dev_num_text = self.normalize_numerals(clean_punct, to_arabic=False)
        recipe_steps = self.normalize_recipe_steps(clean_punct)
        sentences = self.sentence_tokenize(clean_punct)
        raw_tokens = self.word_tokenize(clean_punct)
        
        filtered_tokens, removed_stopwords = self.remove_stopwords(raw_tokens)
        cleaned_tokens = self.filter_tokens(filtered_tokens)
        
        stemmed_tokens = [(t, self.stem_word(t)) for t in cleaned_tokens]
        lemmatized_tokens = [(t, self.lemmatize_word(t)) for t in cleaned_tokens]

        return {
            "original_text": text,
            "script_validation": script_val,
            "unicode_normalized": nfc_text,
            "whitespace_cleaned": no_extra_ws,
            "punctuation_normalized": punct_norm,
            "repeated_punctuation_cleaned": clean_punct,
            "arabic_numerals_normalized": arabic_num_text,
            "devanagari_numerals_normalized": dev_num_text,
            "recipe_steps": recipe_steps,
            "sentences": sentences,
            "total_sentences": len(sentences),
            "raw_tokens": raw_tokens,
            "total_raw_tokens": len(raw_tokens),
            "stopwords_removed": removed_stopwords,
            "total_stopwords_removed": len(removed_stopwords),
            "filtered_tokens": cleaned_tokens,
            "total_filtered_tokens": len(cleaned_tokens),
            "stemmed_tokens": stemmed_tokens,
            "lemmatized_tokens": lemmatized_tokens,
            "limitation_note": "मराठी ही विभक्तीप्रधान (fusional/agglutinative) भाषा असल्याने इंग्रजी पोर्टर स्टेमर इथे चालत नाही. येथे नियम-आधारित प्रत्यय-निष्कासन (suffix-stripping) आणि शब्दकोश-आधारित लेमॅटायझर वापरला आहे."
        }
