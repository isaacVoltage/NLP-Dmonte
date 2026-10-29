"""
Experiment 5A: Marathi Chunking Module
Implements rule-based chunking tailored for Marathi recipe syntax:
- NP (Noun Phrase): Modifier + Quantity + Unit + Head Noun (उदा. [एक चमचा मोहरी], [चिरलेला कांदा])
- VP (Verb Phrase): Verbal participle + Auxiliary / Imperative Verb (उदा. [गरम करा], [शिजवून घ्या], [तळून घ्या])
- PP (Postpositional Phrase): Nominal noun with locative/instrumental postposition (उदा. [पातेल्यात], [मध्यम आचेवर], [तेलामध्ये])
"""
import re
from typing import List, Dict, Any, Tuple
from src.utils import load_ingredients, load_cooking_actions

class MarathiChunker:
    def __init__(self):
        self.ingredients = {i["marathi"] for i in load_ingredients()}
        self.postposition_suffixes = ["ात", "ामध्ये", "ावर", "ाने", "ावरून", "ातून", "ाकडे", "ासाठी"]
        self.postposition_words = ["मध्ये", "वर", "खाली", "आत", "बाहेर", "समोरील", "शेजारी", "सोबत"]
        
        self.action_stems = {
            "करा", "घाला", "परता", "तळा", "भाजा", "शिजवा", "उकळा", "मिसळा", "वाटा", "किसा", "मळा", "लाटा",
            "वाढा", "द्या", "घ्या", "ठेवा", "काढा", "ओता", "टाका", "सर्व्ह करा"
        }
        self.auxiliary_verbs = {"घ्या", "द्या", "करा", "ठेवा", "काढा", "पहा", "बघा", "पाहिजे"}

    def chunk_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        """
        Parses sentence tokens and groups them into syntactic chunks:
        NP (Noun Phrase), VP (Verb Phrase), PP (Postpositional Phrase), O (Other).
        """
        raw_tokens = re.findall(r'[\u0900-\u097F0-9a-zA-Z]+|[.,!?;]', sentence)
        chunks = []
        i = 0
        n = len(raw_tokens)

        while i < n:
            token = raw_tokens[i]

            # 1. Check for Postpositional Phrase (PP)
            # Token ending in locative suffix OR followed by postposition word
            is_pp = False
            for suff in self.postposition_suffixes:
                if token.endswith(suff) and len(token) > len(suff) + 1:
                    chunks.append({
                        "phrase": token,
                        "label": "PP",
                        "description": "Postpositional Phrase (सप्तमी / अव्यय रूप)",
                        "tokens": [token]
                    })
                    is_pp = True
                    break
            if is_pp:
                i += 1
                continue

            if i + 1 < n and raw_tokens[i+1] in self.postposition_words:
                pp_phrase = f"{token} {raw_tokens[i+1]}"
                chunks.append({
                    "phrase": pp_phrase,
                    "label": "PP",
                    "description": "Postpositional Phrase (शब्दयोगी अव्यय जोडलेले पद)",
                    "tokens": [token, raw_tokens[i+1]]
                })
                i += 2
                continue

            # 2. Check for Verb Phrase (VP)
            # Participle + Auxiliary (e.g. गरम करा, शिजवून घ्या, तळून घ्या)
            if i + 1 < n and (token.endswith("ून") or token.endswith("त") or token in ["गरम", "स्वच्छ", "बारीक", "सर्व्ह"]):
                if raw_tokens[i+1] in self.auxiliary_verbs or raw_tokens[i+1] in self.action_stems:
                    vp_phrase = f"{token} {raw_tokens[i+1]}"
                    chunks.append({
                        "phrase": vp_phrase,
                        "label": "VP",
                        "description": "Verb Phrase (संयुक्त किंवा सहाय्यकारी क्रियापद समूह)",
                        "tokens": [token, raw_tokens[i+1]]
                    })
                    i += 2
                    continue

            # Single action imperative
            if token in self.action_stems:
                chunks.append({
                    "phrase": token,
                    "label": "VP",
                    "description": "Verb Phrase (क्रियापद - कृती निर्देश)",
                    "tokens": [token]
                })
                i += 1
                continue

            # 3. Check for Noun Phrase (NP)
            # (Adjective / Quantity / Unit) + Noun
            np_tokens = [token]
            j = i + 1
            while j < n:
                next_tok = raw_tokens[j]
                # Break if next token is punctuation or starts VP or PP
                if next_tok in [".", ",", "!", "?"] or next_tok in self.action_stems or any(next_tok.endswith(s) for s in self.postposition_suffixes):
                    break
                if next_tok in self.postposition_words:
                    break
                # If next token is noun-like or unit or ingredient
                np_tokens.append(next_tok)
                j += 1
                if len(np_tokens) >= 4:  # reasonable NP span limit
                    break

            if len(np_tokens) > 1:
                np_phrase = " ".join(np_tokens)
                chunks.append({
                    "phrase": np_phrase,
                    "label": "NP",
                    "description": "Noun Phrase (नाम व विशेषण/परिमाण समूह)",
                    "tokens": np_tokens
                })
                i = j
                continue
            else:
                # Fallback token
                chunks.append({
                    "phrase": token,
                    "label": "NP" if token in self.ingredients or len(token) > 2 else "O",
                    "description": "Noun (नाम)" if token in self.ingredients else "Other Token",
                    "tokens": [token]
                })
                i += 1

        return chunks
