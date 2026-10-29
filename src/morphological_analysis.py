"""
Experiment 3: Morphological Analysis for Marathi Recipe Text
Performs morphological decomposition of Marathi words into root forms and grammatical affixes:
- Nominal inflections (विभक्ती प्रत्यय: -ात, -ाने, -ाला, -ाची, -ाचा, -ाचे)
- Verbal inflections and participles (कृदंत / क्रियाविशेषण: -ून, -ताना, -लेला, -ताना, -णे)
- Compound tokens and cooking postpositions.
"""
from typing import Dict, Any, List

class MarathiMorphologicalAnalyzer:
    def __init__(self):
        # Case markers / postpositions
        self.case_suffixes = {
            "ात": {"case": "सप्तमी (Locative / In)", "meaning": "मध्ये/आत (in/into)"},
            "ामध्ये": {"case": "सप्तमी (Locative / In)", "meaning": "मध्ये (inside)"},
            "ावर": {"case": "सप्तमी (Locative / On)", "meaning": "वर (on/upon)"},
            "ाने": {"case": "तृतीया (Instrumental / By)", "meaning": "द्वारे/ने (by/with)"},
            "ाला": {"case": "द्वितीया/चतुर्थी (Dative/Accusative)", "meaning": "कडे/ला (to/for)"},
            "ाची": {"case": "षष्ठी (Genitive / Of - F)", "meaning": "संबंध दर्शक (of)"},
            "ाचा": {"case": "षष्ठी (Genitive / Of - M)", "meaning": "संबंध दर्शक (of)"},
            "ाचे": {"case": "षष्ठी (Genitive / Of - N)", "meaning": "संबंध दर्शक (of)"},
            "ावरून": {"case": "पंचमी (Ablative / From above)", "meaning": "वरून (from above)"},
            "ातून": {"case": "पंचमी (Ablative / Out of)", "meaning": "मधून (out of)"}
        }
        
        # Verbal inflections and participles
        self.verb_suffixes = {
            "ून": {"type": "भूतकाळ धातूसाधित (Absolutive / Past participle)", "aspect": "क्रिया पूर्ण करून (having done action)"},
            "ताना": {"type": "वर्तमान धातूसाधित (Continuative participle)", "aspect": "क्रिया सुरू असताना (while doing)"},
            "लेला": {"type": "भूतकाळ विशेषण (Past adjectival participle - M)", "aspect": "झालेली क्रिया (completed action)"},
            "लेली": {"type": "भूतकाळ विशेषण (Past adjectival participle - F)", "aspect": "झालेली क्रिया (completed action)"},
            "लेले": {"type": "भूतकाळ विशेषण (Past adjectival participle - N)", "aspect": "झालेली क्रिया (completed action)"},
            "णार": {"type": "भविष्यकाळ धातूसाधित (Future participle)", "aspect": "घडणारी क्रिया (will happen)"},
            "वे": {"type": "विध्यर्थ रूप (Optative/Imperative)", "aspect": "करावयाची क्रिया (should do)"},
            "ा": {"type": "आज्ञार्थ रूप (Imperative / Instruction)", "aspect": "आज्ञा / पाककृती सूचना (instruction)"}
        }

    def analyze_token(self, token: str) -> Dict[str, Any]:
        """Analyzes a single Marathi token morphologically."""
        token_clean = token.strip()
        analysis = {
            "token": token_clean,
            "root_candidate": token_clean,
            "category": "Unknown / Uninflected",
            "affix": None,
            "description": "मूळ किंवा अविभक्त शब्द (Base or uninflected word)"
        }

        # Check verbal suffixes
        for suff, meta in sorted(self.verb_suffixes.items(), key=lambda x: len(x[0]), reverse=True):
            if token_clean.endswith(suff) and len(token_clean) > len(suff) + 1:
                analysis["root_candidate"] = token_clean[:-len(suff)]
                analysis["category"] = "Verbal Participle / Verb"
                analysis["affix"] = suff
                analysis["description"] = f"{meta['type']} - {meta['aspect']}"
                return analysis

        # Check nominal case suffixes
        for suff, meta in sorted(self.case_suffixes.items(), key=lambda x: len(x[0]), reverse=True):
            if token_clean.endswith(suff) and len(token_clean) > len(suff) + 1:
                analysis["root_candidate"] = token_clean[:-len(suff)]
                analysis["category"] = "Noun / Nominal Inflection"
                analysis["affix"] = suff
                analysis["description"] = f"{meta['case']} - अर्थ: {meta['meaning']}"
                return analysis

        return analysis

    def analyze_sentence(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """Analyzes a sequence of Marathi tokens."""
        return [self.analyze_token(t) for t in tokens]
