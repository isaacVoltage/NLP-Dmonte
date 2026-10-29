"""
Experiment 3: Regex Extractor Module for Marathi Recipe Text
Extracts:
A. Quantities (Devanagari/Arabic numbers, fractions, number words)
B. Measurement Units
C. Cooking Durations
D. Temperatures and Heat Levels
E. Cooking Actions
F. Recipe Structure Patterns
G. Ingredient Names (Dictionary-assisted + Contextual Regex)
"""
import re
from typing import List, Dict, Any, Optional
from src.utils import load_ingredients, load_cooking_actions

DEV_BOUNDARY_START = r"(?<![\u0900-\u097F0-9a-zA-Z])"
DEV_BOUNDARY_END = r"(?![\u0900-\u097F0-9a-zA-Z])"

class MarathiRegexExtractor:
    def __init__(self):
        # Ingredient dictionary
        self.ingredients_data = load_ingredients()
        # Cooking actions dictionary
        self.actions_data = load_cooking_actions()

        # Build ingredient lookup patterns
        self.ingredient_terms = []
        for item in self.ingredients_data:
            self.ingredient_terms.append(item["marathi"])
            for var in item.get("variants", []):
                self.ingredient_terms.append(var)
        self.ingredient_terms = sorted(list(set(self.ingredient_terms)), key=len, reverse=True)

        # Word numbers in Marathi
        self.marathi_num_words = (
            r"(?:एक|दोन|तीन|चार|पाच|सहा|सात|आठ|नऊ|दहा|अकरा|बारा|तेरा|चौदा|पंधरा|"
            r"वीस|पंचवीस|तीस|चाळीस|पन्नास|शंभर|अर्धा|अर्धी|अर्धे|पाव|पावणे|सव्वा|दीड|अडीच|"
            r"दोन-तीन|तीन-चार|एक-दोन)"
        )
        
        # Digits (Devanagari, Arabic, decimal, and fractions)
        self.digit_pattern = r"(?:\d+(?:\.\d+)?|[०-९]+(?:\.[०-९]+)?|\d+/\d+|[०-९]+/[०-९]+)"
        
        # Combined quantity pattern
        self.quantity_regex = rf"(?:{self.digit_pattern}|{self.marathi_num_words})"

        # Measurement Units in Marathi & English
        self.units_list = [
            "चमचा", "चमचे", "चमचेभर", "टीस्पून", "टेबलस्पून",
            "कप", "वाट्या", "वाटी", "भांडे",
            "ग्रॅम", "किलो", "किलोग्रॅम", "लिटर", "मिलीलीटर", "मिली",
            "चिमूट", "चिमूटभर", "तुकडा", "तुकडे", "नग", "दाणे", "पाने", "पान", "पाकळ्या", "काप",
            "cup", "cups", "tsp", "tbsp", "gm", "kg", "ml", "liter"
        ]
        self.unit_regex = r"(?:" + "|".join(re.escape(u) for u in sorted(self.units_list, key=len, reverse=True)) + r")"

        # Cooking Durations (e.g. पाच मिनिटे, १० मिनिटे, अर्धा तास, दोन तास)
        self.duration_regex = re.compile(
            rf"{DEV_BOUNDARY_START}({self.quantity_regex})\s*(?:ते\s*{self.quantity_regex}\s*)?(मिनिटे|मिनिटात|मिनिटांसाठी|तास|तासात|तासांसाठी|सेकंद|min|mins|minutes|hours){DEV_BOUNDARY_END}",
            re.IGNORECASE
        )

        # Cooking Temperatures & Heat levels
        self.temperature_regex = re.compile(
            rf"{DEV_BOUNDARY_START}(मंद\s*आच(?:ेवर)?|मध्यम\s*आच(?:ेवर)?|मोठ्या\s*आच(?:ेवर)?|कडकडीत\s*गरम|हलके\s*गरम|कोमट|"
            rf"(?:\d+|[०-९]+)\s*(?:अंश\s*सेल्सिअस|अंश|°C|डिग्री)|रूम\s*टेंपरेचर){DEV_BOUNDARY_END}",
            re.IGNORECASE
        )

        # Cooking Actions
        self.action_terms = []
        for a in self.actions_data:
            self.action_terms.extend(a.get("forms", []))
        self.action_terms.extend([
            "घाला", "घालून", "मिसळा", "मिसळून", "शिजवा", "शिजवून", "उकळा", "उकळून",
            "परता", "परतून", "कापा", "कापून", "भाजा", "भाजून", "तळा", "तळून",
            "वाटा", "वाटून", "किसा", "किसून", "मळा", "मळून", "लाटा", "लाटून",
            "वाफवा", "वाफवून", "वाफ काढा", "तडतडू द्या", "गरम करा", "सर्व्ह करा", "वाढा"
        ])
        self.action_terms = sorted(list(set(self.action_terms)), key=len, reverse=True)
        self.action_regex = re.compile(
            rf"{DEV_BOUNDARY_START}(" + "|".join(re.escape(a) for a in self.action_terms) + rf"){DEV_BOUNDARY_END}"
        )

        # Cooking Utensils / Tools
        self.utensil_terms = [
            "कढई", "कढईत", "पातेले", "पातेल्यात", "तवा", "तव्यावर", "कुकर", "कुकरमध्ये",
            "मिक्सर", "मिक्सरमध्ये", "चाळणी", "चाळणीत", "पळी", "चमचा", "झारा", "विळी", "सुरी", "बाऊल", "भांडे", "भांड्यात"
        ]
        self.utensil_regex = re.compile(
            rf"{DEV_BOUNDARY_START}(" + "|".join(re.escape(u) for u in self.utensil_terms) + rf"){DEV_BOUNDARY_END}"
        )

    def extract_quantities_and_units(self, sentence: str) -> List[Dict[str, Any]]:
        """Extracts quantity + unit pairs like 'दोन चमचे', '५० ग्रॅम', 'एक कप'."""
        pattern = re.compile(rf"{DEV_BOUNDARY_START}({self.quantity_regex})\s*({self.unit_regex}){DEV_BOUNDARY_END}", re.IGNORECASE)
        results = []
        for match in pattern.finditer(sentence):
            results.append({
                "original_text": match.group(0),
                "type": "QUANTITY_AND_UNIT",
                "quantity": match.group(1),
                "unit": match.group(2),
                "start": match.start(),
                "end": match.end(),
                "sentence": sentence
            })
        return results

    def extract_durations(self, sentence: str) -> List[Dict[str, Any]]:
        """Extracts cooking durations like 'पाच मिनिटे', 'अर्धा तास'."""
        results = []
        for match in self.duration_regex.finditer(sentence):
            results.append({
                "original_text": match.group(0),
                "type": "DURATION",
                "value": match.group(0).strip(),
                "quantity": match.group(1),
                "unit": match.group(2),
                "start": match.start(),
                "end": match.end(),
                "sentence": sentence
            })
        return results

    def extract_temperatures(self, sentence: str) -> List[Dict[str, Any]]:
        """Extracts temperatures and heat levels like 'मंद आच', 'मध्यम आचेवर', '180°C'."""
        results = []
        for match in self.temperature_regex.finditer(sentence):
            results.append({
                "original_text": match.group(0),
                "type": "TEMPERATURE",
                "value": match.group(0).strip(),
                "start": match.start(),
                "end": match.end(),
                "sentence": sentence
            })
        return results

    def extract_actions(self, sentence: str) -> List[Dict[str, Any]]:
        """Extracts cooking verbs and action instructions."""
        results = []
        for match in self.action_regex.finditer(sentence):
            results.append({
                "original_text": match.group(0),
                "type": "ACTION",
                "value": match.group(0).strip(),
                "start": match.start(),
                "end": match.end(),
                "sentence": sentence
            })
        return results

    def extract_utensils(self, sentence: str) -> List[Dict[str, Any]]:
        """Extracts cooking tools and vessels."""
        results = []
        for match in self.utensil_regex.finditer(sentence):
            results.append({
                "original_text": match.group(0),
                "type": "UTENSIL",
                "value": match.group(0).strip(),
                "start": match.start(),
                "end": match.end(),
                "sentence": sentence
            })
        return results

    def extract_ingredients(self, sentence: str) -> List[Dict[str, Any]]:
        """Extracts ingredient mentions in the sentence using configurable dictionary."""
        results = []
        matched_spans = []

        for term in self.ingredient_terms:
            pattern = re.compile(rf"{DEV_BOUNDARY_START}{re.escape(term)}{DEV_BOUNDARY_END}", re.IGNORECASE)
            for m in pattern.finditer(sentence):
                # Avoid overlapping spans
                if any(start <= m.start() < end or start < m.end() <= end for start, end in matched_spans):
                    continue
                matched_spans.append((m.start(), m.end()))
                
                # Check for preceding quantity and unit
                prefix = sentence[:m.start()].strip()
                q_match = re.search(rf"({self.quantity_regex})\s*({self.unit_regex})?$", prefix)
                qty = q_match.group(1) if q_match else "Not specified"
                unit = q_match.group(2) if (q_match and q_match.group(2)) else "Not specified"

                results.append({
                    "original_text": m.group(0),
                    "type": "INGREDIENT",
                    "value": term,
                    "quantity": qty,
                    "unit": unit,
                    "start": m.start(),
                    "end": m.end(),
                    "sentence": sentence
                })
        return results

    def extract_recipe_structure(self, text: str) -> List[Dict[str, Any]]:
        """Identifies recipe step boundaries, sequential markers, and numbered lists."""
        step_pattern = re.compile(r"(?:^|\n)(?:\d+\.|\d+\)|\([०-९\d]+\)|प्रथम|त्यानंतर|नंतर|आता|शेवटी|सर्वप्रथम)\s*([^.\n]+[.!?]?)", re.MULTILINE)
        steps = []
        for idx, match in enumerate(step_pattern.finditer(text), start=1):
            steps.append({
                "step_number": idx,
                "text": match.group(0).strip(),
                "content": match.group(1).strip()
            })
        return steps

    def extract_all(self, text: str) -> Dict[str, Any]:
        """Runs all regular expression extraction models over input Marathi text."""
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if len(s.strip()) > 2]
        
        all_quantities = []
        all_durations = []
        all_temperatures = []
        all_actions = []
        all_utensils = []
        all_ingredients = []

        for sent in sentences:
            all_quantities.extend(self.extract_quantities_and_units(sent))
            all_durations.extend(self.extract_durations(sent))
            all_temperatures.extend(self.extract_temperatures(sent))
            all_actions.extend(self.extract_actions(sent))
            all_utensils.extend(self.extract_utensils(sent))
            all_ingredients.extend(self.extract_ingredients(sent))

        structure_steps = self.extract_recipe_structure(text)

        # Build combined tabular items for display
        combined_records = []
        for ing in all_ingredients:
            sent = ing["sentence"]
            sent_actions = [a["value"] for a in all_actions if a["sentence"] == sent]
            combined_records.append({
                "Ingredient": ing["value"],
                "Quantity": ing["quantity"],
                "Unit": ing["unit"],
                "Action": "/".join(sent_actions) if sent_actions else "Not specified",
                "Step / Sentence": sent
            })

        return {
            "quantities_and_units": all_quantities,
            "durations": all_durations,
            "temperatures": all_temperatures,
            "actions": all_actions,
            "utensils": all_utensils,
            "ingredients": all_ingredients,
            "structure_steps": structure_steps,
            "combined_table": combined_records,
            "total_matches": (
                len(all_quantities) + len(all_durations) + len(all_temperatures) +
                len(all_actions) + len(all_utensils) + len(all_ingredients)
            ),
            "disclaimer": "Regular expressions provide pattern recognition over known constructs but do not substitute full syntactic or semantic parsing."
        }
