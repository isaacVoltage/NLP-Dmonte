"""
Integrated Recipe Information Extraction Pipeline
Combines Experiments 2 through 8 into one unified recipe analysis system:
- Preprocessing and script validation
- Regex & Morphological extraction
- Recipe Named Entity Recognition (NER)
- Chunking (NP, VP, PP)
- HMM / Viterbi POS tagging
- Structured tabular synthesis with export options
"""
import re
from typing import Dict, Any, List
from src.preprocessing import MarathiPreprocessor
from src.regex_extractor import MarathiRegexExtractor
from src.recipe_ner import MarathiRecipeNER
from src.chunking import MarathiChunker
from src.hmm_pos_tagger import MarathiHMMPosTagger

class IntegratedRecipeExtractor:
    def __init__(self):
        self.preprocessor = MarathiPreprocessor()
        self.regex_extractor = MarathiRegexExtractor()
        self.ner = MarathiRecipeNER()
        self.chunker = MarathiChunker()
        self.pos_tagger = MarathiHMMPosTagger()

    def process_recipe(self, recipe_text: str) -> Dict[str, Any]:
        """
        Executes end-to-end extraction workflow on Marathi recipe text.
        """
        # Step 1: Input validation
        val = self.preprocessor.validate_input(recipe_text)
        if not val["valid"]:
            return {"success": False, "error": val["message"]}

        script_val = self.preprocessor.validate_script(recipe_text)

        # Step 2: Preprocessing
        clean_text = self.preprocessor.normalize_unicode(recipe_text)
        clean_text = self.preprocessor.normalize_punctuation(clean_text)
        steps = self.preprocessor.normalize_recipe_steps(clean_text)
        raw_tokens = self.preprocessor.word_tokenize(clean_text)
        filtered_tokens, removed_stopwords = self.preprocessor.remove_stopwords(raw_tokens)

        # Step 3: Regex Pattern Extraction
        regex_results = self.regex_extractor.extract_all(clean_text)

        # Step 4: Named Entity Recognition
        entities = self.ner.extract_entities(clean_text)
        highlighted_html = self.ner.render_annotated_html(clean_text)

        # Step 5: Syntactic Chunking
        chunk_results = []
        for step in steps:
            chunks = self.chunker.chunk_sentence(step)
            chunk_results.append({
                "step_text": step,
                "chunks": chunks
            })

        # Step 6: HMM POS Tagging (sample first 2 steps or full sentence)
        pos_results, trellis = self.pos_tagger.tag_sentence(clean_text)

        # Step 7: Structured Tabular Synthesis
        # Maps ingredients to quantity, unit, actions, and step
        structured_items = []
        all_ings = regex_results.get("ingredients", [])
        all_actions = regex_results.get("actions", [])
        all_durations = regex_results.get("durations", [])
        all_temps = regex_results.get("temperatures", [])
        all_utensils = regex_results.get("utensils", [])

        # Deduplicate ingredients by term
        seen_ings = set()
        for ing in all_ings:
            val_term = ing["value"]
            if val_term in seen_ings:
                continue
            seen_ings.add(val_term)

            sent = ing["sentence"]
            # Associated action in same sentence or step
            step_actions = [a["value"] for a in all_actions if a["sentence"] == sent]
            step_durations = [d["value"] for d in all_durations if d["sentence"] == sent]
            step_utensils = [u["value"] for u in all_utensils if u["sentence"] == sent]

            structured_items.append({
                "Ingredient": val_term,
                "Quantity": ing.get("quantity", "Not specified"),
                "Unit": ing.get("unit", "Not specified"),
                "Action": " / ".join(step_actions) if step_actions else "Not specified",
                "Duration": " / ".join(step_durations) if step_durations else "Not specified",
                "Utensil": " / ".join(step_utensils) if step_utensils else "Not specified",
                "Context": sent
            })

        return {
            "success": True,
            "original_text": recipe_text,
            "script_validation": script_val,
            "normalized_steps": steps,
            "total_steps": len(steps),
            "tokens_count": len(raw_tokens),
            "filtered_tokens_count": len(filtered_tokens),
            "stopwords_removed_count": len(removed_stopwords),
            "extracted_ingredients_count": len(seen_ings),
            "extracted_actions_count": len(all_actions),
            "extracted_durations": [d["value"] for d in all_durations],
            "extracted_temperatures": [t["value"] for t in all_temps],
            "extracted_utensils": list(set(u["value"] for u in all_utensils)),
            "structured_table": structured_items,
            "entities": entities,
            "highlighted_html": highlighted_html,
            "chunks_by_step": chunk_results,
            "pos_tags": pos_results
        }
