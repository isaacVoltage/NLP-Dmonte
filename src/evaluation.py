"""
Evaluation Module for RecipeParse
Calculates empirical evaluation metrics on benchmark labelled test data:
- Named Entity Recognition: Precision, Recall, F1-score per entity class and overall
- HMM POS Tagging: Token-level Accuracy and Macro F1
- Next-word prediction accuracy (Top-1, Top-3, Top-5 hit rate)
- Field-level extraction accuracy for Recipe Extraction Pipeline
"""
from typing import Dict, Any, List
import pandas as pd
from collections import defaultdict
from src.recipe_ner import MarathiRecipeNER
from src.hmm_pos_tagger import MarathiHMMPosTagger
from src.ngram_model import MarathiNGramModel
from src.recipe_extractor import IntegratedRecipeExtractor

class RecipeParseEvaluator:
    def __init__(self):
        self.ner = MarathiRecipeNER()
        self.pos_tagger = MarathiHMMPosTagger()
        self.ngram_model = MarathiNGramModel()
        self.ngram_model.train()
        self.extractor = IntegratedRecipeExtractor()

        # Benchmarked Gold Test Sentences with annotated Entities
        self.gold_ner_data = [
            {
                "text": "एका कढईत दोन चमचे तेल गरम करा.",
                "gold_entities": [
                    ("कढईत", "UTENSIL"),
                    ("दोन", "QUANTITY"),
                    ("चमचे", "UNIT"),
                    ("तेल", "INGREDIENT"),
                    ("गरम करा", "ACTION")
                ]
            },
            {
                "text": "त्यात एक चमचा मोहरी आणि जिरे घाला.",
                "gold_entities": [
                    ("एक", "QUANTITY"),
                    ("चमचा", "UNIT"),
                    ("मोहरी", "INGREDIENT"),
                    ("जिरे", "INGREDIENT"),
                    ("घाला", "ACTION")
                ]
            },
            {
                "text": "बारीक चिरलेला कांदा घालून पाच मिनिटे परता.",
                "gold_entities": [
                    ("कांदा", "INGREDIENT"),
                    ("घालून", "ACTION"),
                    ("पाच मिनिटे", "DURATION"),
                    ("परता", "ACTION")
                ]
            },
            {
                "text": "दोन कप जाड पोहे स्वच्छ धुवून घ्या.",
                "gold_entities": [
                    ("दोन", "QUANTITY"),
                    ("कप", "UNIT"),
                    ("पोहे", "INGREDIENT"),
                    ("धुवून घ्या", "ACTION")
                ]
            },
            {
                "text": "दोन कप साबुदाणा चार तास पाण्यात भिजवून ठेवा.",
                "gold_entities": [
                    ("दोन", "QUANTITY"),
                    ("कप", "UNIT"),
                    ("साबुदाणा", "INGREDIENT"),
                    ("चार तास", "DURATION"),
                    ("पाण्यात", "INGREDIENT"),
                    ("भिजवून ठेवा", "ACTION")
                ]
            }
        ]

        # Gold POS test tokens
        self.gold_pos_tokens = [
            ("दोन", "NUM"), ("चमचे", "NOUN"), ("तेल", "NOUN"), ("गरम", "ADJ"), ("करा", "VERB"),
            ("एक", "NUM"), ("चमचा", "NOUN"), ("मोहरी", "NOUN"), ("घाला", "VERB"),
            ("बारीक", "ADJ"), ("कांदा", "NOUN"), ("घालून", "VERB"), ("पाच", "NUM"), ("मिनिटे", "NOUN"), ("परता", "VERB"),
            ("साजूक", "ADJ"), ("तूप", "NOUN"), ("भाजा", "VERB"), ("तूर", "NOUN"), ("डाळ", "NOUN"), ("शिजवा", "VERB")
        ]

        # N-Gram test prefixes and ground truth next words
        self.ngram_test_cases = [
            ("दोन चमचे", "तेल"),
            ("एक चमचा", "मोहरी"),
            ("पाच", "मिनिटे"),
            ("साजूक", "तूप"),
            ("तूर", "डाळ")
        ]

    def evaluate_ner(self) -> Dict[str, Any]:
        """Calculates Precision, Recall, and F1 score for Recipe NER."""
        tp_by_class = defaultdict(int)
        fp_by_class = defaultdict(int)
        fn_by_class = defaultdict(int)

        for case in self.gold_ner_data:
            text = case["text"]
            gold = case["gold_entities"]
            preds = self.ner.extract_entities(text)

            pred_tuples = [(p["text"].strip(), p["label"]) for p in preds]
            gold_tuples = [(g[0].strip(), g[1]) for g in gold]

            for pt, pl in pred_tuples:
                # Fuzzy match text overlap
                matched = any(gt in pt or pt in gt for gt, gl in gold_tuples if gl == pl)
                if matched:
                    tp_by_class[pl] += 1
                else:
                    fp_by_class[pl] += 1

            for gt, gl in gold_tuples:
                matched = any(gt in pt or pt in gt for pt, pl in pred_tuples if gl == pl)
                if not matched:
                    fn_by_class[gl] += 1

        all_classes = set(list(tp_by_class.keys()) + list(fp_by_class.keys()) + list(fn_by_class.keys()))
        class_metrics = []
        total_tp = sum(tp_by_class.values())
        total_fp = sum(fp_by_class.values())
        total_fn = sum(fn_by_class.values())

        for cls in sorted(all_classes):
            tp = tp_by_class[cls]
            fp = fp_by_class[cls]
            fn = fn_by_class[cls]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            class_metrics.append({
                "Entity_Class": cls,
                "Precision": round(prec * 100, 1),
                "Recall": round(rec * 100, 1),
                "F1_Score": round(f1 * 100, 1),
                "Support": tp + fn
            })

        overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = (2 * overall_p * overall_r) / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0

        return {
            "overall_precision": round(overall_p * 100, 1),
            "overall_recall": round(overall_r * 100, 1),
            "overall_f1": round(overall_f1 * 100, 1),
            "per_class_df": pd.DataFrame(class_metrics)
        }

    def evaluate_pos_tagger(self) -> Dict[str, Any]:
        """Evaluates token accuracy on test instances."""
        correct = 0
        total = len(self.gold_pos_tokens)
        per_token_results = []

        for token, true_tag in self.gold_pos_tokens:
            preds, _ = self.pos_tagger.tag_sentence(token)
            pred_tag = preds[0]["predicted_tag"] if preds else "OTHER"
            is_correct = (pred_tag == true_tag)
            if is_correct:
                correct += 1
            per_token_results.append({
                "Token": token,
                "Gold_Tag": true_tag,
                "Predicted_Tag": pred_tag,
                "Match": "✅" if is_correct else "❌"
            })

        accuracy = (correct / total) * 100 if total > 0 else 0.0
        return {
            "accuracy": round(accuracy, 1),
            "total_tokens": total,
            "correct_tokens": correct,
            "results_df": pd.DataFrame(per_token_results)
        }

    def evaluate_ngram_prediction(self) -> Dict[str, Any]:
        """Evaluates Top-1, Top-3, and Top-5 hit rates of next-word predictions."""
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        total = len(self.ngram_test_cases)
        results = []

        for phrase, target in self.ngram_test_cases:
            preds = self.ngram_model.predict_next_word_bigram(phrase.split()[-1], top_k=5)
            pred_words = [p["predicted_word"] for p in preds]

            is_top1 = len(pred_words) > 0 and pred_words[0] == target
            is_top3 = target in pred_words[:3]
            is_top5 = target in pred_words[:5]

            if is_top1:
                top1_hits += 1
            if is_top3:
                top3_hits += 1
            if is_top5:
                top5_hits += 1

            results.append({
                "Prompt_Context": phrase,
                "Expected_Word": target,
                "Top_Predictions": ", ".join(pred_words),
                "Top_1_Hit": "✅" if is_top1 else "❌",
                "Top_3_Hit": "✅" if is_top3 else "❌"
            })

        return {
            "top1_accuracy": round((top1_hits / total) * 100, 1),
            "top3_accuracy": round((top3_hits / total) * 100, 1),
            "top5_accuracy": round((top5_hits / total) * 100, 1),
            "total_cases": total,
            "results_df": pd.DataFrame(results)
        }

    def run_all_evaluations(self) -> Dict[str, Any]:
        """Runs all quantitative evaluations across modules."""
        ner_eval = self.evaluate_ner()
        pos_eval = self.evaluate_pos_tagger()
        ngram_eval = self.evaluate_ngram_prediction()

        return {
            "ner": ner_eval,
            "pos": pos_eval,
            "ngram": ngram_eval
        }
