import pytest
from src.recipe_extractor import IntegratedRecipeExtractor
from src.evaluation import RecipeParseEvaluator

def test_integrated_recipe_pipeline():
    extractor = IntegratedRecipeExtractor()
    sample_recipe = (
        "एका पातेल्यात दोन चमचे तेल गरम करा. त्यात एक चमचा मोहरी घाला. "
        "त्यानंतर चिरलेला कांदा घालून पाच मिनिटे परता."
    )
    result = extractor.process_recipe(sample_recipe)
    assert result["success"] is True
    assert result["extracted_ingredients_count"] >= 2
    assert len(result["structured_table"]) >= 2
    assert "highlighted_html" in result
    assert len(result["pos_tags"]) > 0

    # Verify structured fields
    ingredients = [row["Ingredient"] for row in result["structured_table"]]
    assert "तेल" in ingredients
    assert "मोहरी" in ingredients or "कांदा" in ingredients

def test_evaluator_metrics():
    evaluator = RecipeParseEvaluator()
    results = evaluator.run_all_evaluations()
    assert "ner" in results
    assert "pos" in results
    assert "ngram" in results
    assert results["ner"]["overall_f1"] > 50.0
    assert results["pos"]["accuracy"] > 50.0
