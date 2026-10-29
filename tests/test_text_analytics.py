import pytest
from src.text_analytics import MarathiTextAnalytics

def test_recipe_statistics():
    analytics = MarathiTextAnalytics()
    stats = analytics.get_recipe_statistics()
    assert stats["total_recipes"] >= 20
    assert stats["total_tokens"] > 500
    assert stats["vocabulary_size"] > 100

def test_frequencies():
    analytics = MarathiTextAnalytics()
    ing_df = analytics.get_ingredient_frequencies(5)
    assert not ing_df.empty
    assert "Ingredient" in ing_df.columns

    action_df = analytics.get_cooking_action_frequencies(5)
    assert not action_df.empty
    assert "Cooking_Action" in action_df.columns

def test_missing_data_summary():
    analytics = MarathiTextAnalytics()
    missing_df = analytics.get_missing_data_summary()
    assert not missing_df.empty
    assert "Field" in missing_df.columns
