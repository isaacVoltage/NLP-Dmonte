"""
Experiment 8: Exploratory Data Analysis (EDA) and Word Cloud Dashboard
Performs comprehensive data analytics on Marathi recipe corpus:
- Devanagari Word Cloud generation using system font
- Word frequency analysis (Words, Ingredients, Cooking Actions, Units, Durations)
- Recipe corpus statistics and distributions
- Ingredient co-occurrence matrix and heatmap
- Category comparison and descriptive statistics
- Missing-data analysis
- Correlation analysis
"""
import io
import re
import os
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from src.utils import (
    load_recipes_df,
    load_recipes,
    load_stopwords,
    load_ingredients,
    load_cooking_actions,
    get_devanagari_font_path
)

class MarathiTextAnalytics:
    def __init__(self):
        self.recipes_df = load_recipes_df()
        self.recipes_list = load_recipes()
        self.stopwords = set(load_stopwords())
        self.ingredients_meta = load_ingredients()
        self.cooking_actions_meta = load_cooking_actions()
        self.font_path = get_devanagari_font_path()

    def tokenize_corpus(self, remove_stopwords: bool = True) -> List[str]:
        """Tokenizes all recipe text in the corpus."""
        all_tokens = []
        for text in self.recipes_df.get('text', []):
            tokens = re.findall(r'[\u0900-\u097F]+', str(text))
            for t in tokens:
                t_clean = t.strip()
                if len(t_clean) > 1:
                    if remove_stopwords and t_clean in self.stopwords:
                        continue
                    all_tokens.append(t_clean)
        return all_tokens

    def generate_wordcloud_image(self, max_words: int = 100, width: int = 800, height: int = 400) -> Optional[plt.Figure]:
        """
        Generates Word Cloud with proper Devanagari font rendering (e.g. Nirmala UI).
        """
        tokens = self.tokenize_corpus(remove_stopwords=True)
        if not tokens:
            return None

        word_counts = Counter(tokens)

        # Wordcloud requires font_path for Devanagari
        wc_kwargs = {
            "width": width,
            "height": height,
            "background_color": "#0f172a",
            "max_words": max_words,
            "colormap": "Oranges",
            "collocations": False
        }
        if self.font_path:
            wc_kwargs["font_path"] = self.font_path

        wc = WordCloud(**wc_kwargs).generate_from_frequencies(word_counts)

        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0f172a')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        return fig

    def get_word_frequencies(self, top_n: int = 25) -> pd.DataFrame:
        """Calculates overall word frequencies in the corpus."""
        tokens = self.tokenize_corpus(remove_stopwords=True)
        counts = Counter(tokens).most_common(top_n)
        df = pd.DataFrame(counts, columns=["Word", "Frequency"])
        return df

    def get_ingredient_frequencies(self, top_n: int = 20) -> pd.DataFrame:
        """Calculates frequency of ingredients across all recipes."""
        ing_counter = Counter()
        for _, row in self.recipes_df.iterrows():
            ing_raw = str(row.get('ingredients_list', ''))
            ings = [i.strip() for i in ing_raw.split('|') if i.strip()]
            for item in ings:
                ing_counter[item] += 1
        
        df = pd.DataFrame(ing_counter.most_common(top_n), columns=["Ingredient", "Recipe_Count"])
        df["Percentage"] = (df["Recipe_Count"] / len(self.recipes_df) * 100).round(1)
        return df

    def get_cooking_action_frequencies(self, top_n: int = 15) -> pd.DataFrame:
        """Calculates frequency of cooking verbs in recipe texts."""
        action_counter = Counter()
        action_stems = [
            "घाला", "परता", "भाजा", "तळा", "शिजवा", "उकळा", "मिसळा", "वाटा",
            "किसा", "धुवा", "मळा", "लाटा", "वाफवा", "सजवा", "सर्व्ह करा"
        ]
        
        full_text = " ".join(self.recipes_df['text'].dropna())
        for act in action_stems:
            count = len(re.findall(rf"(?<![\u0900-\u097F0-9a-zA-Z]){re.escape(act)}(?![\u0900-\u097F0-9a-zA-Z])", full_text))
            if count > 0:
                action_counter[act] = count

        df = pd.DataFrame(action_counter.most_common(top_n), columns=["Cooking_Action", "Frequency"])
        return df

    def get_unit_frequencies(self) -> pd.DataFrame:
        """Calculates frequency of measurement units."""
        units = ["चमचा", "चमचे", "कप", "वाटी", "ग्रॅम", "किलो", "लिटर", "चिमूट", "तुकडा"]
        full_text = " ".join(self.recipes_df['text'].dropna())
        unit_counts = Counter()
        for u in units:
            cnt = len(re.findall(rf"(?<![\u0900-\u097F0-9a-zA-Z]){re.escape(u)}(?![\u0900-\u097F0-9a-zA-Z])", full_text))
            if cnt > 0:
                unit_counts[u] = cnt
        return pd.DataFrame(unit_counts.most_common(), columns=["Unit", "Frequency"])

    def get_recipe_statistics(self) -> Dict[str, Any]:
        """Calculates recipe corpus length and token statistics."""
        df = self.recipes_df.copy()
        
        # Token count per recipe
        df['token_count'] = df['text'].apply(lambda t: len(re.findall(r'[\u0900-\u097F0-9a-zA-Z]+', str(t))))
        
        all_tokens = re.findall(r'[\u0900-\u097F0-9a-zA-Z]+', " ".join(df['text'].dropna()))
        vocab = set(all_tokens)

        stats = {
            "total_recipes": len(df),
            "total_tokens": len(all_tokens),
            "vocabulary_size": len(vocab),
            "avg_recipe_length": round(df['token_count'].mean(), 1),
            "min_recipe_length": int(df['token_count'].min()),
            "max_recipe_length": int(df['token_count'].max()),
            "avg_cooking_time": round(df['cooking_time_minutes'].mean(), 1),
            "avg_ingredients_count": round(df['ingredients_count'].mean(), 1),
            "length_distribution": df[['title', 'category', 'cooking_time_minutes', 'ingredients_count', 'token_count']].to_dict(orient='records')
        }
        return stats

    def get_ingredient_cooccurrence_matrix(self, top_n: int = 10) -> pd.DataFrame:
        """Calculates co-occurrence counts between top ingredients."""
        top_ings_df = self.get_ingredient_frequencies(top_n)
        top_ings = top_ings_df["Ingredient"].tolist()

        matrix = pd.DataFrame(0, index=top_ings, columns=top_ings)

        for _, row in self.recipes_df.iterrows():
            ing_raw = str(row.get('ingredients_list', ''))
            ings_in_rec = set(i.strip() for i in ing_raw.split('|'))
            
            present_top = [i for i in top_ings if i in ings_in_rec]
            for i1 in present_top:
                for i2 in present_top:
                    matrix.loc[i1, i2] += 1

        return matrix

    def get_category_comparison(self) -> pd.DataFrame:
        """Compares recipe lengths, cooking times, and ingredient counts by category."""
        df = self.recipes_df.copy()
        df['token_count'] = df['text'].apply(lambda t: len(re.findall(r'[\u0900-\u097F0-9a-zA-Z]+', str(t))))
        
        summary = df.groupby('category').agg(
            recipe_count=('id', 'count'),
            avg_cooking_time=('cooking_time_minutes', 'mean'),
            avg_ingredients=('ingredients_count', 'mean'),
            avg_tokens=('token_count', 'mean')
        ).round(1).reset_index()
        return summary

    def get_missing_data_summary(self) -> pd.DataFrame:
        """Analyzes missing values in recipe metadata and instructions."""
        df = self.recipes_df.copy()
        missing_records = []

        total = len(df)
        fields = [
            ("title", "नाव (Title)"),
            ("category", "प्रवर्ग (Category)"),
            ("cooking_time_minutes", "स्वयंपाक वेळ (Cooking Time)"),
            ("temperature", "उष्णता पातळी (Temperature)"),
            ("ingredients_list", "घटक यादी (Ingredients List)"),
            ("text", "कृती मजकूर (Recipe Text)")
        ]

        for col, label in fields:
            missing_cnt = df[col].isna().sum() if col in df.columns else total
            missing_pct = round((missing_cnt / total) * 100, 1)
            missing_records.append({
                "Field": col,
                "Description": label,
                "Missing_Count": int(missing_cnt),
                "Missing_Percentage": f"{missing_pct}%",
                "Completeness": f"{round(100 - missing_pct, 1)}%"
            })

        return pd.DataFrame(missing_records)

    def get_correlation_data(self) -> pd.DataFrame:
        """Calculates correlation between numeric variables."""
        df = self.recipes_df.copy()
        df['token_count'] = df['text'].apply(lambda t: len(re.findall(r'[\u0900-\u097F0-9a-zA-Z]+', str(t))))
        
        numeric_cols = df[['cooking_time_minutes', 'ingredients_count', 'token_count']]
        return numeric_cols.corr().round(3)
