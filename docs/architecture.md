# RecipeParse: System Architecture and Data Flow

This document details the high-level architecture, module decomposition, and data flow of the **RecipeParse** system.

---

## 🏛️ System Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                            STREAMLIT WEB APPLICATION                              |
|                                                                                   |
|  [Sidebar Navigation]                                    [Theme Controller]       |
|  - Home (app.py)                                         - 🌙 Dark Mode           |
|  - 1_Recipe_Analyzer.py                                  - ☀️ Light Mode           |
|  - 2_Text_Preprocessing.py                                                        |
|  - 3_Regex_Extractor.py                                                           |
|  - 4_Ngram_Model.py                                                               |
|  - 5_Chunking_and_NER.py                                                          |
|  - 6_HMM_POS_Tagging.py                                                           |
|  - 7_Word_Sense_Disambiguation.py                                                 |
|  - 8_EDA_Dashboard.py                                                             |
|  - 9_Evaluation.py                                                                |
|  - 10_About.py                                                                    |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        RECIPEPARSE NLP CORE ENGINE (src/)                         |
|                                                                                   |
|  +--------------------+   +----------------------+   +-------------------------+  |
|  | preprocessing.py   |   | regex_extractor.py   |   | chunking.py             |  |
|  | - Tokenizer        |   | - Indic Boundaries   |   | - NP/VP/PP Chunking     |  |
|  | - Stemmer/Lemma    |   | - Quantity & Unit    |   +-------------------------+  |
|  | - Script Validator |   | - Duration & Temp    |   | recipe_ner.py           |  |
|  | - Normalizer       |   | - Action & Utensil   |   | - 7-Class Entity Parser |  |
|  +--------------------+   +----------------------+   +-------------------------+  |
|                                                                                   |
|  +--------------------+   +----------------------+   +-------------------------+  |
|  | hmm_pos_tagger.py  |   | lesk_wsd.py          |   | text_analytics.py       |  |
|  | - HMM Estimator    |   | - Overlap Evaluator  |   | - Devanagari Word Cloud |  |
|  | - viterbi.py       |   | - Lexicon Query      |   | - Frequency Analytics   |  |
|  | - Trellis Decoder  |   | - Sense Selection    |   | - Co-occurrence Heatmap |  |
|  +--------------------+   +----------------------+   +-------------------------+  |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  | recipe_extractor.py: Integrated Recipe Pipeline                             |  |
|  | evaluation.py: Quantitative Performance Benchmarks (P/R/F1/Accuracy)        |  |
|  | utils.py: Unicode conversions, font resolvers, theme stylesheets           |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                              DATA REPOSITORY (data/)                              |
|                                                                                   |
|  - recipes.json / recipes.csv / marathi_recipes.txt (22 Marathi Recipes)         |
|  - ingredients.json (60+ culinary items with categories and inflections)          |
|  - cooking_actions.json (Marathi cooking verbs and conjugations)                  |
|  - stopwords_mr.txt (148 curated functional words)                               |
|  - pos_tagged_corpus.csv (20 annotated sentences, 158 tokens)                    |
|  - wsd_lexicon.json (Polysemous Marathi words with senses and glosses)            |
+-----------------------------------------------------------------------------------+
```

---

## 🔄 End-to-End Data Flow

```
[Raw Marathi Recipe Text]
           │
           ▼
[1. Text Preprocessing]
  ├── Unicode NFC Normalization
  ├── Whitespace & Punctuation Cleaning
  ├── Script Validation (% Devanagari characters)
  └── Sentence & Word Tokenization
           │
           ▼
[2. Pattern Extraction & Morphology]
  ├── Regex Extractor (Quantities, Units, Durations, Temperatures)
  ├── Morphological Suffix Analysis
  └── Ingredient Dictionary Matcher
           │
           ▼
[3. Syntactic & Entity Parsing]
  ├── Chunking: NP (Noun Phrases), VP (Verb Phrases), PP (Postpositions)
  └── Recipe NER: INGREDIENT, QUANTITY, UNIT, ACTION, DURATION, TEMP, UTENSIL
           │
           ▼
[4. Sequence Modeling & Disambiguation]
  ├── HMM / Viterbi POS Tagging (Token syntactic categories)
  └── Lesk Algorithm (Polysemous word disambiguation)
           │
           ▼
[5. Structured Synthesis & Analytics]
  ├── Synthesis: Structured Recipe Table (Ingredient, Quantity, Unit, Action, Context)
  ├── Word Frequency & Co-occurrence Matrix
  └── Export: CSV & JSON
```
