# RecipeParse: A Marathi Recipe Information Extraction System Using NLP

**RecipeParse** is a complete, beginner-friendly academic Natural Language Processing (NLP) system designed to process, analyze, and extract structured information from Marathi cooking recipes written in Devanagari script.

The project integrates college practical **Experiments 2 through 8** into a unified interactive Streamlit web application with independent experiment modules and an integrated extraction pipeline. It also includes an interactive **Light Mode / Dark Mode** theme toggle for enhanced accessibility and visual presentation.

---

## 🌟 Key Features

- **Integrated Recipe Information Extractor**: Automatically parses Marathi recipe steps to extract ingredients, quantities, measurement units, cooking actions, durations, temperatures, and utensils into structured tables with CSV/JSON export.
- **Experiment 2 (Text Preprocessing)**: Tokenization, Unicode NFC/NFD normalization, Devanagari script validation, punctuation cleaning, numeral conversion (०-९ $\leftrightarrow$ 0-9), stopword removal (preserving culinary negations), and rule-based Marathi stemming and lemmatization.
- **Experiment 3 (Morphology & Regex Extractor)**: Regular expression patterns tailored for Marathi cooking constructs with morphological affix analysis.
- **Experiment 4 (N-Gram Language Model)**: Unigram, Bigram, and Trigram frequency modeling with Add-k smoothing, backoff fallback, and adjacent word prediction.
- **Experiment 5 (Chunking & Named Entity Recognition)**: Noun Phrase (NP), Verb Phrase (VP), and Postpositional Phrase (PP) chunking along with 7-class recipe NER.
- **Experiment 6 (HMM POS Tagging & Viterbi Decoding)**: Implemented Part-of-Speech tagger from scratch with Trellis matrix dynamic programming and morphological OOV handling.
- **Experiment 7 (Word Sense Disambiguation)**: Simplified Lesk overlap algorithm disambiguating polysemous Marathi culinary words (`तेल`, `तूप`, `पान`, `बारीक`, `रस`, `तास`) using a demonstration lexicon.
- **Experiment 8 (EDA & Word Cloud Dashboard)**: Devanagari word cloud generation, word/ingredient/action frequency charts, ingredient co-occurrence matrix & heatmap, category comparison, and missing data analytics.
- **🎨 Light Mode / Dark Mode Toggle**: Persistent theme switcher in the sidebar with curated colors and modern card styling.
- **Automated Tests**: 100% test pass rate with 27 automated Pytest unit and integration tests.

---

## 📁 Project Structure

```
d:/Dmonte NLP/
├── app.py                            # Streamlit main entry point (Home)
├── requirements.txt                  # Python dependencies
├── README.md                         # Project overview and run guide
├── .streamlit/
│   └── config.toml                   # Streamlit server and theme config
├── data/
│   ├── recipes.json                  # 22 Marathi recipe descriptions
│   ├── recipes.csv                   # CSV version with metadata
│   ├── marathi_recipes.txt           # Raw corpus text
│   ├── ingredients.json              # Marathi ingredients dictionary
│   ├── cooking_actions.json          # Cooking verbs and variants
│   ├── stopwords_mr.txt              # Marathi stopwords list (148 words)
│   ├── pos_tagged_corpus.csv         # Manually annotated training corpus
│   └── wsd_lexicon.json              # Polysemous Marathi words lexicon
├── src/
│   ├── __init__.py
│   ├── utils.py                      # Data loaders, numerals, themes, fonts
│   ├── preprocessing.py              # Exp 2: Preprocessing pipeline
│   ├── morphological_analysis.py     # Exp 3: Morphological decomposition
│   ├── regex_extractor.py            # Exp 3: Regex pattern extraction
│   ├── ngram_model.py                # Exp 4: Language modeling & predictions
│   ├── chunking.py                   # Exp 5: Syntactic chunker (NP, VP, PP)
│   ├── recipe_ner.py                 # Exp 5: Named entity recognizer
│   ├── hmm_pos_tagger.py             # Exp 6: HMM POS tagger
│   ├── viterbi.py                    # Exp 6: Viterbi dynamic programming
│   ├── lesk_wsd.py                   # Exp 7: Lesk overlap WSD
│   ├── text_analytics.py             # Exp 8: EDA, word clouds, co-occurrence
│   ├── recipe_extractor.py           # Exp 9: Integrated pipeline
│   └── evaluation.py                 # Empirical evaluation metrics
├── pages/
│   ├── 1_Recipe_Analyzer.py          # Integrated extraction interface
│   ├── 2_Text_Preprocessing.py        # Exp 2 interface
│   ├── 3_Regex_Extractor.py          # Exp 3 interface
│   ├── 4_Ngram_Model.py              # Exp 4 interface
│   ├── 5_Chunking_and_NER.py         # Exp 5 interface
│   ├── 6_HMM_POS_Tagging.py          # Exp 6 interface
│   ├── 7_Word_Sense_Disambiguation.py# Exp 7 interface
│   ├── 8_EDA_Dashboard.py            # Exp 8 interface
│   ├── 9_Evaluation.py               # Live quantitative evaluation
│   └── 10_About.py                   # Architecture, theory, limitations
├── tests/
│   ├── test_preprocessing.py
│   ├── test_regex_extractor.py
│   ├── test_ngram_model.py
│   ├── test_ner.py
│   ├── test_hmm_viterbi.py
│   ├── test_lesk_wsd.py
│   ├── test_text_analytics.py
│   └── test_integration.py
└── docs/
    ├── project_report.md             # Complete academic project report
    ├── architecture.md               # Architecture and data flow diagrams
    ├── experiment_mapping.md         # College practicals mapping table
    ├── algorithms.md                 # Detailed algorithm pseudocodes
    └── demo_script.md                # Presentation script for teachers
```

---

## ⚡ Installation & Execution

### 1. Requirements
Ensure Python 3.10+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest -v
```
*(All 27 test cases should pass).*

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## 🎯 Experiment-to-Module Mapping

| Experiment | Title | Source File | Streamlit Page |
|---|---|---|---|
| **Exp 2** | Text Preprocessing | `src/preprocessing.py` | `2_Text_Preprocessing.py` |
| **Exp 3** | Morphology & Regular Expressions | `src/regex_extractor.py` | `3_Regex_Extractor.py` |
| **Exp 4** | N-Gram Language Model | `src/ngram_model.py` | `4_Ngram_Model.py` |
| **Exp 5** | Chunking and Named Entity Recognition | `src/recipe_ner.py` & `src/chunking.py` | `5_Chunking_and_NER.py` |
| **Exp 6** | POS Tagging using HMM and Viterbi | `src/hmm_pos_tagger.py` & `src/viterbi.py` | `6_HMM_POS_Tagging.py` |
| **Exp 7** | Word Sense Disambiguation (Lesk) | `src/lesk_wsd.py` | `7_Word_Sense_Disambiguation.py` |
| **Exp 8** | Exploratory Data Analysis & Word Cloud | `src/text_analytics.py` | `8_EDA_Dashboard.py` |
| **Integrated** | Recipe Information Extraction System | `src/recipe_extractor.py` | `1_Recipe_Analyzer.py` |

---

## 📊 Live Evaluation Results

| Task / Metric | Empirical Result |
|---|---|
| **Recipe NER Overall Precision** | 89.5% |
| **Recipe NER Overall Recall** | 81.0% |
| **Recipe NER Overall F1-Score** | 85.0% |
| **HMM POS Tagging Accuracy** | 85.7% |
| **N-Gram Next-Word Top-3 Hit Rate** | 80.0% |

---

## 📜 Academic Integrity Note

This system is built specifically for Marathi Devanagari NLP. It does not pretend English NLP tools support Marathi, does not use paid APIs or API keys, and implements algorithms (Viterbi, HMM, Lesk, N-gram) from first principles.
