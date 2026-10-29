# RecipeParse: College Practical Experiment Mapping

This document details the mapping between college Natural Language Processing practical experiments (Experiments 2 through 8) and their concrete implementations in the RecipeParse codebase.

---

## 📋 Comprehensive Experiment Mapping Table

| College Practical | Practical Title | Key Concepts & Objectives | Core Implementation Module | Streamlit Page | Automated Test Suite |
|---|---|---|---|---|---|
| **Experiment 2** | Text Preprocessing | Tokenization, Stopword removal, Stemming vs Lemmatization, Devanagari script validation, Unicode NFC/NFD normalization, Numeral conversion (०-९ $\leftrightarrow$ 0-9), Punctuation cleaning | `src/preprocessing.py` | `pages/2_Text_Preprocessing.py` | `tests/test_preprocessing.py` |
| **Experiment 3** | Morphological Analysis and Regular Expressions | Quantities, Units, Durations, Temperatures, Cooking actions, Recipe step patterns, Morphological affix analysis | `src/regex_extractor.py`<br>`src/morphological_analysis.py` | `pages/3_Regex_Extractor.py` | `tests/test_regex_extractor.py` |
| **Experiment 4** | N-Gram Language Model | Corpus tokenization, Unigram/Bigram/Trigram counts, Add-k Laplace smoothing, Backoff fallback, Next-word prediction | `src/ngram_model.py` | `pages/4_Ngram_Model.py` | `tests/test_ngram_model.py` |
| **Experiment 5** | Chunking and Named Entity Recognition | Syntactic chunking (Noun Phrase - NP, Verb Phrase - VP, Postpositional Phrase - PP), 7-class recipe NER, HTML span highlighting | `src/chunking.py`<br>`src/recipe_ner.py` | `pages/5_Chunking_and_NER.py` | `tests/test_ner.py` |
| **Experiment 6** | Part-of-Speech Tagging using HMM and Viterbi | Hidden Markov Model, Initial $\pi$, Transition $A$, Emission $B$ matrices, Trellis dynamic programming, Morphological OOV handling | `src/hmm_pos_tagger.py`<br>`src/viterbi.py` | `pages/6_HMM_POS_Tagging.py` | `tests/test_hmm_viterbi.py` |
| **Experiment 7** | Word Sense Disambiguation using Lesk Algorithm | Polysemous word detection, WordNet/Lexicon signature extraction, Stopword removal, Word overlap score computation, Winning sense selection | `src/lesk_wsd.py` | `pages/7_Word_Sense_Disambiguation.py` | `tests/test_lesk_wsd.py` |
| **Experiment 8** | Exploratory Data Analysis & Word Cloud | Devanagari Word Cloud, Word/Ingredient/Action frequencies, Co-occurrence heatmap, Category analysis, Missing data, Correlations | `src/text_analytics.py` | `pages/8_EDA_Dashboard.py` | `tests/test_text_analytics.py` |
| **Integrated System** | Integrated Recipe Information Extractor | End-to-end recipe extraction, Structured table generation, Data exports (CSV, JSON) | `src/recipe_extractor.py` | `pages/1_Recipe_Analyzer.py` | `tests/test_integration.py` |

---

## 🔍 Detailed Practical Specifications

### Experiment 2: Text Preprocessing
- **Aim**: Clean and normalize Marathi recipe text for downstream NLP tasks.
- **Implemented Techniques**:
  1. Word Tokenization: preserves Devanagari conjuncts, matras, and numbers.
  2. Sentence Tokenization: normalizes Marathi danda `।` to `.` and splits at sentence boundaries.
  3. Stopword Removal: 148 curated functional words while strictly preserving cooking ingredients, units, and negations (नाही, नका, नको).
  4. Rule-Based Stemmer: strips inflectional case endings (`-ात`, `-ाचे`, `-ाला`, `-ून`, `-ताना`).
  5. Lemmatizer: domain dictionary mapping inflected cooking terms to root lemmas (`कांद्याला` $\to$ `कांदा`, `तेलात` $\to$ `तेल`).
  6. Script Validation: checks Unicode range `\u0900-\u097F` and computes Devanagari density percentage.
  7. Numeral Normalization: converts Devanagari numerals (`०-९`) to standard Arabic digits (`0-9`) and vice versa.

### Experiment 3: Morphological Analysis & Regular Expressions
- **Aim**: Identify structured culinary patterns using regular expressions and morphological parsing.
- **Implemented Patterns**:
  - Quantities: `दोन चमचे`, `एक कप`, `५० ग्रॅम`, `०.५ लिटर`, `1/2 cup`.
  - Units: `चमचा`, `चमचे`, `कप`, `वाटी`, `ग्रॅम`, `किलो`, `लिटर`, `मिलीलीटर`, `चिमूट`.
  - Durations: `पाच मिनिटे`, `१० मिनिटे`, `अर्धा तास`, `दोन तास`.
  - Temperatures: `मंद आच`, `मध्यम आचेवर`, `कडकडीत`, `180°C`.
  - Actions: `घाला`, `परता`, `शिजवा`, `तळा`, `उकळा`, `भाजा`, `वाटा`, `मिसळा`.

### Experiment 4: N-Gram Language Model
- **Aim**: Model word transition probabilities and predict adjacent words in cooking instructions.
- **Algorithms**: Maximum Likelihood Estimation with Add-$k$ smoothing ($k=0.5$).
- **Features**: Trigram with backoff to Bigram and Unigram when context is unseen.

### Experiment 5: Chunking & Named Entity Recognition (NER)
- **Aim**: Parse shallow syntactic phrases and extract named culinary entities.
- **Syntactic Chunks**:
  - `NP`: Noun Phrase (`[एक चमचा मोहरी]`, `[चिरलेला कांदा]`)
  - `VP`: Verb Phrase (`[गरम करा]`, `[शिजवून घ्या]`)
  - `PP`: Postpositional Phrase (`[पातेल्यात]`, `[मध्यम आचेवर]`)
- **Named Entities**: `INGREDIENT`, `QUANTITY`, `UNIT`, `ACTION`, `DURATION`, `TEMPERATURE`, `UTENSIL`.

### Experiment 6: HMM POS Tagging & Viterbi Algorithm
- **Aim**: Decode the optimal Part-of-Speech tag sequence without third-party black-box taggers.
- **Tagset**: `NOUN`, `VERB`, `ADJ`, `ADV`, `NUM`, `PRONOUN`, `POSTPOSITION`, `PARTICLE`, `OTHER`.
- **Implementation**: Computes full Trellis matrix $V(t, j)$ with log-probability arithmetic.

### Experiment 7: Word Sense Disambiguation (Lesk Algorithm)
- **Aim**: Disambiguate polysemous Marathi words from context sentence.
- **Target Words**: `तेल`, `तूप`, `पान`, `बारीक`, `रस`, `तास`.
- **Implementation**: Computes genuine word overlap: $\text{Score}(S) = |\text{Context} \cap \text{Definition}(S)|$.

### Experiment 8: Exploratory Data Analysis & Word Cloud
- **Aim**: Statistical and visual text analytics over Marathi recipe corpus.
- **Features**: Devanagari Word Cloud, Frequency bar charts, Ingredient co-occurrence matrix & heatmap, category comparisons, missing data analysis, and CSV downloads.
