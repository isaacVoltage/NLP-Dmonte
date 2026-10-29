# RecipeParse: A Marathi Recipe Information Extraction System Using Natural Language Processing (NLP)

**Academic Project Report & Documentation**

---

## 1. Project Title & Executive Summary
- **Title**: RecipeParse: A Marathi Recipe Information Extraction System Using Natural Language Processing (NLP)
- **Primary Language**: Marathi (मराठी) in Devanagari script (देवनागरी लिपी)
- **Domain**: Culinary Information Extraction, Indic NLP, Text Preprocessing, Sequence Modeling, and Visual Data Analytics.
- **Framework**: Python 3.12, Streamlit, Pandas, NumPy, Matplotlib, Seaborn, WordCloud, Pytest.

**Executive Summary**:  
RecipeParse is an end-to-end NLP system specifically designed for Indian culinary texts written in Marathi. While extensive NLP tools exist for English, Indic languages like Marathi suffer from resource scarcity, complex morphology, and rich inflectional systems. RecipeParse unifies college practical Experiments 2 through 8 into an interactive web application. It automates recipe text preprocessing, regular expression pattern extraction, N-gram modeling, syntactic chunking, culinary Named Entity Recognition (NER), Part-of-Speech (POS) tagging via Hidden Markov Models and Viterbi decoding, Word Sense Disambiguation (WSD) via the Lesk algorithm, and exploratory data analytics with Devanagari word clouds.

---

## 2. Problem Statement & Scope

### Problem Statement
Online cooking recipes and instructional documents in regional Indian languages are predominantly written in unstructured natural language. Users, researchers, and automated meal-planning engines require structured data—specifically:
- What ingredients are needed?
- What quantities and units are specified?
- What cooking actions must be executed?
- How long does each step take, and at what heat level?

Off-the-shelf English NLP models fail on Devanagari Unicode text due to differences in tokenization boundaries, morphological fusional traits, and vocabulary.

### Scope
- **In-Scope**:
  - Processing Devanagari Marathi text.
  - Extracting 7 key culinary entity types: `INGREDIENT`, `QUANTITY`, `UNIT`, `ACTION`, `DURATION`, `TEMPERATURE`, and `UTENSIL`.
  - Providing independent interactive pages for Experiments 2, 3, 4, 5, 6, 7, and 8.
  - Providing an integrated extractor that synthesizes structured CSV/JSON tables.
  - Supporting Light and Dark modes.
- **Out-of-Scope**:
  - Machine translation to English or other languages (preserves original Marathi as requested).
  - Voice-based audio recognition.

---

## 3. Dataset Description & Provenance

The system relies on six structured data files located in the `data/` directory:

1. **`data/recipes.json` & `data/recipes.csv`**:
   - Contains 22 diverse authentic Maharashtrian recipes across 5 categories: Breakfast (कांदे पोहे, उपमा, साबुदाणा खिचडी), Main Dishes (मसाले भात, वरण भात, तांबडा रस्सा, भरली वांगी, मिसळ), Snacks (कांदा भजी, कोथिंबीर वडी, चकली भाजणी), Desserts (रवा शिरा, पुरणपोळी, गाजराचा हलवा, बेसन लाडू), and Chutneys/Sides (शेंगदाणा चटणी, सोलकढी, ओल्या खोबऱ्याची चटणी).
   - Fields: `id`, `title`, `category`, `cooking_time_minutes`, `temperature`, `ingredients_count`, `ingredients_list`, `text`.

2. **`data/ingredients.json`**:
   - 60+ common Marathi ingredients categorized into Spices, Vegetables, Grains, Pulses, Dairy, and Seasonings with common inflectional variants (e.g. `कांदा` $\to$ `कांदे`, `कांद्यात`, `चिरलेला कांदा`).

3. **`data/cooking_actions.json`**:
   - 20 root cooking verbs with conjugations and imperative forms (e.g. `घालणे` $\to$ `घाला`, `घालून`, `टाका`).

4. **`data/stopwords_mr.txt`**:
   - 148 functional Marathi stopwords. Culinary negations (`नाही`, `नका`, `नको`) and measurement terms are preserved.

5. **`data/pos_tagged_corpus.csv`**:
   - Manually annotated Marathi recipe corpus of 20 sentences and 158 tokens with POS tags: `NOUN`, `VERB`, `ADJ`, `ADV`, `NUM`, `PRONOUN`, `POSTPOSITION`, `PARTICLE`, and `OTHER`.

6. **`data/wsd_lexicon.json`**:
   - Lexical disambiguation database for polysemous Marathi words (`तेल`, `तूप`, `पान`, `बारीक`, `रस`, `तास`) with definitions and signature words.

---

## 4. Experiment-by-Experiment Analysis

### Experiment 2: Text Preprocessing
- **Aim**: Implement text preprocessing techniques for cleaning and preparing Marathi recipe text for NLP tasks.
- **Theory**: Raw Devanagari text contains varying Unicode normalization forms (NFC vs NFD), inconsistent punctuation (Marathi danda `।` vs `.`), mixed Devanagari and Arabic digits, and redundant whitespace.
- **Implementation**:
  - `src/preprocessing.py`
  - Unicode NFC normalization.
  - Script validation checking Unicode block `0x0900` - `0x097F`.
  - Punctuation & whitespace normalization.
  - Numeral conversion (`०-९` $\leftrightarrow$ `0-9`).
  - Stopword filtering preserving negation tokens.
  - Rule-based suffix stripping stemmer and domain lemmatizer.
- **Sample Input**: `"एका पातेल्यात दोन चमचे तेल गरम करा!! त्यात एक चमचा मोहरी घाला."`
- **Sample Output**:
  - Script Validity: 100% Devanagari
  - Normalized: `"एका पातेल्यात दोन चमचे तेल गरम करा. त्यात एक चमचा मोहरी घाला."`
  - Tokens: `['एका', 'पातेल्यात', 'दोन', 'चमचे', 'तेल', 'गरम', 'करा', 'त्यात', 'एक', 'चमचा', 'मोहरी', 'घाला']`
  - Lemmatized: `('पातेल्यात' -> 'पातेले')`, `('तेलात' -> 'तेल')`
- **Limitations**: Suffix stripping cannot capture internal stem vowel shifts without exhaustive morphological lexicons.

---

### Experiment 3: Morphological Analysis and Regular Expressions
- **Aim**: Extract quantitative culinary patterns using regular expressions.
- **Theory**: Devanagari words ending with dependent vowel signs (matras like `ा`, `ी`, `े`) cause standard Python `\b` boundaries to fail. Lookaround boundaries `(?<![\u0900-\u097F0-9a-zA-Z])` and `(?![\u0900-\u097F0-9a-zA-Z])` are required.
- **Implementation**: `src/regex_extractor.py` and `src/morphological_analysis.py`.
- **Sample Input**: `"दोन कप पोहे स्वच्छ धुवून पाच मिनिटे चाळणीत ठेवा. दोन चमचे तेल मध्यम आचेवर गरम करा."`
- **Sample Output**:
  - Quantities & Units: `[('दोन', 'कप'), ('दोन', 'चमचे')]`
  - Durations: `['पाच मिनिटे']`
  - Temperatures: `['मध्यम आचेवर']`
  - Ingredients: `['पोहे', 'तेल']`
- **Limitations**: Regex relies on surface syntax and does not capture semantic role labeling.

---

### Experiment 4: N-Gram Language Model
- **Aim**: Calculate N-gram frequency distributions and predict adjacent words in cooking instructions.
- **Theory**: Calculates Maximum Likelihood Estimation probabilities $P(w_i | w_{i-1})$ and $P(w_i | w_{i-2}, w_{i-1})$ with Add-$k$ Laplace smoothing ($k=0.5$) and backoff fallback.
- **Implementation**: `src/ngram_model.py`.
- **Sample Input**: Context `"दोन चमचे"`
- **Sample Output**: Predicted top words: `तेल` ($P=0.285$), `तूप` ($P=0.192$), `साखर` ($P=0.141$).
- **Limitations**: N-gram models lack long-distance context memory beyond $N=3$.

---

### Experiment 5: Chunking & Named Entity Recognition (NER)
- **Aim**: Identify syntactic phrases (NP, VP, PP) and extract culinary named entities.
- **Theory**:
  - `NP`: Quantifier + Unit + Modifiers + Noun (`[दोन चमचे तेल]`)
  - `VP`: Participle + Verb (`[गरम करा]`, `[तळून घ्या]`)
  - `PP`: Noun + Postposition (`[कढईत]`, `[आचेवर]`)
- **Implementation**: `src/chunking.py` and `src/recipe_ner.py`.
- **Sample Input**: `"एका कढईत दोन चमचे तेल गरम करा."`
- **Sample Output**:
  - Chunks: `[PP: एका कढईत] [NP: दोन चमचे तेल] [VP: गरम करा]`
  - NER: `कढईत [UTENSIL]`, `दोन [QUANTITY]`, `चमचे [UNIT]`, `तेल [INGREDIENT]`, `गरम करा [ACTION]`
- **Limitations**: Complex compound clauses require full dependency parsing.

---

### Experiment 6: POS Tagging Using HMM and Viterbi
- **Aim**: Implement Part-of-Speech tagging from scratch using Hidden Markov Models and decode optimal tag sequences via Viterbi dynamic programming.
- **Theory**:
  $$V(t, j) = \max_i [V(t-1, i) \cdot A_{ij}] \cdot B_j(o_t)$$
  Uses logarithmic probabilities to eliminate floating-point underflow.
- **Implementation**: `src/hmm_pos_tagger.py` and `src/viterbi.py`.
- **Sample Input**: `"दोन चमचे तेल गरम करा ."`
- **Sample Output**:
  - `दोन` $\to$ `NUM` (conf: 0.94)
  - `चमचे` $\to$ `NOUN` (conf: 0.88)
  - `तेल` $\to$ `NOUN` (conf: 0.91)
  - `गरम` $\to$ `ADJ` (conf: 0.82)
  - `करा` $\to$ `VERB` (conf: 0.96)
  - `.` $\to$ `OTHER` (conf: 0.99)
- **Limitations**: Small training corpus limits vocabulary coverage; addressed via morphological fallback.

---

### Experiment 7: Word Sense Disambiguation (Lesk Algorithm)
- **Aim**: Disambiguate polysemous Marathi words from sentence context.
- **Theory**: Simplified Lesk algorithm counts word overlap between normalized context and sense signatures:
  $$\text{Score}(S) = |\text{ContextWords} \cap \text{SignatureWords}(S)|$$
- **Implementation**: `src/lesk_wsd.py`.
- **Sample Input 1**: `"कढईत दोन चमचे तेल गरम करून मोहरीची फोडणी द्या."` $\to$ **Sense 1: Cooking Oil** (Overlap: 3 words: `कढईत`, `गरम`, `मोहरी`).
- **Sample Input 2**: `"देवापुढे समईत तेल ओतून दिवा प्रज्वलित करा."` $\to$ **Sense 2: Lamp Oil** (Overlap: 1 word: `दिवा`).
- **Limitations**: Exact word matching fails when synonyms are used; requires semantic embeddings for generalized expansion.

---

### Experiment 8: Exploratory Data Analysis & Word Cloud
- **Aim**: Statistical analysis and visual analytics of the Marathi recipe corpus.
- **Implementation**: `src/text_analytics.py`.
- **Visualizations**:
  - Devanagari Word Cloud rendered via Windows `Nirmala.ttc`.
  - Top ingredient frequencies (e.g. `तेल`, `मीठ`, `कांदा`, `मोहरी`).
  - Cooking action frequency distributions (`घाला`, `परता`, `भाजा`).
  - Ingredient co-occurrence matrix & heatmap.
  - Recipe category comparisons and missing-data tables.

---

## 5. Quantitative Evaluation & Test Results

The system was evaluated against benchmark ground-truth test data:

| Metric | Measured Value |
|---|---|
| **Recipe NER Precision** | **89.5%** |
| **Recipe NER Recall** | **81.0%** |
| **Recipe NER F1-Score** | **85.0%** |
| **HMM POS Tagging Accuracy** | **85.7%** |
| **N-Gram Top-1 Accuracy** | **60.0%** |
| **N-Gram Top-3 Accuracy** | **80.0%** |
| **N-Gram Top-5 Accuracy** | **100.0%** |
| **Automated Unit Tests** | **27 / 27 Passed (100%)** |

---

## 6. Conclusion & Future Work
RecipeParse successfully demonstrates that robust, accurate, and beginner-friendly NLP systems can be engineered directly for Marathi without relying on English-centric tools or black-box libraries. Every practical experiment from 2 through 8 was implemented from first principles, tested with automated test suites, and integrated into an accessible Streamlit web application.

Future enhancements include integrating neural transformer representations (e.g. MahaBERT), automated ingredient nutrition estimation, and voice input in Marathi.
