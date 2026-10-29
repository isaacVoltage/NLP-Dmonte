# RecipeParse: Teacher Demonstration & Viva Script

This script is a step-by-step guide for presenting **RecipeParse** to a teacher, professor, or external practical examiner.

---

## 🎙️ Phase 1: Introduction & Project Overview (1-2 Minutes)

### Speaker Script:
> "Good morning / afternoon, Sir / Ma'am. Today I am presenting my Natural Language Processing project: **RecipeParse: A Marathi Recipe Information Extraction System Using NLP**."
> 
> "Most existing NLP systems are developed for English. Marathi, written in Devanagari script, presents unique linguistic challenges such as complex agglutinative morphology, free word order, and dependent vowel signs (matras). 
> 
> This project demonstrates how we integrated all college practicals from **Experiment 2 to Experiment 8** into one unified, production-ready system with a modern Streamlit interface, automated test suite, and light/dark theme modes."

---

## 🖥️ Phase 2: Live Demonstration Walkthrough (5-7 Minutes)

### Step 1: Home Page & Dark/Light Mode Toggle
1. Open the app (`http://localhost:8501`).
2. Point out the KPI metric cards: **22 recipes**, **60+ ingredients**, and **100% test pass status**.
3. **Show the Theme Switcher**: Toggle from `🌙 Dark Mode` to `☀️ Light Mode` in the sidebar and demonstrate that the styling smoothly switches and persists.

### Step 2: Exp 1 / Integrated Recipe Analyzer (The "Hero" Feature)
1. Navigate to **1. Recipe Analyzer** in the sidebar.
2. Select the sample: *"कांदे पोहे (Kande Pohe)"* or paste:
   ```text
   एका पातेल्यात दोन चमचे तेल गरम करा. त्यात एक चमचा मोहरी घाला. त्यानंतर बारीक चिरलेला कांदा घालून पाच मिनिटे परता.
   ```
3. Click **विश्लेषण करा (Analyze Recipe)**.
4. Show the teacher:
   - **Structured Table**: Ingredients (`तेल`, `मोहरी`, `कांदा`), Quantities (`दोन`, `एक`), Units (`चमचे`, `चमचा`), Actions (`गरम करा`, `घाला`, `परता`), and Durations (`पाच मिनिटे`).
   - **Inline Entity Highlighting**: Colorful badges showing `[INGREDIENT]`, `[QUANTITY]`, `[UNIT]`, `[ACTION]`.
   - **Download Buttons**: Click *CSV डाउनलोड करा* to show instantaneous CSV export.

### Step 3: Experiment 2 (Text Preprocessing)
1. Navigate to **2. Text Preprocessing**.
2. Show:
   - **Devanagari Script Validation**: Percentage of Devanagari characters (e.g. 96%).
   - **Tokenization**: Word and sentence segmentation preserving conjuncts.
   - **Stopword Removal**: Show that stopwords like `आणि`, `त्यात` are removed while cooking negations (`नाही`, `नका`) are preserved.
   - **Stemming vs Lemmatization**: Show `कांद्याला` $\to$ `कांदा` and `तेलात` $\to$ `तेल`.
   - **Numeral Normalization**: Conversion between `५०` and `50`.

### Step 4: Experiment 3 (Morphology & Regex Extractor)
1. Navigate to **3. Regex Extractor**.
2. Explain the lookaround boundaries `(?<![\u0900-\u097F0-9a-zA-Z])` needed for Devanagari characters where standard `\b` fails.
3. Show extracted quantities, units, and durations.

### Step 5: Experiment 4 (N-Gram Language Model)
1. Navigate to **4. Ngram Model**.
2. Type `दोन चमचे` and show Bigram vs Trigram predictions (predicts `तेल`, `तूप`, `मोहरी` with Add-$k$ Laplace smoothed probabilities).
3. Emphasize the academic disclaimer that predictions are corpus-derived statistical models.

### Step 6: Experiment 5 (Chunking & NER)
1. Navigate to **5. Chunking & NER**.
2. Input: `एका कढईत दोन चमचे तेल गरम करा.`
3. Show syntactic chunks:
   - `[PP: एका कढईत]` (Postpositional locative phrase)
   - `[NP: दोन चमचे तेल]` (Noun phrase with quantity + unit + noun)
   - `[VP: गरम करा]` (Verb phrase with action instruction)

### Step 7: Experiment 6 (HMM POS Tagging & Viterbi Algorithm)
1. Navigate to **6. HMM POS Tagging**.
2. Emphasize to the teacher:
   > "Sir/Ma'am, this POS tagger is written from scratch. We did not use any external black-box library. We trained the HMM matrices ($A, B, \pi$) and implemented Viterbi dynamic programming."
3. Click the **Viterbi Trellis Steps** tab and show the step-by-step matrix log-probabilities!

### Step 8: Experiment 7 (Word Sense Disambiguation - Lesk Algorithm)
1. Navigate to **7. Word Sense Disambiguation**.
2. Select target word: `तेल`.
   - Context 1 (Cooking): `कढईत दोन चमचे तेल गरम करून मोहरीची फोडणी द्या.` $\to$ Disambiguates to **खाद्यतेल (Cooking Oil)** with overlap words `कढई`, `मोहरी`, `गरम`.
   - Context 2 (Ritual/Lighting): `देवापुढे समईत तेल ओतून दिवा प्रज्वलित करा.` $\to$ Disambiguates to **दिवा/इंधन तेल (Lamp oil)** with overlap words `दिवा`.

### Step 9: Experiment 8 (EDA Dashboard)
1. Navigate to **8. EDA Dashboard**.
2. Show the rendered **Devanagari Word Cloud** using `Nirmala.ttc`.
3. Show the **Ingredient Co-occurrence Matrix & Heatmap** (e.g. `मोहरी` and `जिरे` appearing together).

### Step 10: Evaluation & Automated Tests
1. Navigate to **9. Evaluation**.
2. Point out that the metrics are empirical:
   - **NER Overall F1**: 85.0%
   - **POS Tagging Accuracy**: 85.7%
   - **Next-Word Top-3 Hit Rate**: 80.0%
3. Open terminal and run: `python -m pytest -v` to demonstrate 27 passed tests in real-time.

---

## ❓ Anticipated Teacher Questions & Model Answers

**Q1: Why couldn't you use the NLTK Porter Stemmer for Marathi?**
> *Answer*: The Porter Stemmer was designed specifically for English morphological rules (such as stripping `-ing`, `-ed`, `-s`). Marathi is an inflectional Indo-Aryan language with complex agglutinative case markers (`-ात`, `-ाला`, `-ाचे`) and internal stem vowel shifts. Applying English suffix rules produces nonsensical results on Devanagari Unicode. We built a custom Marathi suffix-stripper and lemmatizer.

**Q2: What is the purpose of the Viterbi algorithm in Experiment 6?**
> *Answer*: In a Hidden Markov Model, calculating the probability of all possible state paths has exponential complexity $O(|S|^T)$. The Viterbi algorithm uses dynamic programming to find the most probable sequence of hidden states $\arg\max_S P(W, S)$ in polynomial time $O(T \cdot |S|^2)$ by storing the maximum path probability ending at each state at each time step.

**Q3: How does the Lesk algorithm resolve ambiguous words?**
> *Answer*: The Simplified Lesk algorithm compares the content words in the context sentence with the dictionary definition (gloss) and example sentences of each candidate sense. The sense with the highest word overlap count is selected.
