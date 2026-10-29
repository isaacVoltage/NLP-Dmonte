"""
RecipeParse - About Project Page
Complete documentation of algorithms, architecture, experiment mapping, and academic details.
"""
import streamlit as st
import pandas as pd
from src.utils import init_app_page

current_theme = init_app_page("About - प्रकल्पाविषयी", "ℹ️")

st.markdown(
    """
    <div class="main-header">
        <h1>ℹ️ About RecipeParse (प्रकल्पाविषयी माहिती)</h1>
        <p>A Marathi Recipe Information Extraction System Using Natural Language Processing (NLP)</p>
    </div>
    """,
    unsafe_allow_html=True
)

tab_arch, tab_algo, tab_data, tab_limit = st.tabs([
    "🏛️ सिस्टीम आर्किटेक्चर (System Architecture)",
    "⚙️ वापरलेले अल्गोरिदम (Algorithms Explained)",
    "📦 डेटासेट वर्णन (Dataset & Sources)",
    "⚠️ मर्यादा व भविष्यातील वाव (Limitations & Future Scope)"
])

with tab_arch:
    st.subheader("प्रणालीची वास्तुकला (System Architecture & Pipeline)")
    st.markdown(
        """
        ```
        +-------------------------------------------------------------------------+
        |                 वापरकर्ता इनपुट (User Input: Marathi Recipe Text)        |
        +-------------------------------------------------------------------------+
                                           |
                                           v
        +-------------------------------------------------------------------------+
        |  Exp 2: Text Preprocessing & Normalization                              |
        |  - Unicode NFC Normalization                                            |
        |  - Whitespace & Punctuation Cleaning (। -> .)                           |
        |  - Word & Sentence Tokenization                                         |
        |  - Stopwords Filtering (Preserving Negations & Quantities)             |
        +-------------------------------------------------------------------------+
                                           |
                                           v
        +-------------------------------------------------------------------------+
        |  Exp 3 & 5: Information Extraction & Syntax Parsing                     |
        |  - Regex Extractor (Quantities, Units, Durations, Temperatures)         |
        |  - Morphological Analyzer (Suffix Decomposition)                        |
        |  - Chunking (NP, VP, PP Groups)                                         |
        |  - Recipe Named Entity Recognition (NER)                                |
        +-------------------------------------------------------------------------+
                                           |
                                           v
        +-------------------------------------------------------------------------+
        |  Exp 6 & 7: Sequence Labeling & Semantics                               |
        |  - Hidden Markov Model (HMM) Parameter Estimation                       |
        |  - Viterbi Dynamic Programming Decoding for POS Tags                    |
        |  - Simplified Lesk Overlap Algorithm for WSD Disambiguation             |
        +-------------------------------------------------------------------------+
                                           |
                                           v
        +-------------------------------------------------------------------------+
        |  Exp 4 & 8: Language Modeling & Analytics                               |
        |  - N-gram Language Modeling (Unigram, Bigram, Trigram Predictions)      |
        |  - Exploratory Data Analysis (EDA) & Devanagari Word Cloud              |
        |  - Structured Recipe Table Generation (CSV / JSON Export)               |
        +-------------------------------------------------------------------------+
        ```
        """
    )

with tab_algo:
    st.subheader("अल्गोरिदम आणि कार्यपद्धती (Algorithms Summary)")
    st.markdown(
        """
        1. **मराठी स्टेमिंग व लेमॅटायझेशन**:
           - इंग्रजी पोर्टर स्टेमरऐवजी मराठीतील विभक्ती प्रत्यय (*-ात, -ाला, -ून, -ताना*) काढून मूळ रूप शोधणारे नियम-आधारित निष्कासन.
        2. **N-ग्राम भाषा मॉडेल (N-gram Model)**:
           - बिगग्राम व ट्रायग्राम संभाव्यता मोजण्यासाठी मॅक्सिमम लाइक्लीहूड एस्टिमेशन (MLE) सोबत **Add-k (Laplace) स्मूथिंग** आणि बॅकऑफ (Backoff) पद्धत.
        3. **Hidden Markov Model (HMM) व व्हिटरबी अल्गोरिदम**:
           - शब्दांच्या जातींचे (POS) टॅगिंग करण्यासाठी $P(S|W)$ चे कमाल मान शोधण्यासाठी व्हिटरबी ट्रेलिस डायनॅमिक प्रोग्रामिंग मॅट्रिक्स.
        4. **सिम्पलीफाईड लेस्क अल्गोरिदम (Lesk WSD)**:
           - संदर्भातील शब्द आणि शब्दकोशातील व्याख्या यांच्यातील शब्द ओव्हरलॅप (Word Overlap Score) मोजून सर्वाधिक जुळणारा अर्थ निवडणे.
        """
    )

with tab_data:
    st.subheader("डेटासेट आणि स्रोत (Dataset Description)")
    st.markdown(
        """
        - **पाककृती संग्रह (data/recipes.json / csv / txt)**: २२ अस्सल महाराष्ट्रीयन पाककृती (उदा. कांदे पोहे, उपमा, पुरणपोळी, मिसळ, तांबडा रस्सा इ.).
        - **घटक शब्दकोश (data/ingredients.json)**: ६० हून अधिक मराठी मसाल्यांचे, भाज्यांचे व डाळींचे वर्गवारीसह संकलन.
        - **पाककृती क्रियापदे (data/cooking_actions.json)**: २० मुख्य पाककृती धातू व त्यांची विविध रूपे.
        - **स्टॉपवर्ड्स सूची (data/stopwords_mr.txt)**: १४८ मराठी स्टॉपवर्ड्स.
        - **POS टॅग कॉर्पस (data/pos_tagged_corpus.csv)**: २० हाताने लेबल केलेली वाक्ये व १५८ टोकन्स.
        - **WSD लेक्सिकॉन (data/wsd_lexicon.json)**: 'तेल', 'तूप', 'पान', 'बारीक', 'रस', 'तास' या अनेकार्थी शब्दांचे सुस्पष्ट अर्थ व संदर्भ.
        """
    )

with tab_limit:
    st.subheader("मर्यादा आणि भविष्यातील व्याप्ती (Limitations & Future Scope)")
    st.markdown(
        """
        - **मर्यादा (Limitations)**:
          - मर्यादित आकाराच्या डेटासेटमुळे काही अप्रचलित किंवा प्रादेशिक बोलीभाषेतील शब्द ओळखण्यात अडचणी येऊ शकतात.
          - रेगेक्स पॅटर्न्स विशिष्ट व्याकरणावर आधारलेले असल्याने जटिल मिश्र वाक्यांमध्ये काही घटक वगळले जाऊ शकतात.
        - **भविष्यातील वाव (Future Scope)**:
          - मोठ्या प्रमाणात क्राउडसोर्स्ड मराठी पाककृती कॉर्पस गोळा करणे.
          - डीप लर्निंग मॉडेल्स (जसे की IndicBERT किंवा Marathi-T5) चा वापर करून एंड-टू-एंड एनईआर आणि डिपेंडन्सी पार्सिंग लागू करणे.
          - रेसिपीवर आधारित पोषणमूल्य (Nutritional Information) गणना करणे.
        """
    )
