"""
RecipeParse: A Marathi Recipe Information Extraction System Using Natural Language Processing (NLP)
Home Page / Entry Point
"""
import streamlit as st
import pandas as pd
from src.utils import init_app_page, load_recipes, load_ingredients

current_theme = init_app_page("Home - मुख्यपृष्ठ", "🍲")

# Header Banner
st.markdown(
    """
    <div class="main-header">
        <h1>🍲 RecipeParse: मराठी पाककृती माहिती निष्कासन प्रणाली</h1>
        <p>A Marathi Recipe Information Extraction System Using Natural Language Processing (NLP)</p>
    </div>
    """,
    unsafe_allow_html=True
)

recipes = load_recipes()
ingredients = load_ingredients()

# Metric summary cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>पाककृती संग्रह (Recipes)</h3>
            <p>{len(recipes)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>घटक शब्दकोश (Ingredients)</h3>
            <p>{len(ingredients)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        """
        <div class="metric-card">
            <h3>अभ्यासक्रम प्रात्यक्षिके</h3>
            <p>2 ते 8</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col4:
    st.markdown(
        """
        <div class="metric-card">
            <h3>सिस्टम स्थिती</h3>
            <p style="color: #10b981;">100% Validated</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Tabs for structured academic presentation
tab1, tab2, tab3, tab4 = st.tabs(["🎯 प्रकल्प उद्दिष्टे (Aim & Objectives)", "📚 समाविष्ट प्रात्यक्षिके (Experiments 2-8)", "🛠️ वापरलेले तंत्रज्ञान (Tech Stack)", "🚀 प्रारंभ कसा करावा (Quick Start)"])

with tab1:
    st.subheader("प्रकल्प ध्येय (Project Aim)")
    st.markdown(
        """
        **मराठी पाककृती माहिती निष्कासन प्रणाली (RecipeParse)** चे मुख्य ध्येय देवनागरी लिपीतील मराठी पाककृती मजकुराचे 
        नैसर्गिक भाषा प्रक्रियेच्या (Natural Language Processing - NLP) विविध तंत्रांद्वारे विश्लेषण करणे, त्यातून साहित्य (Ingredients), 
        प्रमाण (Quantities), एकके (Units), स्वयंपाक कृती (Actions), लागणारा वेळ (Durations), आणि तापमान (Temperatures) यांचे 
        संरचित निष्कासन (Structured Extraction) करणे आणि शैक्षणिक प्रात्यक्षिकांचे प्रामाणिक सादरीकरण करणे हे आहे.
        """
    )
    st.subheader("प्रमुख उद्दिष्टे (Key Objectives)")
    st.markdown(
        """
        1. **मराठी मजकूर पूर्वप्रक्रिया**: युनिकोड नॉर्मलायझेशन, विरामचिन्हे शुध्दीकरण, अंक रूपांतरण आणि स्टॉपवर्ड्स निष्कासन.
        2. **मॉर्फोलॉजिकल विश्लेषण व रेग्युलर एक्स्प्रेशन्स**: विभक्ती प्रत्यय, स्वयंपाक क्रियापदे व घटकांच्या पॅटर्न्सची ओळख.
        3. **N-Gram लँग्वेज मॉडेल**: युनिग्रॅम, बायग्रॅम, ट्रायग्रॅम वारंवारता मोजणे आणि पुढील शब्दाचे भाकीत (Next-word Prediction).
        4. **चंकिंग व नामनिर्देशित घटक ओळख (NER)**: नाम समूह (NP), क्रियापद समूह (VP), शब्दयोगी समूह (PP) व स्वयंपाक एंटिटीजची ओळख.
        5. **HMM व व्हिटरबी अल्गोरिदम**: Hidden Markov Model द्वारे शब्दांच्या जातींचे (POS Tagging) अचूक डिकोडिंग.
        6. **लेस्क अल्गोरिदम (WSD)**: अनेक अर्थ असणाऱ्या मराठी शब्दांचा संदर्भावरून अचूक अर्थ शोधणे.
        7. **अन्वेषणात्मक डेटा विश्लेषण (EDA)**: मराठी वर्डक्लाउड, वारंवारता वितरण तक्ते आणि आलेखांद्वारे सांख्यिकी विश्लेषण.
        """
    )

with tab2:
    st.subheader("महाविद्यालयीन प्रात्यक्षिके व मॉड्युल मॅपिंग (Experiments Mapping)")
    exp_data = [
        {"क्र.": "Exp 2", "प्रात्यक्षिक नाव": "Text Preprocessing (मजकूर पूर्वप्रक्रिया)", "मॉड्युल": "src/preprocessing.py", "पेज": "2_Text_Preprocessing"},
        {"क्र.": "Exp 3", "प्रात्यक्षिक नाव": "Morphology & Regular Expressions (आकारशास्त्र व रेगेक्स)", "मॉड्युल": "src/regex_extractor.py", "पेज": "3_Regex_Extractor"},
        {"क्र.": "Exp 4", "प्रात्यक्षिक नाव": "N-Gram Language Modeling (N-ग्राम भाषा मॉडेल)", "मॉड्युल": "src/ngram_model.py", "पेज": "4_Ngram_Model"},
        {"क्र.": "Exp 5", "प्रात्यक्षिक नाव": "Chunking and Named Entity Recognition (चंकिंग व NER)", "मॉड्युल": "src/recipe_ner.py & chunking.py", "पेज": "5_Chunking_and_NER"},
        {"क्र.": "Exp 6", "प्रात्यक्षिक नाव": "POS Tagging using HMM & Viterbi (व्हिटरबी POS टॅगिंग)", "मॉड्युल": "src/hmm_pos_tagger.py & viterbi.py", "पेज": "6_HMM_POS_Tagging"},
        {"क्र.": "Exp 7", "प्रात्यक्षिक नाव": "Word Sense Disambiguation using Lesk (लेस्क WSD)", "मॉड्युल": "src/lesk_wsd.py", "पेज": "7_Word_Sense_Disambiguation"},
        {"क्र.": "Exp 8", "प्रात्यक्षिक नाव": "Exploratory Data Analysis & Word Cloud (EDA डॅशबोर्ड)", "मॉड्युल": "src/text_analytics.py", "पेज": "8_EDA_Dashboard"},
        {"क्र.": "Integrated", "प्रात्यक्षिक नाव": "Integrated Marathi Recipe Analyzer (एकात्मिक विश्लेषक)", "मॉड्युल": "src/recipe_extractor.py", "पेज": "1_Recipe_Analyzer"}
    ]
    st.dataframe(pd.DataFrame(exp_data), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("वापरलेले तंत्रज्ञान आणि लायब्ररी (Tech Stack)")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            """
            - **प्रोग्रामिंग भाषा**: Python 3.12 (Native Unicode Devanagari handling)
            - **वेब इंटरफेस**: Streamlit (Responsive Dark/Light mode theme)
            - **डेटा स्ट्रक्चर्स व विश्लेषण**: Pandas, NumPy
            - **अल्गोरिदम अंमलबजावणी**:
              - Viterbi Dynamic Programming (Custom from scratch)
              - Hidden Markov Model (MLE with Laplace Smoothing)
              - Simplified Lesk Overlap Algorithm
              - N-gram Language Model with Add-k Smoothing
            """
        )
    with col_t2:
        st.markdown(
            """
            - **व्हिज्युअलायझेशन**: Matplotlib, Seaborn, WordCloud (with Windows Nirmala Devanagari Font)
            - **नियमित अभिव्यक्ती (Regex)**: Indic Lookaround Boundaries
            - **स्वयंचलित चाचणी**: Pytest (100% passing test suite across all modules)
            - **डेटा फॉरमॅट्स**: JSON, CSV, TXT (UTF-8 encoding)
            """
        )

with tab4:
    st.subheader("सिस्टम वापरण्यासाठी मार्गदर्शक (How to Use)")
    st.markdown(
        """
        1. **डाव्या बाजूच्या साइडबारमधून (Sidebar)** हवे ते प्रात्यक्षिक अथवा **1. Recipe Analyzer** निवडा.
        2. **Dark / Light Mode**: साइडबारमधील **🎨 थीम निवडा** पर्यायावरून एका क्लिकवर डार्क किंवा लाईट मोड बदला.
        3. **नमुना पाककृती तपासा**: प्रत्येक पानावर रेडीमेड नमुना उदाहरणे दिलेली आहेत, ज्यावर क्लिक करून लगेच प्रक्रिया पाहता येते.
        4. **डाउनलोड पर्याय**: विश्लेषणाचे तक्ते थेट **CSV किंवा JSON** स्वरूपात डाउनलोड करता येतात.
        """
    )

st.markdown("---")
st.info("💡 **टीप**: वरील साइडबार मेनूचा वापर करून आपण प्रत्येक प्रात्यक्षिकाची स्वतंत्र कार्यपद्धती आणि एकत्रित रेसिपी विश्लेषक तपासू शकता.")
