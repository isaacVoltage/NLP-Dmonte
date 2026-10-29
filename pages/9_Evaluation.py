"""
RecipeParse - Evaluation Page
Presents empirical quantitative evaluation results calculated live from the benchmark dataset.
"""
import streamlit as st
import pandas as pd
from src.utils import init_app_page
from src.evaluation import RecipeParseEvaluator

current_theme = init_app_page("Evaluation - मूल्यांकन", "📊")

st.markdown(
    """
    <div class="main-header">
        <h1>📊 Quantitative Evaluation & Benchmarking (मूल्यांकन अहवाल)</h1>
        <p>मराठी पाककृती माहिती निष्कासन प्रणालीची प्रत्यक्ष मोजलेली अचूकता, प्रिसिजन, रिकॉल आणि F1-स्कोअर</p>
    </div>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def run_evaluation():
    evaluator = RecipeParseEvaluator()
    return evaluator.run_all_evaluations()

with st.spinner("मूल्यांकन मेट्रिक्स मोजत आहे..."):
    eval_results = run_evaluation()

ner_res = eval_results["ner"]
pos_res = eval_results["pos"]
ngram_res = eval_results["ngram"]

# Metric Cards
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>NER Overall F1-Score</h3>
            <p style="color: #10b981;">{ner_res['overall_f1']}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>NER Overall Precision</h3>
            <p>{ner_res['overall_precision']}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>POS Tagging Accuracy</h3>
            <p style="color: #3b82f6;">{pos_res['accuracy']}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>Next-Word Top-3 Hit</h3>
            <p style="color: #f59e0b;">{ngram_res['top3_accuracy']}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

tab_ner, tab_pos, tab_ngram, tab_notes = st.tabs([
    "🏷️ NER मूल्यमापन (Precision / Recall / F1)",
    "🧬 POS टॅगिंग अचूकता (Token Accuracy)",
    "🔮 N-Gram भाकीत अचूकता (Next-Word Hits)",
    "📝 मूल्यमापन पद्धत व मर्यादा (Evaluation Methodology)"
])

with tab_ner:
    st.subheader("Named Entity Recognition (NER) वर्गनिहाय मेट्रिक्स")
    st.markdown("प्रत्येक एंटिटी प्रकारासाठी मोजलेले प्रिसिजन, रिकॉल व F1-स्कोअर:")
    st.dataframe(ner_res["per_class_df"], use_container_width=True, hide_index=True)

with tab_pos:
    st.subheader("HMM / Viterbi POS टॅगिंग अचूकता निकाल")
    st.markdown(f"**एकूण तपासलेले चाचणी शब्द:** {pos_res['total_tokens']} | **अचूक जुळलेले:** {pos_res['correct_tokens']}")
    st.dataframe(pos_res["results_df"], use_container_width=True, hide_index=True)

with tab_ngram:
    st.subheader("N-Gram लँग्वेज मॉडेल चाचणी निकाल (Hit Rates)")
    st.markdown(f"- **Top-1 अचूकता:** {ngram_res['top1_accuracy']}%")
    st.markdown(f"- **Top-3 अचूकता:** {ngram_res['top3_accuracy']}%")
    st.markdown(f"- **Top-5 अचूकता:** {ngram_res['top5_accuracy']}%")
    st.dataframe(ngram_res["results_df"], use_container_width=True, hide_index=True)

with tab_notes:
    st.subheader("मूल्यांकन पद्धती व शैक्षणिक प्रामाणिकता (Academic Integrity)")
    st.info(
        """
        - हे सर्व निकाल सिस्टीममधील चाचणी संचावर (Test Set) प्रत्यक्ष चालवून मोजले गेले आहेत. कोणतेही बनावट स्कोअर लिहिलेले नाहीत.
        - मर्यादित शैक्षणिक डेटासेटमुळे काही अज्ञात शब्दांवर रिकॉल कमी होऊ शकतो, ज्याचे स्पष्टीकरण 'About' आणि 'Limitations' विभागांमध्ये दिले आहे.
        """
    )
