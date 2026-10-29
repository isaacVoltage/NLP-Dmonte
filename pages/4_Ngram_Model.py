"""
RecipeParse - Experiment 4: N-Gram Language Model
Implements unigram, bigram, and trigram models for frequency estimation and next-word prediction.
"""
import streamlit as st
import pandas as pd
from src.utils import init_app_page
from src.ngram_model import MarathiNGramModel

current_theme = init_app_page("Exp 4: N-Gram Language Model", "📊")

st.markdown(
    """
    <div class="main-header">
        <h1>📊 Experiment 4: N-Gram Language Modeling</h1>
        <p>मराठी पाककृती मजकुरावरून युनिग्रॅम, बायग्रॅम, ट्रायग्रॅम वारंवारता गणना आणि पुढील शब्दाचे संभाव्यता भाकीत (Next-word Prediction)</p>
    </div>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def get_trained_ngram():
    model = MarathiNGramModel(k_smoothing=0.5)
    model.train()
    return model

model = get_trained_ngram()

# Interactive Prediction Playground
st.subheader("🔮 पुढील शब्दाचा अंदाज (Next-Word Prediction Playground)")
sample_prompts = ["दोन", "दोन चमचे", "एका कढईत", "बारीक चिरलेला", "पाच", "मंद आचेवर", "साजूक"]
col_in1, col_in2 = st.columns([3, 1])
with col_in1:
    user_phrase = st.selectbox("नमुना शब्द निवडा किंवा खाली टाईप करा (Select Phrase):", sample_prompts)
with col_in2:
    top_k = st.slider("भाकितांची संख्या (Top K)", min_value=1, max_value=10, value=5)

custom_phrase = st.text_input("किंवा स्वतःचा संदर्भ शब्द टाईप करा (Or type custom context):", value=user_phrase)

if custom_phrase:
    comp = model.compare_predictions(custom_phrase, top_k=top_k)

    if "error" in comp:
        st.error(comp["error"])
    else:
        col_bg, col_tg = st.columns(2)
        with col_bg:
            st.markdown("#### 🔹 बायग्रॅम भाकीत (Bigram P(w | w_{i-1}))")
            bg_df = pd.DataFrame(comp["bigram_predictions"])
            if not bg_df.empty:
                st.dataframe(bg_df[["predicted_word", "frequency", "probability", "model"]], use_container_width=True, hide_index=True)
            else:
                st.info("बायग्रॅम संदर्भ सापडला नाही.")

        with col_tg:
            st.markdown("#### 🔺 ट्रायग्रॅम भाकीत (Trigram P(w | w_{i-2}, w_{i-1}))")
            tg_df = pd.DataFrame(comp["trigram_predictions"])
            if not tg_df.empty:
                st.dataframe(tg_df[["predicted_word", "frequency", "probability", "model"]], use_container_width=True, hide_index=True)
            else:
                st.info("ट्रायग्रॅम संदर्भ सापडला नाही.")

        st.info(f"💡 **शैक्षणिक अस्वीकरण (Academic Disclaimer):** {comp['academic_disclaimer']}")

st.markdown("---")

# Frequency Tables
st.subheader("📈 N-Gram वारंवारता विश्लेषण (Corpus Frequency Distributions)")
tab_uni, tab_bi, tab_tri = st.tabs(["1️⃣ युनिग्रॅम (Unigrams)", "2️⃣ बायग्रॅम (Bigrams)", "3️⃣ ट्रायग्रॅम (Trigrams)"])

with tab_uni:
    st.markdown("सर्वात जास्त वेळा आलेले सुटे शब्द (Top Unigrams):")
    uni_data = model.get_unigram_frequency(20)
    df_u = pd.DataFrame(uni_data, columns=["शब्द (Token)", "वारंवारता (Frequency)"])
    st.dataframe(df_u, use_container_width=True, hide_index=True)

with tab_bi:
    st.markdown("सर्वात जास्त वेळा आलेले सलग दोन शब्द (Top Bigrams):")
    bi_data = model.get_bigram_frequency(20)
    df_b = pd.DataFrame(bi_data, columns=["बायग्रॅम जोडी (Bigram)", "वारंवारता (Frequency)"])
    st.dataframe(df_b, use_container_width=True, hide_index=True)

with tab_tri:
    st.markdown("सर्वात जास्त वेळा आलेले सलग तीन शब्द (Top Trigrams):")
    tri_data = model.get_trigram_frequency(20)
    df_t = pd.DataFrame(tri_data, columns=["ट्रायग्रॅम समूह (Trigram)", "वारंवारता (Frequency)"])
    st.dataframe(df_t, use_container_width=True, hide_index=True)
