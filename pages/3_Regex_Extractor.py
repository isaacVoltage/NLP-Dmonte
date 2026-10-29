"""
RecipeParse - Experiment 3: Morphological Analysis and Regex Extractor
Extracts quantities, units, durations, temperatures, actions, and ingredients using regular expressions.
"""
import streamlit as st
import pandas as pd
from src.utils import init_app_page
from src.regex_extractor import MarathiRegexExtractor
from src.morphological_analysis import MarathiMorphologicalAnalyzer

current_theme = init_app_page("Exp 3: Morphology & Regex Extractor", "🔍")

st.markdown(
    """
    <div class="main-header">
        <h1>🔍 Experiment 3: Morphological Analysis & Regex Extractor</h1>
        <p>मराठी पाककृती मजकुरातून रेग्युलर एक्स्प्रेशन्सद्वारे प्रमाण, एकके, वेळ, तापमान, कृती व घटकांचे अचूक निष्कासन</p>
    </div>
    """,
    unsafe_allow_html=True
)

default_text = (
    "एका कढईत दोन चमचे तेल मध्यम आचेवर गरम करा. त्यात एक चमचा मोहरी आणि अर्धा चमचा जिरे घाला. "
    "बारीक चिरलेला कांदा घालून पाच मिनिटे परता. नंतर ५० ग्रॅम रवा मंद आचेवर दहा मिनिटे भाजा."
)

st.subheader("मजकूर प्रविष्ट करा (Enter Marathi Recipe Text)")
input_text = st.text_area("पाककृती वाक्य (Sentence/Instruction):", value=default_text, height=110)

col_b1, col_b2 = st.columns([1, 4])
with col_b1:
    extract_btn = st.button("🔎 पॅटर्न्स शोधा (Extract Patterns)", type="primary", use_container_width=True)

if extract_btn or input_text:
    extractor = MarathiRegexExtractor()
    morph = MarathiMorphologicalAnalyzer()

    results = extractor.extract_all(input_text)

    # Top KPI count cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("परिमाणे व एकके (Q & U)", len(results["quantities_and_units"]))
    with c2:
        st.metric("स्वयंपाक वेळ (Durations)", len(results["durations"]))
    with c3:
        st.metric("तापमान (Temperatures)", len(results["temperatures"]))
    with c4:
        st.metric("क्रियापदे (Actions)", len(results["actions"]))
    with c5:
        st.metric("घटक (Ingredients)", len(results["ingredients"]))

    st.markdown("<br>", unsafe_allow_html=True)

    tab_table, tab_morph, tab_cat, tab_disc = st.tabs([
        "📋 एकत्रित निकाल तक्ता (Combined Table)",
        "🔬 आकारशास्त्रीय विश्लेषण (Morphological Analysis)",
        "📂 वर्गवारीनुसार निष्कासन (Category-wise Details)",
        "⚠️ भाषिक मर्यादा (Linguistic Limitations)"
    ])

    with tab_table:
        st.subheader("निष्कासित घटकांचा एकत्रित तक्ता (Combined Extracted Table)")
        comb = results["combined_table"]
        if comb:
            df_comb = pd.DataFrame(comb)
            st.dataframe(df_comb, use_container_width=True, hide_index=True)
            csv_bytes = df_comb.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 निकाल CSV डाउनलोड करा",
                data=csv_bytes,
                file_name="regex_extraction_results.csv",
                mime="text/csv"
            )
        else:
            st.info("या वाक्यात थेट घटक-कृती जोड आढळली नाही.")

    with tab_morph:
        st.subheader("प्रत्यय व धातूसाधित रूप विश्लेषण (Morphological Affix Decomposition)")
        words = input_text.replace(".", " ").replace(",", " ").split()
        unique_words = sorted(list(set(w.strip() for w in words if len(w.strip()) > 1)))
        morph_results = morph.analyze_sentence(unique_words)

        df_morph = pd.DataFrame(morph_results)
        st.dataframe(df_morph, use_container_width=True, hide_index=True)

    with tab_cat:
        st.subheader("स्वतंत्र पॅटर्न निकाल (Pattern Breakdowns)")
        
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.markdown("#### A. प्रमाण आणि एकके (Quantities & Units)")
            if results["quantities_and_units"]:
                st.dataframe(pd.DataFrame(results["quantities_and_units"])[["original_text", "quantity", "unit"]], use_container_width=True, hide_index=True)
            else:
                st.write("कोणतेही परिमाण आढळले नाही.")

            st.markdown("#### B. स्वयंपाक कालावधी (Cooking Durations)")
            if results["durations"]:
                st.dataframe(pd.DataFrame(results["durations"])[["original_text", "quantity", "unit"]], use_container_width=True, hide_index=True)
            else:
                st.write("कोणताही कालावधी आढळला नाही.")

        with c_p2:
            st.markdown("#### C. उष्णता पातळी व तापमान (Temperatures)")
            if results["temperatures"]:
                st.dataframe(pd.DataFrame(results["temperatures"])[["original_text", "value"]], use_container_width=True, hide_index=True)
            else:
                st.write("कोणतेही तापमान आढळले नाही.")

            st.markdown("#### D. स्वयंपाक कृती व भांडी (Actions & Utensils)")
            acts = [a["value"] for a in results["actions"]]
            uts = [u["value"] for u in results["utensils"]]
            st.markdown(f"**कृती (Actions):** {', '.join(acts) if acts else 'नाही'}")
            st.markdown(f"**भांडी (Utensils):** {', '.join(uts) if uts else 'नाही'}")

    with tab_disc:
        st.subheader("रेग्युलर एक्स्प्रेशन्सच्या भाषिक मर्यादा (Regex Limitations)")
        st.warning(
            f"""
            **अल्गोरिदम मर्यादा:** {results['disclaimer']}
            
            1. **संधि व विभक्ती प्रत्यय**: 'पातेल्यात' मध्ये 'पातेले' हे मूळ रूप ओळखण्यासाठी केवळ रेगेक्स अपुरे पडतात, तिथे मॉर्फोलॉजिकल लेमॅटायझेशन आवश्यक असते.
            2. **सिंटॅक्स व संदर्भ**: रेगेक्स वाक्यातील व्याकरणिक संबंध (जसे कर्ता-कर्म-क्रियापद) समजू शकत नाही.
            3. म्हणून RecipeParse मध्ये रेगेक्स सोबत HMM POS टॅगिंग आणि व्हिटरबी डिकोडिंगची जोड दिली आहे.
            """
        )
