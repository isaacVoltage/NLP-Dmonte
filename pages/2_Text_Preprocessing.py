"""
RecipeParse - Experiment 2: Text Preprocessing
Demonstrates Marathi text preprocessing techniques:
- Tokenization (Word & Sentence)
- Stemming & Lemmatization
- Stopword removal (preserving cooking negations)
- Script validation (Unicode block checking)
- Additional normalization (Unicode NFC, punctuation, numerals, recipe steps)
"""
import streamlit as st
import pandas as pd
from src.utils import init_app_page
from src.preprocessing import MarathiPreprocessor

current_theme = init_app_page("Exp 2: Text Preprocessing", "🧹")

st.markdown(
    """
    <div class="main-header">
        <h1>🧹 Experiment 2: Text Preprocessing (मजकूर पूर्वप्रक्रिया)</h1>
        <p>मराठी पाककृती मजकुराचे टोकनायझेशन, स्टॉपवर्ड्स निष्कासन, स्टेमिंग, लेमॅटायझेशन व लिपी प्रमाणीकरण</p>
    </div>
    """,
    unsafe_allow_html=True
)

sample_texts = [
    "एका पातेल्यात दोन चमचे तेल गरम करा. त्यात एक चमचा मोहरी घाला. बारीक चिरलेला कांदा घालून पाच मिनिटे परता!!",
    "कढईत ५० ग्रॅम साजूक तूप मंद आचेवर तापवा. भाजलेला रवा हळूहळू घालत ढवळा जेणेकरून गाठी होणार नाहीत.",
    "दोन कप जाड पोहे स्वच्छ धुवून घ्या. चवीनुसार १ चमचा मीठ घाला."
]

st.subheader("मजकूर प्रविष्ट करा (Enter Marathi Text)")
selected_sample = st.selectbox("नमुना मजकूर निवडा (Select Sample):", ["-- सानुकूल मजकूर (Custom) --"] + sample_texts)
user_text = st.text_area(
    "प्रक्रियेसाठी मजकूर (Text to Preprocess):",
    value=selected_sample if selected_sample != "-- सानुकूल मजकूर (Custom) --" else sample_texts[0],
    height=120
)

prep = MarathiPreprocessor()

if user_text:
    res = prep.run_full_pipeline(user_text)

    if "error" in res:
        st.error(f"❌ {res['error']}")
    else:
        # Script Validation Box
        s_val = res["script_validation"]
        val_color = "#10b981" if s_val["is_devanagari"] else "#ef4444"
        st.markdown(
            f"""
            <div style="padding: 12px 18px; border-radius: 10px; background: {val_color}18; border: 1px solid {val_color}; margin-bottom: 20px;">
                <strong>लिपी प्रमाणीकरण निकाल (Script Validation):</strong> {s_val['status']}<br>
                <span>देवनागरी वर्ण: <b>{s_val['devanagari_chars']} / {s_val['total_chars']} ({s_val['percentage']}%)</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Tabs for Individual Preprocessing Steps
        t1, t2, t3, t4, t5, t6 = st.tabs([
            "🔤 टोकनायझेशन (Tokenization)",
            "🚫 स्टॉपवर्ड्स (Stopwords Removal)",
            "🌱 स्टेमिंग व लेमॅटायझेशन (Stem & Lemma)",
            "🔢 अंक नॉर्मलायझेशन (Numerals)",
            "🧹 अतिरिक्त पद्धती (Additional Normalization)",
            "📖 भाषिक मर्यादा (Linguistic Limitations)"
        ])

        with t1:
            st.subheader("शब्द व वाक्य विभाजन (Tokenization)")
            c_tok1, c_tok2 = st.columns(2)
            with c_tok1:
                st.markdown(f"**वाक्य विभाजन ({res['total_sentences']} वाक्ये):**")
                for i, s in enumerate(res["sentences"], 1):
                    st.markdown(f"{i}. {s}")
            with c_tok2:
                st.markdown(f"**शब्द विभाजन ({res['total_raw_tokens']} टोकन्स):**")
                tokens_html = " ".join([f"<span class='info-pill'>{t}</span>" for t in res["raw_tokens"]])
                st.markdown(tokens_html, unsafe_allow_html=True)

        with t2:
            st.subheader("स्टॉपवर्ड्स निष्कासन (Stopword Removal)")
            st.markdown(
                """
                > [!NOTE]
                > येथे मराठी स्टॉपवर्ड्स निष्कासित करताना महत्त्वाचे पाककृती घटक, परिमाणे व नकारात्मक शब्द (उदा. नाही, नका, नको) जपले जातात.
                """
            )
            col_sw1, col_sw2 = st.columns(2)
            with col_sw1:
                st.markdown(f"**काढून टाकलेले स्टॉपवर्ड्स ({res['total_stopwords_removed']}):**")
                sw_html = " ".join([f"<span style='display:inline-block; background: #ef444422; border:1px solid #ef4444; color:#ef4444; padding:3px 8px; border-radius:12px; margin:2px;'>{w}</span>" for w in res["stopwords_removed"]])
                st.markdown(sw_html if res["stopwords_removed"] else "कोणतेही स्टॉपवर्ड आढळले नाही.", unsafe_allow_html=True)

            with col_sw2:
                st.markdown(f"**गाळणीनंतर शिल्लक राहिलेले अर्थपूर्ण शब्द ({res['total_filtered_tokens']}):**")
                fil_html = " ".join([f"<span class='info-pill'>{w}</span>" for w in res["filtered_tokens"]])
                st.markdown(fil_html, unsafe_allow_html=True)

        with t3:
            st.subheader("स्टेमिंग वि. लेमॅटायझेशन तुलना (Stemming vs Lemmatization)")
            st.markdown(
                """
                मराठीतील विभक्ती प्रत्यय (उदा. *कांद्याला -> कांदा*, *तेलात -> तेल*, *चिरलेला -> चिरणे*) काढून मूळ रूप मिळवणे:
                """
            )
            pairs = []
            for (w1, s), (w2, l) in zip(res["stemmed_tokens"], res["lemmatized_tokens"]):
                pairs.append({
                    "मूळ टोकन (Original)": w1,
                    "स्टेम रूप (Stemmed)": s,
                    "लेम्मा रूप (Lemmatized)": l,
                    "बदल झाला?": "होय" if (w1 != s or w1 != l) else "नाही"
                })
            st.dataframe(pd.DataFrame(pairs), use_container_width=True, hide_index=True)

        with t4:
            st.subheader("देवनागरी व अरेबिक अंक रूपांतरण (Numeral Normalization)")
            col_n1, col_n2 = st.columns(2)
            with col_n1:
                st.markdown("**अरेबिक अंकांमध्ये रूपांतरित मजकूर (0-9):**")
                st.code(res["arabic_numerals_normalized"], language="text")
            with col_n2:
                st.markdown("**देवनागरी अंकांमध्ये रूपांतरित मजकूर (०-९):**")
                st.code(res["devanagari_numerals_normalized"], language="text")

        with t5:
            st.subheader("इतर आवश्यक पूर्वप्रक्रिया पद्धती (Additional Normalization)")
            st.markdown(f"**1. युनिकोड NFC नॉर्मलायझेशन:** `{res['unicode_normalized']}`")
            st.markdown(f"**2. पुनरावृत्त विरामचिन्हे शुध्दीकरण (उदा. !! -> !):** `{res['repeated_punctuation_cleaned']}`")
            st.markdown(f"**3. अतिरिक्त पांढऱ्या जागा (Whitespace) निष्कासन:** `{res['whitespace_cleaned']}`")
            st.markdown("**4. पाककृती टप्पे विभाजन (Normalized Recipe Steps):**")
            for idx, step in enumerate(res["recipe_steps"], 1):
                st.markdown(f"- **पायरी {idx}:** {step}")

        with t6:
            st.subheader("भाषिक मर्यादा व स्पष्टीकरण (Linguistic Limitations)")
            st.warning(
                f"""
                **शैक्षणिक टीप:** {res['limitation_note']}
                
                - इंग्रजीतील पोर्टर स्टेमर (Porter Stemmer) मराठीसाठी निरुपयोगी आहे कारण मराठी भाषा 'विभक्तीप्रधान व संलक्षी' (fusional/agglutinative) आहे.
                - देवनागरीतील संधि, समास आणि अंत्य स्वरबदल (morpho-phonemic changes) केवळ इंग्रजी नियमांवर चालत नाहीत.
                - म्हणून RecipeParse मध्ये मराठी भाषेसाठी अनुकूलित 'नियम-आधारित प्रत्यय निष्कासन' आणि 'डोमेन-विशिष्ट लेक्झिकॉन मॅपिंग' वापरले गेले आहे.
                """
            )
