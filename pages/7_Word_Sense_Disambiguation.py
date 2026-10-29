"""
RecipeParse - Experiment 7: Word Sense Disambiguation (WSD) using Lesk Algorithm
Resolves the intended sense of polysemous Marathi words from sentence context.
"""
import streamlit as st
import pandas as pd
from src.utils import init_app_page
from src.lesk_wsd import MarathiLeskWSD

current_theme = init_app_page("Exp 7: Lesk WSD", "💡")

st.markdown(
    """
    <div class="main-header">
        <h1>💡 Experiment 7: Word Sense Disambiguation (Lesk Algorithm)</h1>
        <p>मराठी भाषेतील अनेकार्थी शब्दांचा संदर्भावरून योग्य अर्थ ठरवण्यासाठी सिम्पलीफाईड लेस्क ओव्हरलॅप अल्गोरिदम</p>
    </div>
    """,
    unsafe_allow_html=True
)

wsd = MarathiLeskWSD()
available_words = wsd.get_supported_target_words()

preset_contexts = {
    "तेल": [
        "कढईत दोन चमचे तेल गरम करून मोहरीची फोडणी द्या.",
        "देवापुढे समईत तेल ओतून दिवा प्रज्वलित करा."
    ],
    "तूप": [
        "गरम वरण भातावर एक चमचा साजूक तूप वाढा.",
        "यज्ञात पुरोहितांनी तुपाची आहुती दिली."
    ],
    "पान": [
        "तेलात चार कढीपत्ता पाने तडतडू द्या.",
        "पुस्तकाचे दहावे पान उघडून कविता वाचा."
    ],
    "बारीक": [
        "कांदा आणि कोथिंबीर बारीक चिरून घ्या.",
        "आजारी पडल्यामुळे तो खूप बारीक झाला आहे."
    ],
    "रस": [
        "उन्हाळ्यात आंब्याचा गोड रस आणि पुरी खातात.",
        "या ऐतिहासिक कवितेमध्ये वीर रस ओतप्रोत भरलेला आहे."
    ],
    "तास": [
        "साबुदाणा चार तास स्वच्छ पाण्यात भिजत ठेवा.",
        "शाळेत गणिताचा पहिला तास संपला."
    ]
}

st.subheader("1️⃣ अनेकार्थी लक्ष्य शब्द निवडा (Select Ambiguous Target Word)")
col_w1, col_w2 = st.columns([1, 2])
with col_w1:
    target_word = st.selectbox("लक्ष्य शब्द (Target Word):", available_words)

with col_w2:
    context_options = preset_contexts.get(target_word, ["वाक्य प्रविष्ट करा"])
    chosen_ctx = st.selectbox("नमुना संदर्भ वाक्य निवडा (Sample Contexts):", context_options)

st.subheader("2️⃣ संदर्भ वाक्य तपासा किंवा संपादित करा (Sentence Context)")
sentence_input = st.text_input("संदर्भ वाक्य (Sentence):", value=chosen_ctx)

col_run, _ = st.columns([1, 4])
with col_run:
    run_wsd_btn = st.button("✨ अर्थ शोधा (Disambiguate Word)", type="primary", use_container_width=True)

if run_wsd_btn or sentence_input:
    res = wsd.disambiguate(sentence_input, target_word)

    if "error" in res:
        st.error(res["error"])
    else:
        # Winning Sense Banner
        st.markdown(
            f"""
            <div style="padding: 18px; border-radius: 12px; background: rgba(16, 185, 129, 0.15); border: 2px solid #10b981; margin: 15px 0;">
                <span style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: #10b981; font-weight: bold;">
                    🎯 निवडलेला अर्थ (Disambiguated Sense)
                </span>
                <h3 style="margin: 5px 0; color: #f1f5f9;">{res['selected_sense_id']}</h3>
                <p style="font-size: 1.15rem; margin: 5px 0; color: #a7f3d0;"><b>अर्थ (Gloss):</b> {res['selected_gloss']}</p>
                <div style="font-size: 0.95rem; margin-top: 8px;">
                    <b>जुळलेले शब्द (Overlapping Words):</b> 
                    <span style="background: #10b981; color: white; padding: 2px 8px; border-radius: 8px; font-weight: bold;">
                        {', '.join(res['winning_overlapping_words']) if res['winning_overlapping_words'] else 'None (Default Sense Selected)'}
                    </span>
                    &nbsp;&nbsp;<b>एकूण ओव्हरलॅप स्कोअर:</b> {res['best_overlap_count']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("3️⃣ सर्व संभाव्य अर्थांची तुलना (Candidate Senses Comparison)")
        senses_table = []
        for s in res["candidate_senses"]:
            senses_table.append({
                "अर्थ ओळख (Sense ID)": s["sense_id"],
                "व्याख्या (Definition/Gloss)": s["gloss"],
                "ओव्हरलॅप शब्द संख्या (Score)": s["overlap_count"],
                "जुळलेले शब्द (Overlapping Words)": ", ".join(s["overlapping_words"]) if s["overlapping_words"] else "-",
                "निवडला गेला?": "✅ होय (Winner)" if s["sense_id"] == res["selected_sense_id"] else "❌"
            })
        st.dataframe(pd.DataFrame(senses_table), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("4️⃣ अल्गोरिदम पायऱ्या व कार्यपद्धती (Lesk Algorithm Trace)")
        st.markdown(f"**वाक्यातील गाळणी केलेले अर्थपूर्ण शब्द (Context Content Words):**")
        st.markdown(" ".join([f"<span class='info-pill'>{w}</span>" for w in res["context_content_words"]]), unsafe_allow_html=True)
        st.markdown(f"**निर्णय स्पष्टीकरण (Resolution Logic):** {res['explanation']}")
        st.markdown(f"**डेटा स्रोत (Lexicon Provenance):** {res['lexicon_provenance']}")

        st.markdown("---")
        st.warning(
            """
            **अल्गोरिदम मर्यादा (Limitations of Simplified Lesk):**
            1. **शब्दसंग्रह अवलंबित्व**: जर संदर्भ वाक्यातील शब्दांचे थेट जुळणारे समानार्थी शब्द लेक्सिकॉन व्याख्येत नसतील, तर ओव्हरलॅप शून्य येऊ शकतो.
            2. **सिमेंटिक समानता**: लेस्क अल्गोरिदम केवळ शब्द-अक्षरांची तंतोतंत जुळवणी पाहतो; तो 'कढई' आणि 'तवा' हे दोन्ही भांडी आहेत हे आपोआप समजू शकत नाही. यासाठी वर्ड एम्बेडिंग्ज किंवा विस्तृत WordNet आवश्यक असते.
            """
        )
