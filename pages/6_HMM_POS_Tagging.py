"""
RecipeParse - Experiment 6: POS Tagging Using HMM and Viterbi Algorithm
Implements Part-of-Speech tagging from scratch using Hidden Markov Models and Viterbi Trellis decoding.
"""
import streamlit as st
import pandas as pd
from src.utils import init_app_page
from src.hmm_pos_tagger import MarathiHMMPosTagger

current_theme = init_app_page("Exp 6: HMM POS Tagging", "🧬")

st.markdown(
    """
    <div class="main-header">
        <h1>🧬 Experiment 6: POS Tagging Using HMM & Viterbi Algorithm</h1>
        <p>मराठी पाककृती मजकुरासाठी Hidden Markov Model आणि व्हिटरबी डायनॅमिक प्रोग्रामिंग अल्गोरिदमद्वारे शब्दांच्या जातींचे डिकोडिंग</p>
    </div>
    """,
    unsafe_allow_html=True
)

sample_sentences = [
    "एका पातेल्यात दोन चमचे तेल गरम करा .",
    "त्यात एक चमचा मोहरी घाला .",
    "बारीक चिरलेला कांदा घालून पाच मिनिटे परता .",
    "दोन कप साबूदाणा चार तास भिजवा .",
    "गरम तेलात भजी सोनेरी होईपर्यंत कुरकुरीत तळा ."
]

st.subheader("वाक्य प्रविष्ट करा (Enter Marathi Sentence for POS Tagging)")
selected = st.selectbox("नमुना वाक्य निवडा (Select Sample):", sample_sentences)
user_sentence = st.text_input("वाक्य (Sentence):", value=selected)

@st.cache_resource
def get_pos_tagger():
    return MarathiHMMPosTagger()

tagger = get_pos_tagger()

if user_sentence:
    tagged_tokens, trellis_steps = tagger.tag_sentence(user_sentence)

    tab_res, tab_trellis, tab_probs, tab_theory = st.tabs([
        "🏷️ टॅगिंग निकाल (Predicted POS Tags)",
        "🕸️ व्हिटरबी ट्रेलिस मॅट्रिक्स (Viterbi Trellis Steps)",
        "📊 HMM संभाव्यता मॅट्रिक्स (Transition & Initial Probabilities)",
        "📘 अल्गोरिदम गणित व मर्यादा (Theory & Limitations)"
    ])

    with tab_res:
        st.subheader("शब्दांच्या जाती निकाल तक्ता (Token Tagging Results)")
        if tagged_tokens:
            df_pos = pd.DataFrame(tagged_tokens)
            st.dataframe(df_pos, use_container_width=True, hide_index=True)

            st.markdown("#### व्हिज्युअल टॅग प्रतिनिधित्व (Tagged Sequence):")
            tag_badges = " ".join([
                f"<span style='display:inline-block; padding: 4px 10px; margin: 3px; border-radius: 8px; background: rgba(59, 130, 246, 0.15); border: 1px solid #3b82f6;'>"
                f"<b>{t['token']}</b> / <span style='color:#60a5fa; font-weight:bold;'>{t['predicted_tag']}</span>"
                f"</span>"
                for t in tagged_tokens
            ])
            st.markdown(tag_badges, unsafe_allow_html=True)

    with tab_trellis:
        st.subheader("व्हिटरबी डायनॅमिक प्रोग्रामिंग ट्रेलिस पायऱ्या (Step-by-Step Viterbi Trellis)")
        st.markdown(
            """
            प्रत्येक टप्प्यावर (t) मागील सर्व संभाव्य स्टेट्स आणि एमिशन संभाव्यतांचा गुणाकार करून कमाल स्कोअर काढला जातो:
            $$V(t, j) = \\max_i [V(t-1, i) \\cdot P(s_j \\mid s_i)] \\cdot P(w_t \\mid s_j)$$
            """
        )
        if trellis_steps:
            trellis_rows = []
            for step in trellis_steps:
                row = {"Step": step["step"], "Token": step["token"]}
                row.update(step["state_scores"])
                trellis_rows.append(row)
            df_trellis = pd.DataFrame(trellis_rows)
            st.dataframe(df_trellis, use_container_width=True, hide_index=True)

    with tab_probs:
        st.subheader("प्रारंभिक आणि संक्रमण संभाव्यता (Initial & Transition Probabilities)")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("**प्रारंभिक संभाव्यता distribution $\\pi(s)$:**")
            df_init = pd.DataFrame(list(tagger.initial_probs.items()), columns=["POS Tag", "Initial Probability"])
            st.dataframe(df_init, use_container_width=True, hide_index=True)
        with col_p2:
            st.markdown("**संक्रमण मॅट्रिक्स $A[s_i, s_j]$ (Transition Matrix):**")
            df_trans = pd.DataFrame(tagger.transition_probs).round(3)
            st.dataframe(df_trans, use_container_width=True)

    with tab_theory:
        st.subheader("HMM POS टॅगर: सिद्धांत आणि मर्यादा")
        st.markdown(
            """
            1. **स्वतः लिहिलेली अल्गोरिदम अंमलबजावणी**: हा टॅगर कोणत्याही ब्लॅक-बॉक्स लायब्ररीवर (उदा. spaCy/NLTK English tagger) अवलंबून नसून थेट HMM आणि Viterbi सूत्रांवर आधारित आहे.
            2. **अज्ञात शब्दांची हाताळणी (OOV Handling)**: नवीन शब्दांसाठी मराठी विभक्ती प्रत्यय (जसे *-ात, -ून, -ताना*) व संख्या पॅटर्न तपासून संभाव्यता दिली जाते.
            3. **डेटा मर्यादा**: हाताने लेबल केलेल्या २० वाक्यांच्या कॉर्पसमुळे काही दुर्मिळ शब्दांचे टॅग चुकण्याची शक्यता असते. मोठ्या कॉर्पसवर अचूकता अजून वाढवता येते.
            """
        )
