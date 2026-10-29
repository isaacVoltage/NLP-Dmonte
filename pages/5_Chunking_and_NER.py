"""
RecipeParse - Experiment 5: Chunking & Named Entity Recognition (NER)
Extracts culinary entities and syntactic chunks (NP, VP, PP) from Marathi text.
"""
import streamlit as st
import pandas as pd
from src.utils import init_app_page
from src.chunking import MarathiChunker
from src.recipe_ner import MarathiRecipeNER

current_theme = init_app_page("Exp 5: Chunking & NER", "🏷️")

st.markdown(
    """
    <div class="main-header">
        <h1>🏷️ Experiment 5: Chunking & Named Entity Recognition</h1>
        <p>मराठी पाककृती वाक्यांमधून नाम समूह (NP), क्रियापद समूह (VP), शब्दयोगी समूह (PP) आणि स्वयंपाक एंटिटीजची ओळख</p>
    </div>
    """,
    unsafe_allow_html=True
)

sample_sentences = [
    "एका कढईत दोन चमचे तेल गरम करा.",
    "त्यात एक चमचा मोहरी आणि अर्धा चमचा जिरे घाला.",
    "बारीक चिरलेला कांदा घालून पाच मिनिटे परता.",
    "दोन कप साबुदाणा चार तास पाण्यात भिजवून ठेवा.",
    "मंद आचेवर दहा मिनिटे रवा चांगला भाजून घ्या."
]

st.subheader("मराठी वाक्य निवडा किंवा प्रविष्ट करा (Select or Enter Sentence)")
chosen = st.selectbox("नमुना वाक्ये (Sample Sentences):", sample_sentences)
custom_sent = st.text_input("वाक्य तपासा (Input Sentence):", value=chosen)

chunker = MarathiChunker()
ner = MarathiRecipeNER()

if custom_sent:
    tab_ner, tab_chunk, tab_guide = st.tabs([
        "🏷️ नामनिर्देशित एंटिटी ओळख (NER)",
        "🧩 चंकिंग विश्लेषण (NP, VP, PP Chunks)",
        "📚 एंटिटी आणि चंक टॅग मार्गदर्शक (Tag Guide)"
    ])

    with tab_ner:
        st.subheader("हायलाईट केलेले नामनिर्देशित घटक (Inline Highlighted Entities)")
        annotated_html = ner.render_annotated_html(custom_sent)
        st.markdown(
            f"""
            <div style="padding: 18px; border-radius: 12px; border: 1px solid #334155; line-height: 2.2; font-size: 1.2rem; background: rgba(30, 41, 59, 0.4);">
                {annotated_html}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("एंटिटी निकाल तक्ता (Extracted Entities Table)")
        entities = ner.extract_entities(custom_sent)
        if entities:
            df_ent = pd.DataFrame(entities)
            st.dataframe(df_ent[["text", "label", "start", "end"]], use_container_width=True, hide_index=True)
        else:
            st.info("कोणतीही एंटिटी आढळली नाही.")

    with tab_chunk:
        st.subheader("वाक्य चंकिंग विश्लेषण (Syntactic Chunk Parsing)")
        chunks = chunker.chunk_sentence(custom_sent)
        if chunks:
            df_chunks = pd.DataFrame(chunks)
            st.dataframe(df_chunks[["phrase", "label", "description"]], use_container_width=True, hide_index=True)

            st.markdown("#### चंक सादरीकरण (Visual Representation)")
            chunk_display = " ".join([f"<b>[{c['label']}:</b> <i>{c['phrase']}</i><b>]</b>" for c in chunks])
            st.markdown(
                f"""
                <div style="padding: 15px; border-radius: 10px; background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; font-size: 1.1rem;">
                    {chunk_display}
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab_guide:
        st.subheader("एंटिटीज आणि चंकिंग लेबल्सचे अर्थ (Tag Documentation)")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(
                """
                #### Named Entities:
                - **INGREDIENT**: पाककृतीतील खाद्य घटक (उदा. तेल, कांदा, मोहरी, रवा)
                - **QUANTITY**: प्रमाण किंवा संख्या (उदा. दोन, एक, अर्धा, ५०)
                - **UNIT**: मोजमापाचे एकक (उदा. चमचे, कप, वाटी, ग्रॅम)
                - **ACTION**: पाककृती कृती किंवा क्रियापद (उदा. घाला, परता, शिजवा)
                - **DURATION**: लागणारा वेळ (उदा. पाच मिनिटे, अर्धा तास)
                - **TEMPERATURE**: उष्णता पातळी (उदा. मंद आच, मध्यम आचेवर)
                - **UTENSIL**: स्वयंपाकाची भांडी (उदा. कढई, पातेले, कुकर)
                """
            )
        with col_g2:
            st.markdown(
                """
                #### Chunk Labels:
                - **NP (Noun Phrase)**: नाम, विशेषण व परिमाणांचा समूह (उदा. *एक चमचा मोहरी*, *चिरलेला कांदा*)
                - **VP (Verb Phrase)**: मुख्य व सहाय्यकारी क्रियापद समूह (उदा. *गरम करा*, *परतून घ्या*)
                - **PP (Postpositional Phrase)**: शब्दयोगी अव्यय किंवा विभक्ती प्रत्यय जोडलेले पद (उदा. *पातेल्यात*, *मंद आचेवर*)
                """
            )
