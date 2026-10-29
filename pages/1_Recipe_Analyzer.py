"""
RecipeParse - Integrated Marathi Recipe Analyzer
Combines all experiments into an end-to-end extraction pipeline.
"""
import streamlit as st
import pandas as pd
import json
from src.utils import init_app_page, load_recipes
from src.recipe_extractor import IntegratedRecipeExtractor

current_theme = init_app_page("Recipe Analyzer - एकात्मिक विश्लेषक", "🍳")

st.markdown(
    """
    <div class="main-header">
        <h1>🍳 Integrated Recipe Analyzer (एकात्मिक पाककृती विश्लेषक)</h1>
        <p>मराठी पाककृती मजकुरातून घटक, प्रमाण, एकके, वेळ, तापमान आणि कृतींचे संपूर्ण स्वयंचलित निष्कासन</p>
    </div>
    """,
    unsafe_allow_html=True
)

recipes = load_recipes()
sample_options = {r["title"]: r["text"] for r in recipes}

# Input Section
st.subheader("📝 मराठी पाककृती मजकूर प्रविष्ट करा (Enter Recipe Text)")
selected_sample = st.selectbox(
    "नमुना पाककृती निवडा किंवा खाली स्वतःचा मजकूर टाईप करा (Select Sample Recipe):",
    ["-- स्वतःचा मजकूर प्रविष्ट करा (Custom Input) --"] + list(sample_options.keys())
)

default_text = (
    "एका पातेल्यात दोन चमचे तेल गरम करा. त्यात एक चमचा मोहरी घाला. "
    "त्यानंतर बारीक चिरलेला कांदा घालून पाच मिनिटे परता. चवीनुसार एक चमचा मीठ घाला."
)

if selected_sample != "-- स्वतःचा मजकूर प्रविष्ट करा (Custom Input) --":
    input_text = sample_options[selected_sample]
else:
    input_text = default_text

recipe_input = st.text_area(
    "पाककृती मजकूर (Recipe Text in Marathi):",
    value=input_text,
    height=140
)

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    analyze_btn = st.button("🚀 विश्लेषण करा (Analyze Recipe)", type="primary", use_container_width=True)

if analyze_btn or recipe_input:
    extractor = IntegratedRecipeExtractor()
    with st.spinner("पाककृतीचे NLP विश्लेषण सुरू आहे..."):
        result = extractor.process_recipe(recipe_input)

    if not result.get("success", False):
        st.error(f"❌ त्रुटी: {result.get('error', 'अवैध इनपुट')}")
    else:
        st.success("✅ पाककृतीचे विश्लेषण यशस्वीरीत्या पूर्ण झाले!")

        # KPI Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("ओळखलेले घटक (Ingredients)", result["extracted_ingredients_count"])
        with m2:
            st.metric("एकूण कृती (Actions)", result["extracted_actions_count"])
        with m3:
            st.metric("पाककृती टप्पे (Steps)", result["total_steps"])
        with m4:
            st.metric("एकूण शब्द (Tokens)", result["tokens_count"])
        with m5:
            st.metric("देवनागरी प्रमाण", f"{result['script_validation']['percentage']}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs for categorized views
        t_struct, t_entities, t_steps, t_pos, t_json = st.tabs([
            "📊 संरचित माहिती तक्ता (Structured Table)",
            "🏷️ नामनिर्देशित एंटिटीज (Recognized Entities)",
            "📋 टप्प्यांचे वर्गीकरण (Recipe Steps & Chunks)",
            "🏷️ POS टॅगिंग (Part of Speech)",
            "💾 डेटा निर्यात (JSON / CSV Export)"
        ])

        with t_struct:
            st.subheader("घटक व पाककृती माहिती तक्ता (Structured Ingredients & Actions Table)")
            struct_data = result["structured_table"]
            if struct_data:
                df_struct = pd.DataFrame(struct_data)
                st.dataframe(df_struct, use_container_width=True, hide_index=True)

                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    csv_data = df_struct.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 संरचित तक्ता CSV डाउनलोड करा",
                        data=csv_data,
                        file_name="recipe_extracted_table.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col_dl2:
                    json_str = json.dumps(struct_data, ensure_ascii=False, indent=2).encode('utf-8')
                    st.download_button(
                        "📥 संरचित डेटा JSON डाउनलोड करा",
                        data=json_str,
                        file_name="recipe_extracted.json",
                        mime="application/json",
                        use_container_width=True
                    )
            else:
                st.info("या मजकुरात कोणतेही थेट घटक आढळले नाहीत.")

        with t_entities:
            st.subheader("हायलाईट केलेले नामनिर्देशित घटक (Named Entity Highlighting)")
            st.markdown(
                f"""
                <div style="padding: 18px; border-radius: 12px; border: 1px solid #334155; line-height: 2.2; font-size: 1.15rem; background: rgba(30, 41, 59, 0.4);">
                    {result['highlighted_html']}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            
            ent_list = result["entities"]
            if ent_list:
                df_ents = pd.DataFrame(ent_list)[["text", "label", "start", "end"]]
                st.dataframe(df_ents, use_container_width=True, hide_index=True)

        with t_steps:
            st.subheader("पाककृतीतील अनुक्रमिक पायऱ्या (Normalized Sequential Steps)")
            for i, step in enumerate(result["normalized_steps"], start=1):
                st.markdown(f"**टप्पा {i}:** {step}")

            st.markdown("---")
            st.subheader("वाक्य चंकिंग विश्लेषण (Syntactic Chunks: NP, VP, PP)")
            for item in result["chunks_by_step"]:
                with st.expander(f"वाक्य: {item['step_text']}"):
                    chunk_df = pd.DataFrame([
                        {"घटक गट (Phrase)": c["phrase"], "लेबल (Label)": c["label"], "स्पष्टीकरण": c["description"]}
                        for c in item["chunks"]
                    ])
                    st.dataframe(chunk_df, use_container_width=True, hide_index=True)

        with t_pos:
            st.subheader("HMM व व्हिटरबीद्वारे शब्दांच्या जाती (POS Sequence)")
            pos_df = pd.DataFrame(result["pos_tags"])
            if not pos_df.empty:
                st.dataframe(pos_df[["token", "predicted_tag", "confidence", "explanation"]], use_container_width=True, hide_index=True)

        with t_json:
            st.subheader("संपूर्ण निष्कासन निकाल (Complete Extraction Payload)")
            full_json_export = {
                "original_text": result["original_text"],
                "total_steps": result["total_steps"],
                "structured_ingredients": result["structured_table"],
                "durations": result["extracted_durations"],
                "temperatures": result["extracted_temperatures"],
                "utensils": result["extracted_utensils"]
            }
            st.json(full_json_export)
            full_json_bytes = json.dumps(full_json_export, ensure_ascii=False, indent=2).encode('utf-8')
            st.download_button(
                "📥 संपूर्ण निकाल JSON डाउनलोड करा",
                data=full_json_bytes,
                file_name="recipe_full_analysis.json",
                mime="application/json"
            )
