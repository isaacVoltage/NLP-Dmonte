"""
RecipeParse - Experiment 8: EDA and Word Cloud Dashboard
Exploratory Data Analysis of Marathi Recipe Corpus.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import init_app_page
from src.text_analytics import MarathiTextAnalytics

current_theme = init_app_page("Exp 8: EDA Dashboard", "📈")

st.markdown(
    """
    <div class="main-header">
        <h1>📈 Experiment 8: Exploratory Data Analysis & Word Cloud</h1>
        <p>मराठी पाककृती डेटासेटचे अन्वेषणात्मक सांख्यिकी विश्लेषण, वर्डक्लाउड आणि आलेखी सादरीकरण</p>
    </div>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def get_analytics():
    return MarathiTextAnalytics()

analytics = get_analytics()
stats = analytics.get_recipe_statistics()

# Top KPI Summary Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>एकूण पाककृती (Recipes)</h3>
            <p>{stats['total_recipes']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>एकूण शब्द (Total Tokens)</h3>
            <p>{stats['total_tokens']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>शब्दसंग्रह (Vocabulary)</h3>
            <p>{stats['vocabulary_size']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>सरासरी वेळ (Avg Time)</h3>
            <p>{stats['avg_cooking_time']} min</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

tab_wc, tab_freq, tab_cooccur, tab_cat, tab_missing = st.tabs([
    "☁️ मराठी वर्डक्लाउड (Word Cloud)",
    "📊 वारंवारता विश्लेषण (Frequency Distributions)",
    "🔥 घटक सह-संबंध (Ingredient Co-occurrence)",
    "📂 प्रवर्गनिहाय तुलना (Category Comparison)",
    "❓ गहाळ डेटा विश्लेषण (Missing Data & Correlation)"
])

with tab_wc:
    st.subheader("मराठी पाककृती वर्डक्लाउड (Marathi Recipe Word Cloud)")
    st.markdown("मराठी स्टॉपवर्ड्स वगळून सर्वाधिक वापरल्या गेलेल्या शब्दांचे व्हिज्युअल सादरीकरण:")
    with st.spinner("वर्डक्लाउड तयार करत आहे..."):
        fig_wc = analytics.generate_wordcloud_image()
        if fig_wc:
            st.pyplot(fig_wc)
        else:
            st.warning("वर्डक्लाउड जनरेट करण्यासाठी पुरेसा डेटा उपलब्ध नाही.")

with tab_freq:
    st.subheader("वारंवारता वितरण (Frequency Distributions)")
    f_tab1, f_tab2, f_tab3 = st.tabs(["प्रमुख घटक (Top Ingredients)", "पाककृती क्रियापदे (Actions)", "एकके (Measurement Units)"])

    with f_tab1:
        ing_df = analytics.get_ingredient_frequencies(15)
        col_i1, col_i2 = st.columns([3, 2])
        with col_i1:
            st.bar_chart(ing_df.set_index("Ingredient")["Recipe_Count"], color="#10b981")
        with col_i2:
            st.dataframe(ing_df, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 घटक वारंवारता CSV",
                data=ing_df.to_csv(index=False).encode('utf-8'),
                file_name="ingredient_frequencies.csv",
                mime="text/csv"
            )

    with f_tab2:
        act_df = analytics.get_cooking_action_frequencies(12)
        col_a1, col_a2 = st.columns([3, 2])
        with col_a1:
            st.bar_chart(act_df.set_index("Cooking_Action")["Frequency"], color="#ff6b35")
        with col_a2:
            st.dataframe(act_df, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 क्रियापद वारंवारता CSV",
                data=act_df.to_csv(index=False).encode('utf-8'),
                file_name="action_frequencies.csv",
                mime="text/csv"
            )

    with f_tab3:
        unit_df = analytics.get_unit_frequencies()
        st.dataframe(unit_df, use_container_width=True, hide_index=True)

with tab_cooccur:
    st.subheader("घटकांचा सह-संबंध (Ingredient Co-occurrence Matrix & Heatmap)")
    st.markdown("कोणते घटक पाककृतींमध्ये वारंवार एकत्र येतात याचे परस्पर संबंध:")
    co_matrix = analytics.get_ingredient_cooccurrence_matrix(10)
    
    fig_heat, ax = plt.subplots(figsize=(8, 6), facecolor="#1e293b")
    ax.set_facecolor("#1e293b")
    sns.heatmap(co_matrix, annot=True, fmt='d', cmap='Oranges', ax=ax, cbar=False)
    plt.xticks(rotation=45, color="white", fontsize=9)
    plt.yticks(rotation=0, color="white", fontsize=9)
    st.pyplot(fig_heat)

    st.download_button(
        "📥 Co-occurrence Matrix CSV",
        data=co_matrix.to_csv().encode('utf-8'),
        file_name="ingredient_cooccurrence_matrix.csv",
        mime="text/csv"
    )

with tab_cat:
    st.subheader("प्रवर्गनिहाय तुलनात्मक विश्लेषण (Category Comparison)")
    cat_df = analytics.get_category_comparison()
    st.dataframe(cat_df, use_container_width=True, hide_index=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("**प्रवर्गनिहाय पाककृती संख्या:**")
        st.bar_chart(cat_df.set_index("category")["recipe_count"], color="#3b82f6")
    with col_c2:
        st.markdown("**प्रवर्गनिहाय सरासरी स्वयंपाक वेळ (Minutes):**")
        st.bar_chart(cat_df.set_index("category")["avg_cooking_time"], color="#f59e0b")

with tab_missing:
    st.subheader("गहाळ डेटा आणि सहसंबंध विश्लेषण (Missing Data & Correlation)")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### गहाळ मूल्यांचे विश्लेषण (Missing Data Analysis):")
        missing_df = analytics.get_missing_data_summary()
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

    with col_m2:
        st.markdown("#### संख्यात्मक चलांचा परस्पर सहसंबंध (Correlation Analysis):")
        corr_df = analytics.get_correlation_data()
        st.dataframe(corr_df, use_container_width=True)
        st.caption("टीप: सहसंबंध (Correlation) हा केवळ सांख्यिकी संबंध दर्शवतो, तो कारण-परिणाम (Causation) सिद्ध करत नाही.")
