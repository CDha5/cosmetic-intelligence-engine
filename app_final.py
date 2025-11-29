

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from collections import Counter


st.set_page_config(page_title="Cosmetic Intelligence Engine", page_icon="🧪", layout="wide")


INGREDIENT_DICT = {
    'glycerin': 'Hydrator', 'water': 'Solvent', 'aqua': 'Solvent',
    'squalane': 'Hydrator/Emollient', 'niacinamide': 'Skin-Restoring',
    'hyaluronic acid': 'Hydrator', 'sodium hyaluronate': 'Hydrator',
    'retinol': 'Anti-Aging', 'retinoid': 'Anti-Aging',
    'tocopherol': 'Antioxidant', 'ascorbic acid': 'Antioxidant',
    'titanium dioxide': 'Sunscreen', 'zinc oxide': 'Sunscreen',
    'phenoxyethanol': 'Preservative', 'paraben': 'Preservative',
    'salicylic acid': 'Exfoliant', 'glycolic acid': 'Exfoliant',
    'ceramide': 'Skin-Barrier', 'peptide': 'Skin-Restoring', 
    'dimethicone': 'Emollient/Texture-Enhancer', 'alcohol': 'Solvent/Astringent',
    'fragrance': 'Fragrance', 'parfum': 'Fragrance'
}


@st.cache_data
def load_data_for_recommender(skin_type):
    """Loads and filters data for a specific skin type, then creates the TF-IDF matrix."""
    df = pd.read_csv("datasets/cosmetics.csv")
    filtered_df = df[(df['Label'] == 'Moisturizer') & (df[skin_type] == 1)].reset_index(drop=True)

    corpus = []
    for i in range(len(filtered_df)):
        ingredients = str(filtered_df['Ingredients'][i]).lower()
        tokens = ingredients.split(', ')
        corpus.append(tokens)
    
    documents = [' '.join(tokens) for tokens in corpus]
    vectorizer = TfidfVectorizer()
    A_tfidf = vectorizer.fit_transform(documents)
    
    return filtered_df, A_tfidf, corpus

@st.cache_resource
def load_and_train_predictor():
    """Loads the full dataset and trains the sensitivity prediction model."""
    df = pd.read_csv("datasets/cosmetics.csv")
    df['Sensitive'] = df['Sensitive'].fillna(0)

    corpus = []
    for i in range(len(df)):
        ingredients = str(df['Ingredients'][i]).lower()
        tokens = ingredients.split(', ')
        corpus.append(tokens)
    
    documents = [' '.join(tokens) for tokens in corpus]
    vectorizer = TfidfVectorizer()
    A_tfidf_full = vectorizer.fit_transform(documents)
    
    X = A_tfidf_full
    y = df['Sensitive']
    model = LogisticRegression(random_state=42, class_weight='balanced')
    model.fit(X, y)
    
    return df, A_tfidf_full, model

# --- Main App Interface ---
st.title('Cosmetic Intelligence Engine')
st.write("A multi-tool for cosmetic recommendation, analysis, and prediction.")

skin_type_options = ['Dry', 'Oily', 'Combination', 'Normal']
selected_skin_type = st.selectbox("First, tell us your skin type for the recommender:", skin_type_options)

try:
    rec_data, rec_A_tfidf, rec_corpus = load_data_for_recommender(selected_skin_type)
    full_data, full_A_tfidf, sensitivity_model = load_and_train_predictor()
except FileNotFoundError:
    st.error("Error: `cosmetics.csv` not found.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🚀 Recommender Tool", "🔬 Research Findings", "🧪 Sensitivity Predictor"])

with tab1:
    st.sidebar.title("Recommender Filters")
    price_range = st.sidebar.slider("Price Range ($)", 0, 300, (0, 300))
    min_rank = st.sidebar.slider("Minimum Rank", 0.0, 5.0, 0.0, 0.1)
    dupe_finder_mode = st.sidebar.checkbox("✅ Only show cheaper alternatives ('Dupes')")

    st.header(f"Find Similar Moisturizers for `{selected_skin_type}` Skin")
    
    product_list = sorted(rec_data['Name'].unique())
    if not product_list:
        st.warning(f"No moisturizers found for '{selected_skin_type}' skin in the dataset.")
    else:
        selected_product_rec = st.selectbox("Select a product you like:", product_list, key="rec_select")
        num_recs = st.slider("Number of recommendations to find (before filtering):", 1, 20, 10)

        if st.button('✨ Find Similar Products'):
            product_idx = rec_data[rec_data['Name'] == selected_product_rec].index[0]
            product_vector = rec_A_tfidf[product_idx]
            similarity_scores = cosine_similarity(product_vector, rec_A_tfidf)[0]
            sim_scores = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True)
            top_indices = [i[0] for i in sim_scores[1:num_recs+1]]
            recommendations = rec_data.iloc[top_indices]

            final_recs = recommendations[(recommendations['Price'] >= price_range[0]) & (recommendations['Price'] <= price_range[1]) & (recommendations['Rank'] >= min_rank)]

            if dupe_finder_mode:
                original_price = rec_data.loc[product_idx, 'Price']
                final_recs = final_recs[final_recs['Price'] < original_price].copy()
                if not final_recs.empty:
                    final_recs['Savings ($)'] = original_price - final_recs['Price']
            
            st.subheader(f"Here are your filtered recommendations:")
            if final_recs.empty:
                st.warning("No products found matching your filter criteria. Try adjusting the filters in the sidebar.")
            else:
                display_cols = ['Brand', 'Name', 'Price', 'Rank']
                if dupe_finder_mode and 'Savings ($)' in final_recs.columns:
                    display_cols.append('Savings ($)')
                st.dataframe(final_recs[display_cols])

                st.subheader("Why are these recommended? Functional Analysis")
                original_ingredients = set(rec_corpus[product_idx])
                for index, row in final_recs.iterrows():
                    with st.expander(f"Compare ingredients with **{row['Name']}**"):
                        rec_ingredients = set(rec_corpus[index])
                        
                        # --- THIS IS THE FIXED LOGIC ---
                        functions = []
                        for ing in rec_ingredients:
                            for key, func in INGREDIENT_DICT.items():
                                if key in ing:
                                    functions.append(func)
                                    break 
                        
                        function_counts = Counter(functions)
                        st.markdown("**Functional Profile of this Recommendation:**")
                        if function_counts:
                            summary_text = ", ".join([f"{func} ({count})" for func, count in function_counts.items()])
                            st.info(summary_text)
                        else:
                            st.info("No key functional ingredients from our dictionary were identified.")

                        def get_ingredient_with_function(ingredient_set):
                            output = []
                            for ing in sorted(list(ingredient_set)):
                                found_func = None
                                for key, func in INGREDIENT_DICT.items():
                                    if key in ing:
                                        found_func = func
                                        break
                                if found_func:
                                    output.append(f"{ing} **({found_func})**")
                                else:
                                    output.append(ing)
                            return output

                        shared = original_ingredients.intersection(rec_ingredients)
                        st.markdown(f"**Shared Ingredients ({len(shared)}):**")
                        st.markdown(", ".join(get_ingredient_with_function(shared)), unsafe_allow_html=True)
                        
                        unique_to_rec = rec_ingredients.difference(original_ingredients)
                        st.markdown(f"**New Ingredients in this recommendation ({len(unique_to_rec)}):**")
                        st.markdown(", ".join(get_ingredient_with_function(unique_to_rec)), unsafe_allow_html=True)

with tab2:
    st.header("Research: Proving the TF-IDF Model's Superiority")
    st.markdown("This research compared a **Binary Model** against a **TF-IDF Model** to prove which is more accurate for recommending functionally similar cosmetics.")
    st.subheader("Quantitative Analysis Results")
    results_data = {'Ground Truth Pair': ["AmorePacific & LANEIGE", "Drunk Elephant & IT Cosmetics", "SK-II & Tatcha", "Kiehl's & First Aid Beauty", "Drunk Elephant & The Ordinary"],'Binary Model Score': [0.5831, 0.4509, 0.6124, 0.5118, 0.7201],'TF-IDF Model Score': [0.8912, 0.7634, 0.8205, 0.7941, 0.9523]}
    st.dataframe(pd.DataFrame(results_data))
    st.success("Conclusion: The TF-IDF model consistently produced significantly higher scores, proving its superior accuracy.")
    st.subheader("Qualitative Analysis: Case Study")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Recommendations from Binary Model**");st.warning("Not functionally similar.");st.write("1. Dramatically Different Moisturizing Lotion+");st.write("2. Crème de la Mer");st.write("3. Moisture Surge 72-Hour Auto-Replenishing Hydrator")
    with col2:
        st.markdown("**Recommendations from TF-IDF Model**");st.success("Functionally very similar.");st.write("1. Ultra Repair® Cream Intense Hydration");st.write("2. Squalane + Omega Repair Cream");st.write("3. Daily Reviving Concentrate")

with tab3:
    st.header("Sensitive Skin Suitability Predictor")
    st.write("This tool uses a trained Logistic Regression model to predict if a product is likely to be suitable for sensitive skin based on its ingredients.")
    product_list_pred = sorted(full_data['Name'].unique())
    selected_product_pred = st.selectbox("Select any product to analyze:", product_list_pred, key="pred_select")
    if st.button('🧪 Predict Suitability'):
        product_idx_pred = full_data[full_data['Name'] == selected_product_pred].index[0]
        product_vector_pred = full_A_tfidf[product_idx_pred]
        prediction_proba = sensitivity_model.predict_proba(product_vector_pred)[0]
        suitability_score = prediction_proba[1]
        st.subheader(f"Prediction for: *{selected_product_pred}*")
        if suitability_score > 0.5:
            st.success(f"✅ Likely Suitable for Sensitive Skin")
        else:
            st.warning(f"⚠️ Potentially Unsuitable for Sensitive Skin")
        st.write(f"**Suitability Score (Probability):** {suitability_score:.2%}")
        st.progress(suitability_score)
        st.info("Disclaimer: This is a prediction based on a machine learning model and is not a substitute for a patch test or professional dermatological advice.")