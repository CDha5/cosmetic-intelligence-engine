

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Cosmetic Recommender", page_icon="💄")


@st.cache_data
def load_and_process_data():
    """Loads data, filters it, and creates the ingredient matrix."""
    df = pd.read_csv("datasets/cosmetics.csv")
    moisturizers = df[df['Label'] == "Moisturizer"]
    moisturizers_dry = moisturizers[moisturizers["Dry"] == 1].reset_index(drop=True)

    ingredient_idx = {}
    corpus = []
    idx = 0
    for i in range(len(moisturizers_dry)):
        ingredients = moisturizers_dry['Ingredients'][i].lower()
        tokens = ingredients.split(', ')
        corpus.append(tokens)
        for ingredient in tokens:
            if ingredient not in ingredient_idx:
                ingredient_idx[ingredient] = idx
                idx += 1

    M = len(moisturizers_dry)
    N = len(ingredient_idx)
    A = np.zeros((M, N))

    for i, tokens in enumerate(corpus):
        for ingredient in tokens:
            if ingredient in ingredient_idx:
                A[i, ingredient_idx[ingredient]] = 1

    return moisturizers_dry, A


st.title('Cosmetic Recommendation Engine')
st.write("Find moisturizers with similar ingredient lists. A great way to discover new products or find more affordable alternatives!")


try:
    data, A_matrix = load_and_process_data()
except FileNotFoundError:
    st.error("Error: `cosmetics.csv` not found. Please make sure the `datasets` folder and its contents are in the same directory as `app.py`.")
    st.stop()



product_list = sorted(data['Name'].unique())
selected_product = st.selectbox(
    "First, select a moisturizer for dry skin that you like:",
    product_list
)


num_recs = st.slider("How many recommendations would you like?", 1, 10, 5)


if st.button('✨ Find Similar Products'):
    if selected_product:
        product_idx = data[data['Name'] == selected_product].index[0]
        product_vector = A_matrix[product_idx, :].reshape(1, -1)

        similarity_scores = cosine_similarity(product_vector, A_matrix)[0]
        sim_scores = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True)

        top_indices = [i[0] for i in sim_scores[1:num_recs+1]]
        recommendations = data.iloc[top_indices][['Brand', 'Name', 'Price', 'Rank']]

        st.subheader(f"Here are your top {num_recs} recommendations:")
        st.dataframe(recommendations)
    else:
        st.warning("Please select a product first.")