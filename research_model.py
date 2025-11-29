import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def run_experiment():
    """
    This function executes the entire research methodology from data loading
    to model comparison and result presentation.
    """
    # --- 1. Data Collection and Preprocessing ---
    print("Step 1: Loading and preprocessing data...")
    df = pd.read_csv("datasets/cosmetics.csv")
    moisturizers = df[df['Label'] == "Moisturizer"]
    data = moisturizers[moisturizers["Dry"] == 1].reset_index(drop=True)

    corpus = []
    for i in range(len(data)):
        ingredients = str(data['Ingredients'][i]).lower()
        tokens = ingredients.split(', ')
        corpus.append(tokens)
    print(f"Data processed. Found {len(data)} products for the experiment.")
    print("-" * 50)

    # --- 2. Model Implementation ---
    print("Step 2: Implementing both models...")
    
    # Model A (Control): Binary Bag-of-Words Model
    unique_ingredients = sorted(list(set(ing for doc in corpus for ing in doc)))
    vocab_map = {ingredient: i for i, ingredient in enumerate(unique_ingredients)}
    A_binary = np.zeros((len(data), len(unique_ingredients)), dtype=int)
    for doc_idx, doc in enumerate(corpus):
        for ing in doc:
            if ing in vocab_map:
                A_binary[doc_idx, vocab_map[ing]] = 1
    print(f"Model A (Binary) created with shape: {A_binary.shape}")

    # Model B (Proposed Solution): TF-IDF Model
    documents = [' '.join(tokens) for tokens in corpus]
    vectorizer = TfidfVectorizer()
    A_tfidf = vectorizer.fit_transform(documents)
    print(f"Model B (TF-IDF) created with shape: {A_tfidf.shape}")
    print("-" * 50)

    # --- 3. Evaluation Protocol ---
    print("Step 3: Performing quantitative analysis...")
    
    # Ground Truth Establishment
    ground_truth_pairs = [
        ("Color Control Cushion Compact Broad Spectrum SPF 50+", "BB Cushion Hydra Radiance SPF 50"),
        ("Protini™ Polypeptide Cream", "Confidence in a Cream™ Transforming Moisturizing Super Cream"),
        ("Facial Treatment Essence (Pitera Essence)", "The Essence Plumping Skin Softener"),
        ("Ultra Facial Cream", "Ultra Repair® Cream Intense Hydration"),
        ("Virgin Marula Luxury Facial Oil", "100 percent Pure Argan Oil") # Using Argan as a proxy for Marula if not in data
    ]

    print("\n--- Quantitative Results: Cosine Similarity Scores ---")
    print(f"{'Ground Truth Pair':<65} {'Binary Score':<15} {'TF-IDF Score':<15}")
    print("=" * 100)
    
    for prod1_name, prod2_name in ground_truth_pairs:
        try:
            idx1 = data[data['Name'] == prod1_name].index[0]
            idx2 = data[data['Name'] == prod2_name].index[0]

            # Binary Similarity
            sim_binary = cosine_similarity(A_binary[idx1].reshape(1, -1), A_binary[idx2].reshape(1, -1))[0][0]
            
            # TF-IDF Similarity
            sim_tfidf = cosine_similarity(A_tfidf[idx1], A_tfidf[idx2])[0][0]
            
            pair_name = f"{prod1_name[:25]}... & {prod2_name[:25]}..."
            print(f"{pair_name:<65} {sim_binary:<15.4f} {sim_tfidf:<15.4f}")
        except IndexError:
            print(f"Skipping pair: One or both products not found ({prod1_name}, {prod2_name})")
    print("-" * 50)
    
    # --- 4. Qualitative Case Study ---
    print("\nStep 4: Performing qualitative analysis case study...")
    case_study_product = "Ultra Facial Cream"
    
    def get_recommendations(product_name, matrix, data, top_n=3):
        try:
            idx = data[data['Name'] == product_name].index[0]
            sims = cosine_similarity(matrix[idx], matrix)[0]
            sim_scores = sorted(list(enumerate(sims)), key=lambda x: x[1], reverse=True)
            top_indices = [i[0] for i in sim_scores[1:top_n+1]]
            return data.iloc[top_indices]['Name'].tolist()
        except (IndexError, ValueError):
            return ["Product not found for case study."]

    recs_binary = get_recommendations(case_study_product, A_binary, data)
    recs_tfidf = get_recommendations(case_study_product, A_tfidf, data)

    print(f"\n--- Qualitative Results: Recommendations for '{case_study_product}' ---")
    print("\nRecommendations from Binary Model (Model A):")
    for i, rec in enumerate(recs_binary, 1):
        print(f"{i}. {rec}")
        
    print("\nRecommendations from TF-IDF Model (Model B):")
    for i, rec in enumerate(recs_tfidf, 1):
        print(f"{i}. {rec}")
    print("-" * 50)
    print("Experiment complete.")

# --- Run the entire experiment when the script is executed ---
if __name__ == "__main__":
    run_experiment()