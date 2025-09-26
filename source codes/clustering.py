

'''Packaging-Focused Clustering Pipeline.

This script performs clustering on packaging-related reviews to produce 4-6 meta
categories combining specified packaging types, issue categories, and product
categories, with correlations, focusing on the top categories by review count.
'''

# Installation of dependencies


!pip install bertopic umap-learn hdbscan plotly pandas numpy matplotlib seaborn sentence-transformers nltk wordcloud joblib scikit-learn scipy

"""Main ()"""

!pip install bertopic umap-learn hdbscan plotly pandas numpy matplotlib seaborn sentence-transformers nltk wordcloud joblib scikit-learn scipy

import os
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from google.colab import drive, files
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from scipy.stats import chi2_contingency
from datetime import datetime
import re
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import time
import logging
import json

# Constants and Configuration
DATA_DIR: str = '/content/drive/MyDrive/data/'
MODELS_DIR: str = '/content/drive/MyDrive/models'
REPORTS_DIR: str = '/content/drive/MyDrive/reports'
RANDOM_SEED: int = 42
EMBEDDING_MODEL: str = 'all-mpnet-base-v2'
PACKAGING_MODEL_PATH: str = f'{MODELS_DIR}/packaging_tfidf_lr_model_20250925_1629.pkl'

# Focused Keyword Mappings
PACKAGING_ISSUE_CATEGORIES: Dict[str, List[str]] = {
    'Damaged Packaging': ['crushed', 'torn', 'broken', 'old', 'dirty', 'water-damaged', 'dent', 'smashed'],
    'Leakage / Spillage': ['liquids leaking', 'spilled', 'leak', 'spillage'],
    'Tampered / Opened Packaging': ['seal broken', 'opened', 'used', 'tampered']
}
PACKAGING_TYPES: Dict[str, List[str]] = {
    'Bottles & Jars': ['plastic bottles', 'glass bottles', 'wide-mouth jars', 'pump bottles', 'bottle', 'jar', 'container', 'vial'],
    'Tubes & Sticks': ['squeeze tubes', 'roll-on sticks', 'twist-up sticks', 'tube', 'stick', 'tubing', 'deodorant'],
    'Boxes & Cartons': ['folding cartons', 'rigid boxes', 'subscription-style boxes', 'box', 'carton', 'package']
}
PRODUCT_CATEGORIES: Dict[str, List[str]] = {
    'Craft Supplies': ['bow', 'ribbon', 'paint', 'brush', 'adhesive', 'tape'],
    'Home Goods': ['lamp', 'bulb', 'light', 'candle', 'match', 'baking soda', 'cleaner', 'eraser'],
    'Clothing & Accessories': ['sweater', 'belt', 'glove', 'hat', 'necklace', 'pendant', 'watch']
}

def setup_logging() -> None:
    """Set up logging for pipeline execution."""
    timestamp: str = datetime.now().strftime('%Y%m%d_%H%M')
    logging.basicConfig(filename=f'{REPORTS_DIR}/pipeline_log_{timestamp}.log', level=logging.INFO)
    logging.info("Starting pipeline execution")

def setup_nltk() -> None:
    """Download necessary NLTK data."""
    for resource in ['punkt', 'punkt_tab', 'wordnet', 'stopwords', 'omw-1.4', 'averaged_perceptron_tagger_eng']:
        try:
            nltk.download(resource, quiet=True)
        except LookupError:
            pass

def preprocess_text(text_series: pd.Series) -> pd.Series:
    """Preprocess text with lemmatization and stopword removal, preserving key terms.

    Args:
        text_series (pd.Series): Series of text to preprocess.

    Returns:
        pd.Series: Preprocessed text series.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) - {'not', 'no'}
    key_terms = set()
    for mapping in [PACKAGING_ISSUE_CATEGORIES, PACKAGING_TYPES, PRODUCT_CATEGORIES]:
        for keywords in mapping.values():
            for kw in keywords:
                key_terms.add(kw)

    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ''
        text = text.replace('<br />', ' ')
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token, pos='n') for token in tokens if token.isalnum() or token in key_terms]
        tokens = [token for token in tokens if token not in stop_words or token in key_terms]
        return ' '.join(tokens)

    return text_series.apply(clean_text)

def mount_drive() -> None:
    """Mount Google Drive if not already mounted."""
    if not os.path.ismount('/content/drive'):
        drive.mount('/content/drive', force_remount=False)

def upload_files_if_needed(required_files: List[str]) -> None:
    """Prompt user to upload files if not found in DATA_DIR.

    Args:
        required_files (List[str]): List of required file names.

    Raises:
        FileNotFoundError: If file upload fails.
    """
    for file_name in required_files:
        file_path = os.path.join(DATA_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"File {file_name} not found in {DATA_DIR}. Please upload it.")
            uploaded = files.upload()
            if file_name in uploaded:
                os.makedirs(DATA_DIR, exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(uploaded[file_name])
            else:
                raise FileNotFoundError(f"Upload failed for {file_name}.")

def load_data() -> pd.DataFrame:
    """Load and combine balanced datasets, including packaging subset.

    Returns:
        pd.DataFrame: Combined DataFrame with preprocessed text.

    Raises:
        ValueError: If no valid CSV files are loaded.
    """
    start_time = time.time()
    required_files = [
        'train_balanced.csv',
        'val_balanced.csv',
        'test_balanced.csv',
        'packaging_subset.csv'
    ]
    upload_files_if_needed(required_files)
    dfs = []
    for file_name in required_files:
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(file_path):
            temp_df = pd.read_csv(file_path)
            if 'full_text' not in temp_df.columns:
                print(f"Warning: {file_name} does not contain 'full_text' column. Skipping.")
                continue
            dfs.append(temp_df)
        else:
            print(f"Skipping {file_name} as it was not found after upload attempt.")
    if not dfs:
        raise ValueError("No valid CSV files loaded.")
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['full_text_cleaned'] = preprocess_text(combined_df['full_text'])
    elapsed_time = time.time() - start_time
    logging.info(f"Loaded {len(combined_df)} reviews in {elapsed_time:.2f} seconds")
    print(f"Data loading and preprocessing time: {elapsed_time:.2f} seconds")
    return combined_df

def log_unmatched_reviews(df: pd.DataFrame, category_type: str, mapping: Dict[str, List[str]]) -> None:
    """Log reviews that don't match any category keywords.

    Args:
        df (pd.DataFrame): DataFrame with reviews.
        category_type (str): Category type (e.g., 'issue_category').
        mapping (Dict[str, List[str]]): Mapping of categories to keywords.
    """
    unmatched = df[df[category_type].apply(len) == 0]['full_text_cleaned']
    timestamp: str = datetime.now().strftime('%Y%m%d_%H%M')
    log_path = f'{REPORTS_DIR}/unmatched_{category_type}_{timestamp}.txt'
    with open(log_path, 'w') as f:
        f.write(f"Unmatched reviews for {category_type}:\n")
        for text in unmatched.head(100):
            f.write(f"{text}\n")
    print(f"Unmatched reviews logged to {log_path}")

def extract_packaging_reviews(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Extract and tag packaging-focused reviews.

    Args:
        df (pd.DataFrame): Input DataFrame with reviews.
        model_path (str): Path to the packaging model.

    Returns:
        pd.DataFrame: Filtered DataFrame with packaging-related reviews.

    Raises:
        ValueError: If 'full_text_cleaned' column is missing.
        FileNotFoundError: If model file is not found.
    """
    start_time = time.time()
    if 'full_text_cleaned' not in df.columns:
        raise ValueError("DataFrame must have 'full_text_cleaned' column.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Packaging model not found at {model_path}.")
    model = joblib.load(model_path)
    df['predicted_issue'] = model.predict(df['full_text_cleaned']).astype(int)
    filtered_df = df[df['predicted_issue'] == 1].copy()

    def tag_review(text: str, mapping: Dict[str, List[str]]) -> List[str]:
        tags = []
        text_lower = text.lower()
        for category, keywords in mapping.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    tags.append(category)
                    break
        return tags

    filtered_df['issue_category'] = filtered_df['full_text_cleaned'].apply(
        lambda x: tag_review(x, PACKAGING_ISSUE_CATEGORIES)
    )
    filtered_df['packaging_type'] = filtered_df['full_text_cleaned'].apply(
        lambda x: tag_review(x, PACKAGING_TYPES)
    )
    filtered_df['product_category'] = filtered_df['full_text_cleaned'].apply(
        lambda x: tag_review(x, PRODUCT_CATEGORIES)
    )
    log_unmatched_reviews(filtered_df, 'issue_category', PACKAGING_ISSUE_CATEGORIES)
    log_unmatched_reviews(filtered_df, 'packaging_type', PACKAGING_TYPES)
    log_unmatched_reviews(filtered_df, 'product_category', PRODUCT_CATEGORIES)
    elapsed_time = time.time() - start_time
    logging.info(f"Extracted {len(filtered_df)} packaging reviews in {elapsed_time:.2f} seconds")
    print(f"Packaging review extraction time: {elapsed_time:.2f} seconds")
    return filtered_df

def generate_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Generate embeddings for the reviews.

    Args:
        df (pd.DataFrame): DataFrame with reviews and tags.

    Returns:
        np.ndarray: Generated embeddings.
    """
    start_time = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = (
        df['full_text_cleaned'] + ' ' +
        df['issue_category'].apply(lambda x: ' '.join(x)) + ' ' +
        df['packaging_type'].apply(lambda x: ' '.join(x)) + ' ' +
        df['product_category'].apply(lambda x: ' '.join(x))
    )
    embeddings = model.encode(texts.tolist(), batch_size=32, show_progress_bar=True)
    elapsed_time = time.time() - start_time
    logging.info(f"Generated embeddings for {len(df)} reviews in {elapsed_time:.2f} seconds")
    print(f"Embedding generation time: {elapsed_time:.2f} seconds")
    return embeddings

def perform_clustering(embeddings: np.ndarray, df: pd.DataFrame) -> Tuple[pd.DataFrame, BERTopic]:
    """Perform BERTopic clustering for 4-6 meta categories.

    Args:
        embeddings (np.ndarray): Precomputed embeddings.
        df (pd.DataFrame): DataFrame with reviews.

    Returns:
        Tuple[pd.DataFrame, BERTopic]: Clustered DataFrame and topic model.
    """
    start_time = time.time()
    texts = df['full_text_cleaned'].tolist()
    umap_model = UMAP(n_components=3, n_neighbors=150, metric='cosine', random_state=RANDOM_SEED)
    hdbscan_model = HDBSCAN(min_cluster_size=30, min_samples=1, metric='euclidean', cluster_selection_method='eom')
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, min_topic_size=30, nr_topics=6)
    topics, probs = topic_model.fit_transform(texts, embeddings)
    try:
        topic_model.reduce_topics(texts, nr_topics=6)
        topics = topic_model.topics_
    except Exception as e:
        logging.warning(f"Topic reduction failed: {e}. Using original topics.")
    df['cluster_id'] = topics
    df['cluster_prob'] = probs
    noise_count = (df['cluster_id'] == -1).sum()
    logging.info(f"Main clustering: {noise_count} noise points ({noise_count / len(df) * 100:.2f}%)")
    print(f"Number of noise points: {noise_count} ({noise_count / len(df) * 100:.2f}%)")
    noise_reviews = df[df['cluster_id'] == -1]['full_text_cleaned'].head(100)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    with open(f'{REPORTS_DIR}/noise_reviews_{timestamp}.txt', 'w') as f:
        f.write("Sample Noise Reviews:\n")
        for text in noise_reviews:
            f.write(f"{text}\n")
    print(f"Sample noise reviews saved to {REPORTS_DIR}/noise_reviews_{timestamp}.txt")
    non_noise_mask = df['cluster_id'] != -1
    silhouette = -1
    unique_clusters = df.loc[non_noise_mask, 'cluster_id'].unique()
    if non_noise_mask.sum() > 1 and len(unique_clusters) > 1:
        silhouette = silhouette_score(embeddings[non_noise_mask], df.loc[non_noise_mask, 'cluster_id'])
    logging.info(f"Main silhouette score: {silhouette:.4f}")
    print(f"Silhouette Score (excluding noise): {silhouette:.4f}")
    try:
        topic_info = topic_model.get_topic_info()
        topic_info.to_csv(f'{REPORTS_DIR}/main_topic_info_{timestamp}.csv', index=False)
        print(f"Main topic representations saved to {REPORTS_DIR}/main_topic_info_{timestamp}.csv")
        fig = topic_model.visualize_hierarchy()
        fig.write_html(f'{REPORTS_DIR}/topic_hierarchy_{timestamp}.html')
        print(f"Topic hierarchy visualization saved to {REPORTS_DIR}/topic_hierarchy_{timestamp}.html")
    except Exception as e:
        logging.warning(f"Topic info/hierarchy failed: {e}. Skipping.")
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
    projections = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=projections[:, 0], y=projections[:, 1], hue=df['cluster_id'], palette='viridis')
    plt.title('Cluster Visualization')
    plot_path = f'{REPORTS_DIR}/cluster_visual_{timestamp}.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Cluster visual saved to {plot_path}")
    fig = px.scatter(x=projections[:, 0], y=projections[:, 1], color=df['cluster_id'].astype(str), title='Interactive Cluster Visualization')
    fig.write_html(f'{REPORTS_DIR}/cluster_visual_{timestamp}.html')
    print(f"Interactive cluster visual saved to {REPORTS_DIR}/cluster_visual_{timestamp}.html")
    elapsed_time = time.time() - start_time
    logging.info(f"Clustering and visualization time: {elapsed_time:.2f} seconds")
    print(f"Clustering and visualization time: {elapsed_time:.2f} seconds")
    return df, topic_model

def compute_correlations(df: pd.DataFrame, topic_model: BERTopic, timestamp: str) -> None:
    """Compute correlations and save top 4-6 meta categories with most reviews.

    Args:
        df (pd.DataFrame): Clustered DataFrame.
        topic_model (BERTopic): Fitted BERTopic model.
        timestamp (str): Timestamp for file naming.
    """
    non_noise_df = df[df['cluster_id'] != -1].copy()
    correlations_path = f'{REPORTS_DIR}/meta_categories_correlations_{timestamp}.csv'
    meta_categories_path = f'{REPORTS_DIR}/meta_categories_top_{timestamp}.csv'
    metrics_path = f'{REPORTS_DIR}/clustering_metrics_{timestamp}.txt'
    # Explode list columns and reset index to avoid duplicates
    exploded_df = non_noise_df.explode('packaging_type').explode('issue_category').explode('product_category').reset_index(drop=True)
    # Crosstabs for correlations
    crosstab_packaging = pd.crosstab(exploded_df['cluster_id'], exploded_df['packaging_type'], normalize='index') * 100
    crosstab_issue = pd.crosstab(exploded_df['cluster_id'], exploded_df['issue_category'], normalize='index') * 100
    crosstab_product = pd.crosstab(exploded_df['cluster_id'], exploded_df['product_category'], normalize='index') * 100
    # Save crosstabs as individual CSVs
    crosstab_packaging.to_csv(f'{REPORTS_DIR}/crosstab_packaging_{timestamp}.csv')
    crosstab_issue.to_csv(f'{REPORTS_DIR}/crosstab_issue_{timestamp}.csv')
    crosstab_product.to_csv(f'{REPORTS_DIR}/crosstab_product_{timestamp}.csv')
    print(f"Meta categories correlations saved to {REPORTS_DIR}/crosstab_[packaging|issue|product]_{timestamp}.csv")
    # Chi-square tests (on non-normalized crosstabs)
    crosstab_packaging_raw = pd.crosstab(exploded_df['cluster_id'], exploded_df['packaging_type'])
    chi2_packaging, p_packaging, _, _ = chi2_contingency(crosstab_packaging_raw)
    crosstab_issue_raw = pd.crosstab(exploded_df['cluster_id'], exploded_df['issue_category'])
    chi2_issue, p_issue, _, _ = chi2_contingency(crosstab_issue_raw)
    crosstab_product_raw = pd.crosstab(exploded_df['cluster_id'], exploded_df['product_category'])
    chi2_product, p_product, _, _ = chi2_contingency(crosstab_product_raw)
    # Calculate purity
    purity_scores = {
        'issue_category': calculate_cluster_purity(non_noise_df, 'cluster_id', 'issue_category'),
        'packaging_type': calculate_cluster_purity(non_noise_df, 'cluster_id', 'packaging_type'),
        'product_category': calculate_cluster_purity(non_noise_df, 'cluster_id', 'product_category')
    }
    # Combine for meta categories, select top 4-6 by count
    topic_info = topic_model.get_topic_info()
    meta_categories = []
    for cluster_id in non_noise_df['cluster_id'].unique():
        top_packaging = crosstab_packaging.loc[cluster_id].idxmax() if cluster_id in crosstab_packaging.index else 'Unknown'
        top_issue = crosstab_issue.loc[cluster_id].idxmax() if cluster_id in crosstab_issue.index else 'Unknown'
        top_product = crosstab_product.loc[cluster_id].idxmax() if cluster_id in crosstab_product.index else 'Unknown'
        top_words = topic_model.get_topic(cluster_id)[:5]
        top_words = [word[0] for word in top_words] if top_words else []
        meta_category = f"{top_issue} in {top_packaging} for {top_product}"
        count = len(non_noise_df[non_noise_df['cluster_id'] == cluster_id])
        meta_categories.append({
            'Cluster_ID': cluster_id,
            'Meta_Category': meta_category,
            'Top_Packaging_Type': top_packaging,
            'Top_Issue_Category': top_issue,
            'Top_Product_Category': top_product,
            'Top_Words': top_words,
            'Count': count,
            'Packaging_Percentage': crosstab_packaging.loc[cluster_id, top_packaging] if cluster_id in crosstab_packaging.index and top_packaging in crosstab_packaging.columns else 0,
            'Issue_Percentage': crosstab_issue.loc[cluster_id, top_issue] if cluster_id in crosstab_issue.index and top_issue in crosstab_issue.columns else 0,
            'Product_Percentage': crosstab_product.loc[cluster_id, top_product] if cluster_id in crosstab_product.index and top_product in crosstab_product.columns else 0,
            'Issue_Purity': purity_scores['issue_category'].get(cluster_id, 0),
            'Packaging_Purity': purity_scores['packaging_type'].get(cluster_id, 0),
            'Product_Purity': purity_scores['product_category'].get(cluster_id, 0)
        })
    meta_df = pd.DataFrame(meta_categories)
    # Select top 4-6 meta categories by count
    top_meta_df = meta_df.sort_values(by='Count', ascending=False).head(6)
    top_meta_df.to_csv(meta_categories_path, index=False)
    print(f"Top 4-6 meta categories saved to {meta_categories_path}")
    # Save metrics
    with open(metrics_path, 'w') as f:
        f.write("--- Clustering Metrics ---\n\n")
        f.write("Main Cluster Size Distribution:\n")
        cluster_counts = non_noise_df['cluster_id'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            f.write(f"Cluster {cluster_id}: {count} reviews\n")
        f.write("\nTop Keywords per Cluster:\n")
        for cluster_id in cluster_counts.index:
            top_words = topic_model.get_topic(cluster_id)[:5]
            top_words = [word[0] for word in top_words] if top_words else []
            f.write(f"Cluster {cluster_id}: {top_words}\n")
        f.write("\nCluster Purity:\n")
        for category_type, scores in purity_scores.items():
            f.write(f"  {category_type}:\n")
            for cluster_id, purity in scores.items():
                f.write(f"    Cluster {cluster_id}: {purity:.4f}\n")
        f.write("\nChi-Square Test Results:\n")
        f.write(f"packaging_type: chi2={chi2_packaging:.2f}, p={p_packaging:.4f}\n")
        f.write(f"issue_category: chi2={chi2_issue:.2f}, p={p_issue:.4f}\n")
        f.write(f"product_category: chi2={chi2_product:.2f}, p={p_product:.4f}\n")
    print(f"Clustering metrics saved to {metrics_path}")
    print("\nChi-Square Test Results:")
    print(f"packaging_type: chi2={chi2_packaging:.2f}, p={p_packaging:.4f}")
    print(f"issue_category: chi2={chi2_issue:.2f}, p={p_issue:.4f}")
    print(f"product_category: chi2={chi2_product:.2f}, p={p_product:.4f}")

def calculate_cluster_purity(df: pd.DataFrame, cluster_col: str, category_col: str) -> Dict[int, float]:
    """Calculate purity for each cluster based on dominant category.

    Args:
        df (pd.DataFrame): DataFrame with clusters.
        cluster_col (str): Column name for cluster IDs.
        category_col (str): Column name for category tags.

    Returns:
        Dict[int, float]: Purity scores for each cluster.
    """
    purity_scores = {}
    for cluster_id in df[cluster_col].unique():
        if cluster_id == -1:
            continue
        cluster_df = df[df[cluster_col] == cluster_id]
        category_counts = cluster_df[category_col].explode().value_counts()
        if not category_counts.empty:
            dominant_category_count = category_counts.max()
            total_reviews = len(cluster_df)
            purity_scores[cluster_id] = dominant_category_count / total_reviews
    return purity_scores

def save_results(df: pd.DataFrame) -> None:
    """Save clustered DataFrame to Drive as CSV and JSON.

    Args:
        df (pd.DataFrame): Clustered DataFrame.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    df_path_csv = f'{REPORTS_DIR}/clustered_packaging_{timestamp}.csv'
    df_path_json = f'{REPORTS_DIR}/clustered_packaging_{timestamp}.json'
    df.to_csv(df_path_csv, index=False)
    df.to_json(df_path_json, orient='records', indent=2)
    print(f"Clustered data saved to {df_path_csv} (for summarization) and {df_path_json} (for app).")

def main() -> None:
    """Main function to run the clustering pipeline."""
    start_time = time.time()
    setup_logging()
    mount_drive()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    setup_nltk()
    df = load_data()
    packaging_df = extract_packaging_reviews(df, PACKAGING_MODEL_PATH)
    embeddings = generate_embeddings(packaging_df)
    clustered_df, topic_model = perform_clustering(embeddings, packaging_df)
    compute_correlations(clustered_df, topic_model, datetime.now().strftime('%Y%m%d_%H%M'))
    save_results(clustered_df)
    elapsed_time = time.time() - start_time
    logging.info(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()