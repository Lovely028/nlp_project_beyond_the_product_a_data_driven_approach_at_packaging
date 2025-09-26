
'''
Packaging Summarization Pipeline for analyzing Amazon Health and Personal Care review 2023.
Assumes CSV and model files are manually uploaded to /content/ in Colab.
Uses a fine-tuned GPT model for summarization, with fallback to GPT-4o.
Samples 50 reviews to avoid rate limit errors and enforces strict JSON output.
'''

# Standard Installation
"""

!pip install pandas scikit-learn sentence-transformers openai matplotlib numpy

"""Main ()"""

# Standard library imports
import json
import os
import random
import re
from typing import Any, Dict, List

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.colab import files
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import joblib

# Constants
CONTENT_PATH = "/content"
CSV_FILENAME = "packaging_subset.csv"
MODEL_FILENAME = "packaging_tfidf_lr_model_20250925_1629.pkl"
CSV_PATH = os.path.join(CONTENT_PATH, CSV_FILENAME)
MODEL_PATH = os.path.join(CONTENT_PATH, MODEL_FILENAME)
JSON_OUTPUT_PATH = os.path.join(CONTENT_PATH, "summary_finetuned.json")
ISSUES_CHART_PATH = os.path.join(CONTENT_PATH, "packaging_issues_finetuned.png")
TYPES_CHART_PATH = os.path.join(CONTENT_PATH, "packaging_types_finetuned.png")
CATEGORIES_CHART_PATH = os.path.join(CONTENT_PATH, "product_categories_finetuned.png")
REVIEW_COLUMN = "full_text"
N_CLUSTERS = 3
RANDOM_STATE = 42
N_INIT = 10
EMBEDDING_MODEL = "distilbert-base-uncased"
FINETUNED_MODEL_ID = "ft:gpt-3.5-turbo-0125:your-org:custom-name:xxxxxx"  # Replace with actual model ID
FALLBACK_MODEL = "gpt-4o"  # Fallback model if fine-tuned model ID is invalid
TEMPERATURE = 0.0
BYPASS_MODEL = False  # Set to True to skip model-based filtering
MAX_REVIEWS = 50  # Sample size to avoid rate limit errors

def check_files_exist() -> None:
    """Check if required CSV file (and model file, if not bypassed) exists in /content/."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at {CSV_PATH}. Please upload 'packaging_subset.csv' to /content/")
    print(f"Found CSV file: {CSV_PATH}")

    if not BYPASS_MODEL and not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please upload '{MODEL_FILENAME}' to /content/ or set BYPASS_MODEL = True")
    if not BYPASS_MODEL:
        print(f"Found model file: {MODEL_PATH}")

def load_reviews(csv_path: str) -> List[str]:
    """Load review texts from the CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if REVIEW_COLUMN not in df.columns:
            raise ValueError(f"Column '{REVIEW_COLUMN}' not found in CSV. Available: {df.columns.tolist()}")
        return df[REVIEW_COLUMN].dropna().astype(str).tolist()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV file not found at {csv_path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading CSV: {e}") from e

def load_packaging_model(model_path: str) -> Pipeline:
    """Load the TF-IDF + Logistic Regression model from the PKL file using joblib."""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        print(f"Model file found at: {model_path}")

        # Validate file size
        file_size = os.path.getsize(model_path)
        if file_size < 100:  # Arbitrary threshold for empty/invalid files
            raise ValueError(f"Model file '{model_path}' is too small ({file_size} bytes), likely corrupted")

        try:
            model = joblib.load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load pickle file: {e}. Ensure it was created with joblib by modeling1.py.") from e
        if not isinstance(model, Pipeline):
            raise ValueError("Loaded model must be a scikit-learn Pipeline")
        print("Model loaded successfully")
        return model
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found at {model_path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}") from e

def extract_packaging_texts(reviews: List[str], model: Pipeline) -> List[str]:
    """Filter packaging-related reviews using the TF-IDF + LR model."""
    packaging_texts = []
    for review in reviews:
        try:
            prediction = model.predict([review])[0]
            if prediction == "1":  # 1 = packaging-related (as strings per modeling1.py)
                packaging_texts.append(review)
        except Exception as e:
            print(f"Error predicting for review '{review[:50]}...': {e}")
    return packaging_texts

def cluster_texts(packaging_texts: List[str]) -> Dict[int, List[str]]:
    """Embed and cluster the packaging texts."""
    if not packaging_texts:
        raise ValueError("No packaging-related texts to cluster")
    if len(packaging_texts) < N_CLUSTERS:
        raise ValueError(f"Need at least {N_CLUSTERS} texts for clustering; got {len(packaging_texts)}")

    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = embedder.encode(packaging_texts)
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = kmeans.fit_predict(embeddings)

        clustered_reviews: Dict[int, List[str]] = {i: [] for i in range(N_CLUSTERS)}
        for idx, label in enumerate(labels):
            clustered_reviews[label].append(packaging_texts[idx])
        return clustered_reviews
    except Exception as e:
        raise RuntimeError(f"Error during clustering: {e}") from e

def clean_json_response(response_text: str) -> str:
    """Clean GPT-4o response to extract valid JSON."""
    # Remove markdown code fences and extra text
    response_text = re.sub(r'```json\n|\n```', '', response_text)
    response_text = re.sub(r'```.*?```', '', response_text, flags=re.DOTALL)
    response_text = response_text.strip()

    # Ensure response starts with { and ends with }
    if not response_text.startswith('{'):
        response_text = response_text[response_text.find('{'):] if '{' in response_text else '{}'
    if not response_text.endswith('}'):
        response_text = response_text[:response_text.rfind('}') + 1] if '}' in response_text else '{}'

    return response_text

def summarize_with_finetuned_model(clustered_reviews: Dict[int, List[str]], client: OpenAI) -> str:
    """Summarize clustered reviews using the fine-tuned model or fallback to GPT-4o."""
    use_fallback = FINETUNED_MODEL_ID == "ft:gpt-3.5-turbo-0125:your-org:custom-name:xxxxxx"
    model_id = FALLBACK_MODEL if use_fallback else FINETUNED_MODEL_ID
    prompt = (
        "Return a JSON object summarizing the following Amazon reviews. The JSON must have three keys: "
        "'packaging_issues', 'packaging_types', and 'product_categories'. Each key should contain sub-categories "
        "(e.g., 'Damaged packaging', 'Bottles & jars'). Each sub-category must have 'summary' (string), "
        "'count' (integer), and 'percent' (float, percentage of total reviews). "
        "Output only the JSON object, with no additional text or markdown.\n\n"
        f"Reviews grouped by cluster:\n{json.dumps(clustered_reviews, indent=2)}"
    ) if use_fallback else f"Summarize the key packaging themes from these customer reviews into a JSON object:\n\n{json.dumps(clustered_reviews, indent=2)}"

    try:
        print(f"📦 Summarizing with {'fallback model GPT-4o' if use_fallback else f'fine-tuned model {FINETUNED_MODEL_ID}'}")
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE
        )
        response_text = response.choices[0].message.content
        # Clean response to ensure valid JSON
        cleaned_response = clean_json_response(response_text)
        # Validate JSON
        json.loads(cleaned_response)  # Raises JSONDecodeError if invalid
        return cleaned_response
    except Exception as e:
        raise RuntimeError(f"Error during summarization with model {model_id}: {e}") from e

def plot_summary(data: Dict[str, Any], title: str, output_path: str) -> None:
    """Plot and save a bar chart for the summary data."""
    try:
        labels = list(data.keys())
        counts = [data[key]["count"] for key in labels]
        percents = [data[key]["percent"] for key in labels]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, counts, color="skyblue")
        for bar, pct in zip(bars, percents):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.1,
                f"{pct}%",
                ha="center",
                va="bottom",
                fontsize=10
            )
        plt.title(title, fontsize=14)
        plt.ylabel("Number of Mentions")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, format="png", dpi=300)
        plt.close()
        print(f"Saved chart to {output_path}")
    except Exception as e:
        raise RuntimeError(f"Error plotting summary to {output_path}: {e}") from e

def main() -> None:
    """Main function to run the end-to-end packaging summarization pipeline."""
    try:
        # Step 1: Check for manually uploaded files
        print("📥 Checking for uploaded files...")
        check_files_exist()
        client = OpenAI(api_key=userdata.get("OPENAI_API_KEY"))

        # Step 2: Load data and model
        reviews = load_reviews(CSV_PATH)
        print(f"Total reviews loaded: {len(reviews)}")

        # Step 3: Extract packaging-related texts (or bypass)
        if BYPASS_MODEL:
            print("⚠️ Bypassing model-based filtering; using all reviews")
            packaging_texts = reviews
        else:
            model = load_packaging_model(MODEL_PATH)
            packaging_texts = extract_packaging_texts(reviews, model)
        print(f"Packaging-related texts extracted: {len(packaging_texts)}")
        if not packaging_texts:
            print("⚠️ No packaging-related texts found")
            return

        # Step 4: Sample reviews to avoid rate limit
        if len(packaging_texts) > MAX_REVIEWS:
            print(f"Sampling {MAX_REVIEWS} reviews from {len(packaging_texts)} to avoid rate limit")
            packaging_texts = random.sample(packaging_texts, MAX_REVIEWS)

        print("\n🔍 Extracted Packaging Mentions (first 5):")
        for text in packaging_texts[:5]:  # Limit to first 5 for brevity
            print(f"- {text[:100]}...")

        # Step 5: Cluster texts
        clustered_reviews = cluster_texts(packaging_texts)
        print("\n📂 Clustered Reviews:")
        print(json.dumps(clustered_reviews, indent=2))

        # Step 6: Summarize with fine-tuned model or fallback
        final_summary_text = summarize_with_finetuned_model(clustered_reviews, client)
        print("\n📦 Final JSON Structured Summary:")
        print(final_summary_text)

        # Step 7: Save and visualize results
        with open(JSON_OUTPUT_PATH, "w") as f:
            f.write(final_summary_text)
        print(f"Saved JSON summary to {JSON_OUTPUT_PATH}")

        try:
            summary_json = json.loads(final_summary_text)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON summary: {e}. Saving raw output for submission.")
            summary_json = {}  # Fallback to empty dict to skip charting

        if "packaging_issues" in summary_json:
            plot_summary(summary_json["packaging_issues"], "Packaging Issues", ISSUES_CHART_PATH)
        if "packaging_types" in summary_json:
            plot_summary(summary_json["packaging_types"], "Packaging Types", TYPES_CHART_PATH)
        if "product_categories" in summary_json:
            plot_summary(summary_json["product_categories"], "Product Categories", CATEGORIES_CHART_PATH)

        # Auto-download outputs
        print("\n📥 Downloading output files...")
        files.download(JSON_OUTPUT_PATH)
        if summary_json:
            files.download(ISSUES_CHART_PATH)
            files.download(TYPES_CHART_PATH)
            files.download(CATEGORIES_CHART_PATH)

    except FileNotFoundError as e:
        print(f"File error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from model: {e}")
    except Exception as e:
        print(f"Pipeline error: {e}")

if __name__ == "__main__":
    main()