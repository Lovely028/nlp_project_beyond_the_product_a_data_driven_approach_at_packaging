

"""
Preprocess Amazon Health & Personal Care reviews dataset for sentiment and packaging analysis.

Inputs:
    /content/drive/MyDrive/data/Health_and_Personal_Care.jsonl
    /content/drive/MyDrive/data/meta_Health_and_Personal_Care.jsonl
Outputs:
    /content/data/train_balanced.csv, /content/data/val_balanced.csv, /content/data/test_balanced.csv
    /content/data/packaging_subset.csv
    /content/reports/sentiment_distribution.png, /content/reports/packaging_counts.png
    /content/reports/sentiment_before.png, /content/reports/sentiment_after.png

Dependencies:
    Run in Colab after installing:
    !pip install pandas>=1.5.0 numpy>=1.23.0 matplotlib>=3.5.0 seaborn>=0.11.0 scikit-learn>=1.0.0 nltk>=3.7 emoji>=2.0.0
    NLTK 'punkt' and 'punkt_tab' are downloaded automatically.
"""
# Installations
pip install emoji

import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
import emoji
from google.colab import drive


def mount_drive() -> None:
    """Mount Google Drive if not already mounted."""
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    else:
        print("Google Drive already mounted.")


def download_nltk_data() -> None:
    """Download required NLTK data."""
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def create_directories() -> None:
    """Create necessary directories in Colab."""
    os.makedirs('/content/data', exist_ok=True)
    os.makedirs('/content/reports', exist_ok=True)


def check_dataset_files() -> tuple[str, str]:
    """Check if dataset files exist in Google Drive."""
    dataset_path_reviews = '/content/drive/MyDrive/data/Health_and_Personal_Care.jsonl'
    dataset_path_meta = '/content/drive/MyDrive/data/meta_Health_and_Personal_Care.jsonl'

    if not os.path.exists('/content/drive/'):
        raise FileNotFoundError("Google Drive is not mounted. Please mount and authorize.")

    data_dir = '/content/drive/MyDrive/data/'
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        mydrive_dir = '/content/drive/MyDrive/'
        if os.path.exists(mydrive_dir):
            print("Contents of /content/drive/MyDrive/:")
            print(os.listdir(mydrive_dir))
        raise FileNotFoundError(
            f"Dataset files not found at {dataset_path_reviews} and {dataset_path_meta}. "
            "Upload files to /content/drive/MyDrive/data/"
        )

    print(f"Directory found: {data_dir}")
    print(f"Files in {data_dir}:")
    for file in os.listdir(data_dir):
        print(file)

    if not (os.path.exists(dataset_path_reviews) and os.path.exists(dataset_path_meta)):
        raise FileNotFoundError(
            f"Dataset files not found at {dataset_path_reviews} and {dataset_path_meta}. "
            "Upload files to /content/drive/MyDrive/data/"
        )
    print("Dataset files found in Google Drive.")
    return dataset_path_reviews, dataset_path_meta


def load_data(reviews_path: str, meta_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess review and meta data."""
    try:
        df_reviews = pd.read_json(reviews_path, lines=True)
        df_reviews = df_reviews.rename(columns={
            'text': 'reviewText',
            'rating': 'overall',
            'timestamp': 'unixReviewTime'
        })
        print(f"Reviews loaded: {len(df_reviews)}")

        df_meta = pd.read_json(meta_path, lines=True)
        required_meta_cols = ['parent_asin', 'main_category']
        if not all(col in df_meta.columns for col in required_meta_cols):
            print(f"Columns in df_meta: {df_meta.columns.tolist()}")
            raise KeyError(f"Required columns {required_meta_cols} not in meta data.")

        df_meta = df_meta[required_meta_cols].rename(columns={'parent_asin': 'asin'})
        print(f"Meta records loaded: {len(df_meta)}")
        return df_reviews, df_meta
    except Exception as e:
        raise IOError(f"Error loading JSONL files: {e}")


def merge_data(df_reviews: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """Merge reviews with meta data on 'asin'."""
    df = pd.merge(df_reviews, df_meta, on='asin', how='left')
    print(f"Merged reviews: {len(df)}")
    return df


def print_counts(df: pd.DataFrame, stage: str) -> None:
    """Print sentiment and packaging issue counts, skipping packaging if not present."""
    print(f"\n{stage}:")
    sentiment_counts = df['sentiment_label'].value_counts().to_dict()
    print(f"Sentiment counts: Negative={sentiment_counts.get('negative', 0)}, "
          f"Neutral={sentiment_counts.get('neutral', 0)}, "
          f"Positive={sentiment_counts.get('positive', 0)}")

    if 'is_packaging_issue' in df.columns:
        packaging_counts = df['is_packaging_issue'].value_counts().to_dict()
        print(f"Packaging issues: Yes={packaging_counts.get(1, 0)}, No={packaging_counts.get(0, 0)}")
    else:
        print("Packaging issues: Not yet computed")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the merged dataframe."""
    print(f"Total reviews before cleaning: {len(df)}")

    df = df.dropna(subset=['reviewText', 'overall', 'asin', 'unixReviewTime'])
    if 'reviewerID' in df.columns:
        df = df.drop_duplicates(subset=['asin', 'reviewerID'])

    df['full_text'] = df['title'].fillna('') + ' ' + df['reviewText'].fillna('')

    def remove_emojis(text: str) -> str:
        return emoji.replace_emoji(text, replace='')

    df['full_text'] = df['full_text'].apply(remove_emojis)

    df['word_count'] = df['full_text'].apply(lambda x: len(word_tokenize(x)))
    df = df[df['word_count'] > 10].drop(columns=['word_count'])

    print(f"Total reviews after cleaning: {len(df)}")
    return df


def map_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Map overall rating to sentiment label."""
    def get_sentiment_label(rating: float) -> str:
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'

    df['sentiment_label'] = df['overall'].apply(get_sentiment_label)
    return df


def flag_packaging_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Flag packaging issues using regex."""
    keywords = [
        'packag', 'box', 'bubble wrap', 'crush', 'dented', 'torn', 'damaged', 'broken',
        'leak', 'smashed', 'seal', 'tape', 'bottle', 'jar', 'sachet', 'pouch', 'blister',
        'tube', 'container', 'poor packaging'
    ]
    pattern = re.compile(r'\b(?:' + '|'.join(keywords) + r')\b', re.IGNORECASE)
    df['is_packaging_issue'] = df['full_text'].str.contains(pattern, na=False).astype(int)
    return df


def add_review_date(df: pd.DataFrame) -> pd.DataFrame:
    """Convert unixReviewTime to review_date."""
    df['review_date'] = pd.to_datetime(df['unixReviewTime'], unit='s').dt.date
    return df


def generate_visuals(df: pd.DataFrame, balanced_df: pd.DataFrame) -> None:
    """Generate and save visualization plots."""
    plt.figure(figsize=(10, 5))
    sns.countplot(x='sentiment_label', data=df, order=['negative', 'neutral', 'positive'])
    plt.title('Sentiment Distribution Before Balancing')
    plt.savefig('/content/reports/sentiment_before.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.countplot(x='sentiment_label', hue='is_packaging_issue', data=df,
                  order=['negative', 'neutral', 'positive'])
    plt.title('Packaging Issues per Sentiment')
    plt.savefig('/content/reports/packaging_counts.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.countplot(x='sentiment_label', data=balanced_df, order=['negative', 'neutral', 'positive'])
    plt.title('Sentiment Distribution After Balancing')
    plt.savefig('/content/reports/sentiment_after.png')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(ax=axes[0], x='sentiment_label', data=df, order=['negative', 'neutral', 'positive'])
    axes[0].set_title('Before Balancing')
    sns.countplot(ax=axes[1], x='sentiment_label', data=balanced_df,
                  order=['negative', 'neutral', 'positive'])
    axes[1].set_title('After Balancing')
    plt.savefig('/content/reports/sentiment_distribution.png')
    plt.close()


def create_packaging_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Create and save packaging subset."""
    total_packaging = df['is_packaging_issue'].sum()
    print(f"Total packaging issues flagged: {total_packaging}")

    packaging_subset = df[df['is_packaging_issue'] == 1]
    relevant_cols = ['asin', 'full_text', 'sentiment_label', 'is_packaging_issue', 'review_date',
                    'main_category']
    packaging_subset[relevant_cols].to_csv('/content/data/packaging_subset.csv', index=False)
    print("Packaging subset created: Cleaned and ready for downstream processing (e.g., clustering).")
    return packaging_subset


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Balance the dataset by undersampling to the smallest group size."""
    # Undersampling: Select min_group_size samples from each group without replacement
    group_sizes = df.groupby(['sentiment_label', 'is_packaging_issue']).size()
    min_group_size = group_sizes.min()
    print(f"Minimum group size for balancing: {min_group_size}")

    balanced_dfs = []
    for _, group in df.groupby(['sentiment_label', 'is_packaging_issue']):
        balanced_group = group.sample(min_group_size, random_state=42, replace=False)
        balanced_dfs.append(balanced_group)

    balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df


def stratified_split(balanced_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified train/val/test split."""
    balanced_df['stratify_col'] = balanced_df['sentiment_label'] + '_' + \
                                 balanced_df['is_packaging_issue'].astype(str)
    train_df, temp_df = train_test_split(
        balanced_df, test_size=0.3, stratify=balanced_df['stratify_col'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['stratify_col'], random_state=42
    )

    train_df = train_df.drop(columns=['stratify_col'])
    val_df = val_df.drop(columns=['stratify_col'])
    test_df = test_df.drop(columns=['stratify_col'])
    return train_df, val_df, test_df


def save_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save the balanced train/val/test CSVs."""
    relevant_cols = ['asin', 'full_text', 'sentiment_label', 'is_packaging_issue', 'review_date',
                    'main_category']
    train_df[relevant_cols].to_csv('/content/data/train_balanced.csv', index=False)
    val_df[relevant_cols].to_csv('/content/data/val_balanced.csv', index=False)
    test_df[relevant_cols].to_csv('/content/data/test_balanced.csv', index=False)


def main() -> None:
    """Orchestrate the preprocessing pipeline."""
    mount_drive()
    download_nltk_data()
    set_seed()
    create_directories()
    reviews_path, meta_path = check_dataset_files()
    df_reviews, df_meta = load_data(reviews_path, meta_path)
    df = merge_data(df_reviews, df_meta)
    df = clean_data(df)
    df = map_sentiment(df)
    print_counts(df, "After cleaning (before packaging features)")
    df = flag_packaging_issues(df)
    df = add_review_date(df)
    print_counts(df, "After adding sentiment and packaging features")
    create_packaging_subset(df)
    balanced_df = balance_dataset(df)
    generate_visuals(df, balanced_df)
    train_df, val_df, test_df = stratified_split(balanced_df)
    save_datasets(train_df, val_df, test_df)
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()