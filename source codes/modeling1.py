
"""
Pylint-Compliant Combined Classification Pipeline for Velana.net
This script trains and evaluates two separate models:
1. A TF-IDF + Logistic Regression model for packaging issue detection.
2. A fine-tuned DeBERTa-v3 model with LoRA for sentiment analysis.
The script is optimized for a full run in approximately 1.5 hours and follows
software engineering best practices for readability and maintainability.

To prevent Colab runtime disconnection, copy and paste the following JavaScript
into the Colab browser console (right-click > Inspect > Console):
```javascript
function KeepAlive() {
    console.log("Keeping Colab alive...");
    document.querySelector("colab-connect-button").click();
    setTimeout(KeepAlive, 60000);
}
KeepAlive();
```
This script clicks the Colab interface every 60 seconds to keep the session active.
Run it in the console before starting the pipeline.
"""
# --- Section 0: Installation ---
!pip install --no-cache-dir -q peft>=0.5.0 pandas>=1.5.0 numpy>=1.23.0 scikit-learn>=1.0.0
!pip install --no-cache-dir -q transformers>=4.30.0 torch>=2.0.0 matplotlib>=3.5.0 seaborn>=0.11.0
!pip install --no-cache-dir -q nltk>=3.7 nlpaug>=1.1.11 wandb>=0.15.0
print("Dependencies installed successfully.")

# --- Section 1: Imports ---
import os
import joblib
from typing import Tuple, Dict
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import torch
import shutil
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from transformers.trainer_callback import EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nlpaug.augmenter.word as naw
try:
    from google.colab import drive, userdata
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# --- Section 2: Constants and Configuration ---
# File Paths and Identifiers
DATA_DIR = '/content/drive/MyDrive/data/'
MODELS_DIR = '/content/drive/MyDrive/models'
REPORTS_DIR = '/content/drive/MyDrive/reports'
BASE_MODEL = 'microsoft/deberta-v3-base'
# !!! IMPORTANT: Change this to your Hugging Face username !!!
HF_USERNAME = "YOUR_HF_USERNAME_HERE"
# Model & Training Hyperparameters
LORA_R = 8
LORA_ALPHA = 16
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 2e-5
RANDOM_SEED = 42

# --- Section 3: Helper Classes and Functions ---
class HFDataset(torch.utils.data.Dataset):
    """Custom Hugging Face Dataset class."""
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings['input_ids'])

def setup_nltk():
    """Download necessary NLTK data with retry logic."""
    print("Downloading NLTK resources...")
    resources = ['punkt', 'punkt_tab', 'wordnet', 'stopwords', 'averaged_perceptron_tagger_eng']
    for resource in resources:
        for attempt in range(3):
            try:
                nltk.download(resource, quiet=True)
                print(f"Downloaded {resource} successfully.")
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed to download {resource}: {e}")
                if attempt == 2:
                    raise RuntimeError(f"Failed to download NLTK resource {resource} after 3 attempts.")
    print("All NLTK resources downloaded successfully.")

def preprocess_text(text_series: pd.Series) -> pd.Series:
    """Preprocess text with lemmatization and stopword removal."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) - {'not', 'no'}
    def clean_text(text: str) -> str:
        """Helper function to clean a single string."""
        if not isinstance(text, str):
            return ''
        text = text.replace('<br />', ' ')
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    return text_series.apply(clean_text)

def augment_sentiment_data_fast(train_df: pd.DataFrame) -> pd.DataFrame:
    """Fast augmentation for neutral and negative classes with synonyms."""
    print("Augmenting sentiment data (fast mode)...")
    aug = naw.SynonymAug(aug_src='wordnet', aug_max=3)
    class_counts = train_df['sentiment_label'].value_counts()
    max_count = class_counts.max()
    augmented_dfs = [train_df]
    for label in ['neutral', 'negative']:
        class_df = train_df[train_df['sentiment_label'] == label]
        num_to_generate = max_count - len(class_df)
        if num_to_generate > 0:
            print(f"Generating {num_to_generate} new samples for '{label}'...")
            aug_texts = aug.augment(class_df['full_text'].tolist(), n=num_to_generate)
            aug_df = pd.DataFrame({
                'full_text': aug_texts,
                'sentiment_label': [label] * len(aug_texts)
            })
            augmented_dfs.append(aug_df)
    aug_df = pd.concat(augmented_dfs, ignore_index=True)
    aug_df = aug_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Train samples after augmentation: {len(aug_df)}")
    if wandb.run:
        wandb.log({"augmented_train_size": len(aug_df)})
    return aug_df

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, preprocess, and augment datasets for both tasks."""
    print("Loading and preparing data...")
    required_files = [
        os.path.join(DATA_DIR, 'train_balanced.csv'),
        os.path.join(DATA_DIR, 'val_balanced.csv'),
        os.path.join(DATA_DIR, 'test_balanced.csv')
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}. Please upload to {DATA_DIR}")
    try:
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_balanced.csv'))
        val_df = pd.read_csv(os.path.join(DATA_DIR, 'val_balanced.csv'))
        test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_balanced.csv'))
    except FileNotFoundError as e:
        print(f"Error: Data file not found at {DATA_DIR}. {e}")
        raise
    train_df_sentiment = augment_sentiment_data_fast(train_df.copy())
    train_df_sentiment['full_text_cleaned'] = preprocess_text(train_df_sentiment['full_text'])
    train_df_packaging = train_df.copy()
    train_df_packaging['full_text_cleaned'] = preprocess_text(train_df_packaging['full_text'])
    val_df['full_text_cleaned'] = preprocess_text(val_df['full_text'])
    test_df['full_text_cleaned'] = preprocess_text(test_df['full_text'])
    return train_df_sentiment, train_df_packaging, val_df, test_df

# --- Section 4: Model Training Functions ---
def train_packaging_model(
    train_texts: pd.Series, train_labels: pd.Series
) -> Pipeline:
    """Train TF-IDF + Logistic Regression model for packaging classification."""
    print("\n--- Training Packaging Classifier (TF-IDF + Logistic Regression) ---")
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english')),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED))
    ])
    param_grid = {
        'vectorizer__max_features': [15000, 25000],
        'vectorizer__ngram_range': [(1, 2)],
        'classifier__C': [1.0, 10.0]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(train_texts, train_labels)
    print(f"Best TF-IDF + LR params for packaging: {grid_search.best_params_}")
    if wandb.run:
        wandb.log({"packaging_best_params": grid_search.best_params_})
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_path = f'{MODELS_DIR}/packaging_tfidf_lr_model_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        joblib.dump(grid_search.best_estimator_, f)
    print(f"Packaging model saved to {model_path}")
    return grid_search.best_estimator_

def compute_metrics(eval_pred):
    """Compute accuracy for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"eval_accuracy": accuracy}

def fine_tune_deberta(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    label_map: Dict[str, int]
) -> Trainer:
    """Fine-tunes the DeBERTa-v3 model for sentiment classification."""
    print("\n--- Fine-tuning DeBERTa-v3 for Sentiment Analysis ---")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(label_map)
    )
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=['query_proj', 'value_proj'],
        lora_dropout=0.1, bias="none", task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    training_args = TrainingArguments(
        output_dir=f'{MODELS_DIR}/sentiment_deberta_lora_temp',
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        logging_steps=50,
        report_to="wandb" if wandb.run else "none",
        fp16=torch.cuda.is_available(),
        seed=RANDOM_SEED
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    trainer.train()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    final_model_path = f'{MODELS_DIR}/sentiment_deberta_lora_{timestamp}'
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Sentiment model saved to Google Drive at {final_model_path}")
    try:
        if HF_USERNAME != "YOUR_HF_USERNAME_HERE":
            hf_repo_name = f"{HF_USERNAME}/deberta-packaging-sentiment"
            model.push_to_hub(hf_repo_name, token=userdata.get('HF_TOKEN') if IS_COLAB else None)
            tokenizer.push_to_hub(hf_repo_name, token=userdata.get('HF_TOKEN') if IS_COLAB else None)
            print(f"Sentiment model saved to HF Hub: {hf_repo_name}")
        else:
            print("HF_USERNAME not set. Skipping Hugging Face Hub push.")
    except Exception as e:
        print(f"Could not save sentiment model to HF Hub: {e}")
    return trainer

# --- Section 5: Evaluation Function ---
def evaluate_model(predictions: list, true_labels: pd.Series,
                   label_map: Dict, task_name: str):
    """Evaluate a model with a detailed per-class breakdown."""
    print(f"\n--- Evaluating {task_name.upper()} Model ---")
    accuracy = accuracy_score(true_labels, predictions)
    sorted_labels = sorted(list(label_map.keys()), key=lambda x: label_map[str(x)])
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, labels=sorted_labels, average=None
    )
    metrics_per_class = {label: {'precision': p, 'recall': r, 'f1-score': f}
                         for label, p, r, f in zip(sorted_labels, precision, recall, f1)}
    print(f"Model achieved an accuracy of {accuracy * 100:.2f}% on the test dataset.")
    print("\nPrecision, recall, and F1-score for each class are as follows:")
    for class_name, metrics in metrics_per_class.items():
        p, r, f = metrics['precision'] * 100, metrics['recall'] * 100, metrics['f1-score'] * 100
        print(f" Class '{class_name}': Precision={p:.2f}%, Recall={r:.2f}%, F1-score={f:.2f}%")
    if wandb.run:
        wandb.log({f"{task_name}_test_accuracy": accuracy})
    cm = confusion_matrix(true_labels, predictions, labels=sorted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted_labels, yticklabels=sorted_labels)
    plt.title(f'{task_name.capitalize()} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = f'{REPORTS_DIR}/{task_name}_confusion_matrix_{timestamp}.png'
    plt.savefig(output_path)
    plt.show()
    plt.close()
    if wandb.run:
        wandb.log({f"{task_name}_confusion_matrix": wandb.Image(output_path)})
    metrics_df = pd.DataFrame(metrics_per_class).T
    metrics_df.to_csv(f'{REPORTS_DIR}/{task_name}_metrics_{timestamp}.csv')
    print(f"Metrics saved to {REPORTS_DIR}/{task_name}_metrics_{timestamp}.csv")

# --- Section 6: Inference Functions ---
def load_and_predict_packaging(texts: pd.Series, model_path: str) -> list:
    """Load the saved packaging model and make predictions."""
    try:
        with open(model_path, 'rb') as f:
            model = joblib.load(f)
        return model.predict(texts)
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        raise
    except Exception as e:
        print(f"Error loading or predicting with packaging model: {e}")
        raise

def load_and_predict_sentiment(texts: pd.Series, model_path: str, label_map: Dict[str, int]) -> list:
    """Load the saved sentiment model and make predictions."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**{k: v.to(model.device) for k, v in encodings.items()})
        pred_indices = np.argmax(outputs.logits.cpu().numpy(), axis=-1)
        inv_map = {v: k for k, v in label_map.items()}
        return [inv_map[p] for p in pred_indices]
    except Exception as e:
        print(f"Error loading or predicting with sentiment model: {e}")
        raise

# --- Section 7: Main Orchestration ---
def main():
    """Orchestrate the entire classification pipeline for both tasks."""
    run = None
    try:
        if IS_COLAB:
            wandb.login(key=userdata.get('WANDB_API_KEY'))
            run = wandb.init(project="velana-packaging-combined",
                             name=f"full_run_{datetime.now().strftime('%H%M')}")
            login(token=userdata.get('HF_TOKEN'))
        else:
            print("Not in Colab. Skipping wandb and Hugging Face initialization.")
    except (ValueError, TypeError) as e:
        print(f"Initialization Error: Check Colab secrets for WANDB_API_KEY and HF_TOKEN. {e}")
        run = None
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    setup_nltk()
    if IS_COLAB:
        try:
            mount_point = '/content/drive'
            # Create a temporary mount point to avoid conflicts
            temp_mount = '/content/drive_temp'
            if os.path.exists(mount_point) and not os.path.ismount(mount_point):
                shutil.rmtree(mount_point, ignore_errors=True)
                os.makedirs(mount_point, exist_ok=True)
            if not os.path.ismount(mount_point):
                print("Mounting Google Drive...")
                drive.mount(mount_point)
                print("Google Drive mounted successfully.")
            else:
                print("Google Drive already mounted.")
            # Verify data directory and files
            required_files = [
                os.path.join(DATA_DIR, 'train_balanced.csv'),
                os.path.join(DATA_DIR, 'val_balanced.csv'),
                os.path.join(DATA_DIR, 'test_balanced.csv')
            ]
            for file_path in required_files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Data file not found: {file_path}. Please upload to {DATA_DIR}")
        except Exception as e:
            print(f"Failed to mount Google Drive or access data directory: {e}")
            raise
    train_df_sent, train_df_pack, val_df, test_df = load_data()
    # --- Task 1: Packaging Model ---
    packaging_model = train_packaging_model(
        train_df_pack['full_text_cleaned'], train_df_pack['is_packaging_issue'].astype(str)
    )
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    packaging_model_path = f'{MODELS_DIR}/packaging_tfidf_lr_model_{timestamp}.pkl'
    with open(packaging_model_path, 'wb') as f:
        joblib.dump(packaging_model, f)
    packaging_preds = load_and_predict_packaging(test_df['full_text_cleaned'], packaging_model_path)
    packaging_label_map = {'0': 0, '1': 1}
    evaluate_model(packaging_preds, test_df['is_packaging_issue'].astype(str),
                   packaging_label_map, 'packaging_tfidf_lr')
    # --- Task 2: Sentiment Model ---
    sentiment_label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    def tokenize_data(texts, labels):
        encodings = tokenizer(
            texts.tolist(), truncation=True, padding=True, max_length=MAX_SEQ_LENGTH
        )
        encodings['labels'] = [sentiment_label_map[label] for label in labels]
        return encodings
    train_dataset = HFDataset(tokenize_data(
        train_df_sent['full_text_cleaned'], train_df_sent['sentiment_label']
    ))
    val_dataset = HFDataset(tokenize_data(
        val_df['full_text_cleaned'], val_df['sentiment_label']
    ))
    deberta_trainer = fine_tune_deberta(
        train_dataset, val_dataset, sentiment_label_map
    )
    # Evaluate the sentiment model
    raw_preds = deberta_trainer.predict(HFDataset(tokenize_data(
        test_df['full_text_cleaned'], test_df['sentiment_label']
    )))
    pred_indices = np.argmax(raw_preds.predictions, axis=-1)
    inv_map = {v: k for k, v in sentiment_label_map.items()}
    sentiment_preds = [inv_map[p] for p in pred_indices]
    evaluate_model(sentiment_preds, test_df['sentiment_label'],
                   sentiment_label_map, 'deberta_sentiment')
    if run:
        run.finish()
    print("\n✅ Combined pipeline completed successfully.")

if __name__ == "__main__":
    main()