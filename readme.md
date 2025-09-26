Beyond the Product: A Data-Driven Approach to Packaging Intelligence
This repository contains an end-to-end NLP pipeline designed to analyze customer feedback on product packaging from Amazon's "Health and Personal Care" category. By moving beyond traditional product-focused analysis, this project provides actionable insights into how packaging impacts customer satisfaction, brand perception, and operational efficiency.

The system automates the extraction, classification, and summarization of packaging-related feedback, culminating in a structured dataset ready for business intelligence and a deployment-ready summarization module.

📈 Business Case & Impact
In e-commerce, packaging is the first physical touchpoint a customer has with a brand. Poor packaging can lead to damaged products, negative reviews, and costly returns. Conversely, effective packaging enhances the customer experience and builds brand loyalty.

This project addresses key business questions:

Problem Identification: What are the most common packaging issues (e.g., leaks, damage, tampering) our customers face?

Product-Specific Insights: Which products or product categories are most affected by packaging problems?

Sentiment Analysis: How do packaging issues correlate with negative customer sentiment?

Root Cause Analysis: Can we identify thematic clusters of complaints (e.g., "leaking pump bottles for home goods") to guide targeted improvements?

By answering these questions, businesses can make data-driven decisions to:

Reduce Costs: Minimize returns and replacements due to shipping damage.

Improve Customer Satisfaction: Proactively address packaging flaws to improve ratings and reviews.

Optimize Logistics: Inform packaging design choices for specific product lines.

Enhance Brand Image: Deliver a consistently positive and reliable unboxing experience.

🛠️ Technological Pipeline Overview
The project is structured as a sequential pipeline, where the output of each stage serves as the input for the next. This modular design ensures maintainability and scalability.

Data Preprocessing (preprocessing.py): Raw JSONL review data is cleaned, transformed, and balanced to create structured CSV datasets for modeling.

Multi-Task Modeling (modeling1.py, modeling2.py): Two specialized models are trained: one to identify packaging-related reviews and another to analyze sentiment.

Thematic Clustering (clustering.py): Packaging-specific reviews are clustered to uncover high-level themes and correlations between issues, packaging types, and product categories.

AI-Powered Summarization (summarization.py): A fine-tuned LLM distills the clustered insights into a structured JSON object and visualizations for quick analysis.

Dashboard Deployment (Streamlit): The final outputs are consumed by an interactive Streamlit dashboard for stakeholder exploration.

🔬 Pipeline Stages in Detail
1. Data Preprocessing
This initial stage focuses on transforming raw, unstructured review data into a clean, feature-rich format suitable for machine learning.

Input: Raw Health_and_Personal_Care.jsonl and meta_Health_and_Personal_Care.jsonl datasets.

Key Operations:

Merging: Combines review text with product metadata.

Text Cleaning: Removes emojis, handles null values, and combines the title and review body into a full_text field.

Feature Engineering:

sentiment_label: Maps star ratings (1-5) to categorical sentiment (negative, neutral, positive).

is_packaging_issue: Uses a regex pattern with keywords like leak, damaged, box, and seal to perform an initial flagging of packaging-related reviews.

Data Balancing & Splitting: Applies undersampling to create balanced training, validation, and test sets (train_balanced.csv, val_balanced.csv, test_balanced.csv) to prevent model bias.

Output:

Balanced datasets for model training.

A dedicated packaging_subset.csv containing all reviews flagged as packaging-related, which serves as the primary input for the clustering and summarization stages.

2. Multi-Task Modeling
To address two distinct business needs, we train two separate, optimized models. This approach ensures higher accuracy than a single, multi-label model.

Task 1: Packaging Issue Detection
Objective: Classify whether a review is about packaging or not.

Model: TF-IDF Vectorizer + Logistic Regression.

Rationale: This lightweight, interpretable model is highly effective for binary classification where the presence of specific keywords is a strong signal. Its speed is ideal for filtering large volumes of reviews.

Output: A serialized Scikit-learn pipeline (packaging_tfidf_lr_model_*.pkl) used in subsequent steps to accurately identify relevant reviews.

Task 2: Sentiment Analysis
Objective: Perform a nuanced, three-class sentiment classification (positive, neutral, negative).

Model: Fine-tuned microsoft/deberta-v3-base with LoRA (Low-Rank Adaptation).

Rationale: DeBERTa-v3 offers state-of-the-art performance in understanding context and sentiment. LoRA enables efficient fine-tuning on consumer hardware by updating only a small subset of the model's weights.

Training:

Utilizes nlpaug for data augmentation to bolster the minority classes (neutral, negative).

Integrates with wandb for experiment tracking and logging of metrics.

Output: A highly accurate sentiment classification model, saved to Google Drive and ready for deployment or sharing on the Hugging Face Hub.

3. Packaging-Focused Clustering
This stage moves beyond classification to discover why customers are talking about packaging.

Objective: Group packaging reviews into 4-6 distinct, interpretable "meta-categories."

Process:

Filtering: Uses the trained TF-IDF model to select only high-confidence packaging reviews from the dataset.

Embedding: Generates contextual vector embeddings for each review using the all-mpnet-base-v2 SentenceTransformer model.

Topic Modeling: Employs BERTopic, which leverages UMAP for dimensionality reduction and HDBSCAN for density-based clustering, to identify thematic groups.

Categorization & Correlation: Tags each review based on predefined keyword dictionaries for packaging issues, types, and product categories. It then computes Chi-square tests to find statistically significant correlations between the discovered clusters and these tags.

Output:

Meta-Categories: Generates descriptive cluster labels like "Leaking Bottles & Jars for Home Goods" by combining the most correlated tags.

Reports: Saves correlation heatmaps, cluster purity scores, and t-SNE visualizations.

Structured Data: Outputs clustered_packaging_*.csv containing the original reviews enriched with cluster IDs and category tags.

4. AI-Powered Summarization
The final analysis step synthesizes the clustered findings into a format that is immediately usable by business stakeholders.

Objective: Create a concise, quantitative summary of packaging feedback.

Process:

Sampling: Selects a representative sample of 50 reviews from the clustered data to ensure efficient processing and avoid LLM rate limits.

Prompt Engineering: Feeds the clustered reviews into a prompt designed for a fine-tuned GPT model. The prompt explicitly requests a structured JSON output. A fallback to gpt-4o is implemented for robustness.

Structured Output Generation: The model generates a JSON object with primary keys (packaging_issues, packaging_types, product_categories), where each entry contains a summary, count, and percentage.

Visualization: Automatically generates and saves bar charts for each primary key, providing a quick visual overview of the findings.

Output:

summary_finetuned.json: A clean, structured summary file.

*.png chart images for easy inclusion in reports and presentations.

5. Deployment via Streamlit
The final, synthesized insights from this pipeline are made accessible to business stakeholders through an interactive web application.

Framework: Streamlit

Audience: Designed for business users at Velana.net, requiring no technical expertise.

Features: The dashboard allows users to:

Visualize key trends from the summarization charts.

Filter data by product category, issue type, or sentiment.

Drill down into specific thematic clusters to understand nuanced feedback.

View example reviews within each cluster to see the raw customer voice.

Access: The live dashboard is available at: https://velana-packaging-dashboard.streamlit.app/

🚀 Setup & Execution
This project is designed to be run in a Google Colab environment.

Prerequisites
A Google account with access to Google Drive.

Python 3.9+.

API keys for OpenAI and Weights & Biases (W&B) stored as secrets in Colab (OPENAI_API_KEY, WANDB_API_KEY).

Directory Structure
Before running, ensure your Google Drive has the following structure:

/MyDrive/
├── data/
│   ├── Health_and_Personal_Care.jsonl
│   └── meta_Health_and_Personal_Care.jsonl
├── models/
└── reports/

Execution Order
Run the scripts sequentially as notebooks in Google Colab:

preprocessing.py: Generates the initial CSV files in /content/data/.

modeling1.py or modeling2.py: Trains the classification models and saves them to /MyDrive/models/.

clustering.py: Reads the data, uses the packaging model, and saves cluster reports to /MyDrive/reports/.

summarization.py: Manually upload packaging_subset.csv and the .pkl model to the Colab /content/ directory. The script will process them and trigger automatic downloads of the final JSON and PNG reports.

🔮 Future Work
Real-Time Analysis: Integrate the pipeline with a data stream (e.g., from a customer support platform) for continuous monitoring of packaging issues.

Aspect-Based Sentiment: Enhance the sentiment model to perform aspect-based analysis, determining if sentiment is directed at the product itself versus the packaging.

Multimodal Analysis: Incorporate user-submitted images of damaged packaging to add a visual dimension to the analysis.
