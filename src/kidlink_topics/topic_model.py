#!/usr/bin/env python3
"""
Multilingual topic modeling using BERTopic

This model is multilingual and aims at producing topics across languages.


Usage:
    python topic_model.py [CSV_PATH] [--domain-prefix PREFIX]
    
    CSV_PATH: Path to the CSV file (default: data/docs.csv)
    --domain-prefix PREFIX: Optional domain prefix for output filenames
                           (e.g., "kidlink_org_dk"). If not provided,
                           outputs use default names.

Required CSV columns:
- `content`: The text content of each document (string). Rows with empty `content` are dropped.
- `id` (optional): A unique identifier for each document. If absent, the script will use the CSV row index as the document ID.

Example CSV header:
    id,content

Examples:
    python topic_model.py
    python topic_model.py data/docs.csv
    python topic_model.py data/docs.csv --domain-prefix kidlink_org_dk
"""
import os
import sys
import shutil
import argparse
import pandas as pd
import numpy as np
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from utils import load_csv, load_custom_stopwords, remove_stopwords, save_array_to_json, get_device, get_output_paths

MIN_DOCUMENTS_PR_TOPIC = 80  # Minimum documents for a created topic
AMOUNT_OF_KEYWORDS_PR_TOPIC = 50  # Number of keywords to extract per topic

# Detect and use best available device (CUDA for NVIDIA, MPS for Apple Silicon, CPU fallback)
device = get_device()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Multilingual topic modeling using BERTopic")
parser.add_argument("csv_path", nargs="?", default="data/docs.csv", help="Path to the CSV file (default: data/docs.csv)")
parser.add_argument("--domain-prefix", type=str, default=None, help="Optional domain prefix for output filenames (e.g., 'kidlink_org_dk')")
args = parser.parse_args()

csv_path = args.csv_path
domain_prefix = args.domain_prefix

# 1. Load CSV
documents, doc_ids = load_csv(csv_path)

# 2. Remove stopwords (multilingual)
print("Step 2: Removing stopwords from documents...")
combined_stopwords = load_custom_stopwords()
documents_filtered = [remove_stopwords(doc, combined_stopwords) for doc in documents]
print(f"✓ Stopwords removed\n")

# Calculate adaptive min_df based on dataset size
# Small datasets get lower threshold to avoid excluding too many words
# Large datasets get higher threshold to filter noise
num_docs = len(documents_filtered)
# Use percentage-based min_df (0.1% to 1%) so it scales to document subsets too
min_df_percentage = max(0.001, min(0.01, 10 / num_docs)) if num_docs > 0 else 0.001
print(f"Dataset size: {num_docs} documents")
print(f"Adaptive min_df: {min_df_percentage*100:.2f}% (words must appear in this % of documents)\n")

# 3. Create multilingual embedding model
print("Step 3: Loading multilingual embedding model...")
# Use multilingual model that supports Danish, Norwegian, English, Spanish, etc.
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
print(f"✓ Model loaded on {device}\n")

# 4. Create BERTopic model with custom settings
print("Step 4: Configuring BERTopic model...")

# Custom vectorizer with multilingual stopwords
vectorizer_model = CountVectorizer(
    stop_words=list(combined_stopwords),
    max_features=5000,
    ngram_range=(1, 2),  # Include bigrams for better topics
    min_df=min_df_percentage,  # Percentage-based: scales to document subsets
    max_df=0.95  # Exclude words in >95% of documents (likely stopwords we missed)
)

# Initialize BERTopic with custom settings
topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    min_topic_size=MIN_DOCUMENTS_PR_TOPIC,  # Minimum documents per topic
    nr_topics="auto",  # Automatically determine number of topics
    calculate_probabilities=False,  # Faster without probabilities
    top_n_words=AMOUNT_OF_KEYWORDS_PR_TOPIC,  # Generate 50 keywords per topic
    verbose=True
)
print(f"✓ BERTopic model configured\n")

# 5. Fit topic model
print("Step 5: Fitting topic model (this may take a while)...")
topics, probabilities = topic_model.fit_transform(documents_filtered)
print(f"✓ Topic modeling complete\n")

# 6. Extract topic information
print("Step 6: Extracting topic information...")
topic_info = topic_model.get_topic_info()
print(f"✓ Found {len(topic_info) - 1} topics (excluding outliers)\n")

# 7. Create detailed topic data for saving
print("Step 7: Preparing topic data for export...")
topic_data = []

for idx, row in topic_info.iterrows():
    topic_id = row['Topic']
    
    # Skip outlier topic (-1)
    if topic_id == -1:
        continue
    
    # Get top words for this topic
    topic_words = topic_model.get_topic(topic_id)
    if topic_words:
        keywords = [word for word, score in topic_words[:AMOUNT_OF_KEYWORDS_PR_TOPIC]]
        keyword_scores = {word: float(score) for word, score in topic_words[:AMOUNT_OF_KEYWORDS_PR_TOPIC]}
    else:
        keywords = []
        keyword_scores = {}
    
    # Get representative documents for this topic
    topic_docs_indices = np.where(np.array(topics) == topic_id)[0]
    num_docs = len(topic_docs_indices)
    
    # Get sample document (first one in topic)
    if num_docs > 0:
        sample_doc = documents[topic_docs_indices[0]]
        sample = str(sample_doc)[:200].replace('\n', ' ') + "..."
    else:
        sample = ""
    
    # Get up to 80 document IDs for this topic from the CSV's 'id' column
    # These can be used to look up original documents in the CSV later
    document_ids = [doc_ids[idx] for idx in topic_docs_indices[:80]]
    
    topic_entry = {
        'topic_id': int(topic_id),
        'num_docs': int(num_docs),
        'keywords': keywords,
        'keyword_scores': keyword_scores,
        'name': row.get('Name', f"Topic {topic_id}"),
        'sample': sample,
        'document_ids': document_ids
    }
    topic_data.append(topic_entry)

# Sort by number of documents (descending)
topic_data.sort(key=lambda x: x['num_docs'], reverse=True)

print(f"✓ Prepared {len(topic_data)} topics\n")

# 8. Print results
print("\n" + "="*70)
print("TOPIC MODEL RESULTS")
print("="*70)

for topic in topic_data[:10]:  # Print top 10 topics
    print(f"\nTopic {topic['topic_id']}: {topic['name']}")
    print(f"Documents: {topic['num_docs']}")
    print(f"Keywords: {', '.join(topic['keywords'][:AMOUNT_OF_KEYWORDS_PR_TOPIC])}")
    print(f"Sample: {topic['sample']}")
    print("-" * 70)

# 9. Save to file
output_file, model_path = get_output_paths(domain_prefix)
save_array_to_json(topic_data, output_file)

print(f"\n✓ Full results saved to {output_file}")

# 10. Save the model for later use
print("\nStep 8: Saving topic model...")
# Create models directory if it doesn't exist
os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
# Remove existing model if it exists to avoid conflicts
if os.path.exists(model_path):
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)
    else:
        os.remove(model_path)
    print(f"⚠ Removed existing model: {model_path}")
topic_model.save(model_path)
print(f"✓ Model saved to {model_path}/")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total documents: {len(documents)}")
print(f"Topics found: {len(topic_data)}")
print(f"Outliers: {len([t for t in topics if t == -1])}")
print(f"\nOutput files:")
print(f"  - {output_file} (topic data)")
print(f"  - topic_model_bertopic/ (saved model)")
