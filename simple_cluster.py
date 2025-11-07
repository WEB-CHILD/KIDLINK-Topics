#!/usr/bin/env python3
"""
Simple K-means clustering for solrwayback_kidlink-org-dk.csv

Usage:
    python simple_cluster.py
"""
import pandas as pd
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk.corpus import stopwords

# 1. Load CSV
print("Step 1: Loading CSV file...")
df = pd.read_csv("solrwayback_kidlink-org-dk.csv")
documents = df['content'].dropna().tolist()
print(f"✓ Loaded {len(documents)} documents\n")

# 2. Remove stopwords (Danish, Norwegian, English, Spanish)
print("Step 2: Removing stopwords from documents...")
# Ensure NLTK stopwords are available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Build combined stopword set
danish_sw = set(stopwords.words("danish"))
norwegian_sw = set(stopwords.words("norwegian"))
spanish_sw = set(stopwords.words("spanish"))
english_sw = set(stopwords.words("english"))
german_sw = set(stopwords.words("german"))
italian_sw = set(stopwords.words("italian"))
portuguese_sw = set(stopwords.words("portuguese"))
combined_stopwords = set(w.lower() for w in (ENGLISH_STOP_WORDS | english_sw | danish_sw | norwegian_sw | spanish_sw | german_sw | italian_sw | portuguese_sw))

def remove_stopwords(text):
    """Remove stopwords from text, tokenizing on whitespace."""
    if not isinstance(text, str):
        return ""
    tokens = [t.lower() for t in text.split() if t.lower() and t.lower() not in combined_stopwords]
    filtered = " ".join(tokens)
    return filtered if filtered.strip() else text

documents_filtered = [remove_stopwords(doc) for doc in documents]
print(f"✓ Stopwords removed\n")

# 3. Create embeddings
print("Step 3: Creating embeddings with SentenceTransformer...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents_filtered, show_progress_bar=True, batch_size=64, convert_to_numpy=True)
print(f"✓ Created embeddings with shape {embeddings.shape}\n")

# 4. Cluster with K-means
num_clusters = 20
print(f"Step 4: Clustering documents into {num_clusters} clusters...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)
print(f"✓ Clustering complete\n")

# 5. Extract top keywords per cluster using TF-IDF
print("Step 5: Extracting keywords for each cluster...")
df_clustered = pd.DataFrame({"document": documents, "cluster": labels})

def get_keywords(docs, n=10):
    """Extract top n TF-IDF keywords from document list."""
    if not docs:
        return []
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(docs)
    if X.shape[1] == 0:
        return []
    scores = X.sum(axis=0).A1
    indices = np.argsort(scores)[::-1][:n]
    return np.array(vectorizer.get_feature_names_out())[indices].tolist()

# 6. Print results
print("✓ Keyword extraction complete\n")
print("Step 6: Printing cluster results...")
print("\n" + "="*70)
for cluster_id in range(num_clusters):
    cluster_docs = df_clustered[df_clustered.cluster == cluster_id]['document'].tolist()
    keywords = get_keywords(cluster_docs, n=20)
    print(f"\nCluster {cluster_id} ({len(cluster_docs)} docs)")
    print(f"Keywords: {', '.join(keywords)}")
    if cluster_docs:
        sample = str(cluster_docs[0])[:150].replace('\n', ' ')
        print(f"Sample: {sample}...")
    print("-" * 70)
