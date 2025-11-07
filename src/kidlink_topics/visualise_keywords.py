#!/usr/bin/env python3
"""
Visualize cluster keywords from cluster_keywords.json

Usage:
    python visualise_keywords.py
"""
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

print("Loading cluster keywords...")
with open("data/cluster_keywords.json", 'r', encoding='utf-8') as f:
    cluster_data = json.load(f)
print(f"✓ Loaded {len(cluster_data)} clusters\n")

# Create output directory structure
import os
os.makedirs('visualisations/wordclouds', exist_ok=True)

# 1. Generate wordclouds for each cluster
print("Generating wordclouds...")
num_clusters = len(cluster_data)
cols = 4
rows = (num_clusters + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
axes = axes.flatten() if num_clusters > 1 else [axes]

for i, cluster in enumerate(cluster_data):
    # Create word frequency dict from keywords (higher index = higher weight)
    keywords = cluster['keywords']
    word_freq = {word: len(keywords) - idx for idx, word in enumerate(keywords)}
    
    # Generate wordcloud
    wc = WordCloud(width=400, height=300, background_color='white', 
                   colormap='viridis', relative_scaling=0.5).generate_from_frequencies(word_freq)
    
    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].set_title(f"Cluster {cluster['cluster_id']} ({cluster['num_docs']} docs)", 
                      fontsize=10, fontweight='bold')
    axes[i].axis('off')

# Hide unused subplots
for i in range(num_clusters, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('visualisations/wordclouds_all_clusters.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved visualisations/wordclouds_all_clusters.png\n")
plt.close()

# 2. Top keywords bar chart (top 5 keywords from each cluster)
print("Generating top keywords overview...")
fig, ax = plt.subplots(figsize=(14, 8))

cluster_ids = [c['cluster_id'] for c in cluster_data]
top_keywords_per_cluster = [', '.join(c['keywords'][:5]) for c in cluster_data]

y_pos = np.arange(len(cluster_ids))
ax.barh(y_pos, [c['num_docs'] for c in cluster_data], color='steelblue')
ax.set_yticks(y_pos)
ax.set_yticklabels([f"C{cid}: {kw}" for cid, kw in zip(cluster_ids, top_keywords_per_cluster)], 
                    fontsize=8)
ax.set_xlabel('Number of Documents')
ax.set_title('Cluster Sizes and Top 5 Keywords')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('visualisations/cluster_overview.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved visualisations/cluster_overview.png\n")
plt.close()

# 3. Individual wordclouds (save separately for easier viewing)
print("Generating individual wordclouds...")

for cluster in cluster_data:
    keywords = cluster['keywords']
    word_freq = {word: len(keywords) - idx for idx, word in enumerate(keywords)}
    
    wc = WordCloud(width=800, height=600, background_color='white',
                   colormap='viridis', relative_scaling=0.5).generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(10, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f"Cluster {cluster['cluster_id']} - {cluster['num_docs']} documents", 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"visualisations/wordclouds/cluster_{cluster['cluster_id']:02d}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()

print(f"✓ Saved individual wordclouds to visualisations/wordclouds/ directory\n")
print("Done! Generated:")
print("  - visualisations/wordclouds_all_clusters.png (overview)")
print("  - visualisations/cluster_overview.png (bar chart)")
print("  - visualisations/wordclouds/cluster_XX.png (individual)")