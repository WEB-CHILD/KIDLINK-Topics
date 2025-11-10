#!/usr/bin/env python3
"""
Comprehensive visualization script for BERTopic model results.

This script generates multiple visualizations from topic modeling results stored in JSON format.
It creates visual representations to help understand topic distributions, keyword importance, 
and relationships between topics and their representative terms.

Visualizations Generated:
1. Word clouds for all topics (grid layout). Quite a heavy plot.
   - Combined overview showing all topics in one figure
   - Uses word frequency or scores to size terms appropriately

2. Topic overview bar chart
   - Horizontal bar chart showing document counts per topic
   - Displays top 5 keywords for each topic
   - Helpful for understanding relative topic sizes

3. Topic distribution pie chart
   - Shows proportional distribution of top 10 topics
   - Groups remaining topics into "Other" category
   - Provides percentage breakdown of document assignments

4. Individual high-resolution word clouds
   - Separate PNG file for each topic
   - Higher quality for detailed examination
   - Saved to visualisations/topics/wordclouds/ directory

5. Keyword importance heatmap
   - Matrix visualization of top 10 topics × top 10 keywords
   - Color intensity represents keyword importance/score
   - Useful for comparing keyword distributions across topics

Input:
    Expects JSON file at: data/topic_model_results_80_docs_50_keywords.json
    Format: Array of topic objects with fields:
        - topic_id: Integer topic identifier
        - num_docs: Number of documents in topic
        - keywords: List of keyword strings
        - keyword_scores: Optional dict of {keyword: score}
        - name: Optional topic name/label

Output:
    All visualizations saved to visualisations/topics/ directory:
    - wordclouds_all_topics.png
    - topic_overview.png
    - topic_distribution.png
    - keyword_heatmap.png
    - wordclouds/topic_XX.png (one per topic)

Usage:
    python visualise_topics.py
"""
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import os

print("Loading topic model results...")
with open("data/topic_model_results_80_docs_50_keys_more_stops.json", 'r', encoding='utf-8') as f:
    topic_data = json.load(f)
print(f"✓ Loaded {len(topic_data)} topics\n")

# Create output directory structure
os.makedirs('visualisations/topics/wordclouds', exist_ok=True)

# 1. Generate wordclouds for each topic
print("Generating topic wordclouds...")
num_topics = len(topic_data)
cols = 4
rows = (num_topics + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
axes = axes.flatten() if num_topics > 1 else [axes]

for i, topic in enumerate(topic_data):
    # Use keyword scores if available, otherwise use rank-based weights
    if topic.get('keyword_scores'):
        word_freq = topic['keyword_scores']
    else:
        keywords = topic['keywords']
        word_freq = {word: len(keywords) - idx for idx, word in enumerate(keywords)}
    
    # Generate wordcloud
    wc = WordCloud(width=400, height=300, background_color='white', 
                   colormap='plasma', relative_scaling=0.5).generate_from_frequencies(word_freq)
    
    axes[i].imshow(wc, interpolation='bilinear')
    topic_name = topic.get('name', f"Topic {topic['topic_id']}")
    axes[i].set_title(f"{topic_name}\n({topic['num_docs']} docs)", 
                      fontsize=9, fontweight='bold')
    axes[i].axis('off')

# Hide unused subplots
for i in range(num_topics, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('visualisations/topics/wordclouds_all_topics.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved visualisations/topics/wordclouds_all_topics.png\n")
plt.close()

# 2. Topic sizes bar chart with top keywords
print("Generating topic overview...")
fig, ax = plt.subplots(figsize=(14, max(8, num_topics * 0.4)))

topic_ids = [t['topic_id'] for t in topic_data]
topic_names = [t.get('name', f"Topic {t['topic_id']}") for t in topic_data]
top_keywords = [', '.join(t['keywords'][:5]) for t in topic_data]

y_pos = np.arange(len(topic_ids))
colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(topic_ids)))

ax.barh(y_pos, [t['num_docs'] for t in topic_data], color=colors)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"T{tid}: {kw}" for tid, kw in zip(topic_ids, top_keywords)], 
                    fontsize=8)
ax.set_xlabel('Number of Documents', fontsize=10)
ax.set_title('Topic Sizes and Top 5 Keywords', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('visualisations/topics/topic_overview.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved visualisations/topics/topic_overview.png\n")
plt.close()

# 3. Topic distribution pie chart (top 10 topics)
print("Generating topic distribution chart...")
top_n = min(10, len(topic_data))
top_topics = topic_data[:top_n]
other_docs = sum(t['num_docs'] for t in topic_data[top_n:])

labels = [f"T{t['topic_id']}: {', '.join(t['keywords'][:2])}" for t in top_topics]
sizes = [t['num_docs'] for t in top_topics]

if other_docs > 0:
    labels.append(f"Other ({len(topic_data) - top_n} topics)")
    sizes.append(other_docs)

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                    colors=colors, startangle=90)

# Improve label readability
for text in texts:
    text.set_fontsize(9)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)

ax.set_title('Topic Distribution (Top 10 Topics)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualisations/topics/topic_distribution.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved visualisations/topics/topic_distribution.png\n")
plt.close()

# 4. Individual high-resolution wordclouds
print("Generating individual topic wordclouds...")

for topic in topic_data:
    # Use keyword scores if available
    if topic.get('keyword_scores'):
        word_freq = topic['keyword_scores']
    else:
        keywords = topic['keywords']
        word_freq = {word: len(keywords) - idx for idx, word in enumerate(keywords)}
    
    wc = WordCloud(width=800, height=600, background_color='white',
                   colormap='plasma', relative_scaling=0.5).generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(10, 7))
    plt.imshow(wc, interpolation='bilinear')
    topic_name = topic.get('name', f"Topic {topic['topic_id']}")
    plt.title(f"{topic_name} - {topic['num_docs']} documents", 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"visualisations/topics/wordclouds/topic_{topic['topic_id']:02d}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()

print(f"✓ Saved individual wordclouds to visualisations/topics/wordclouds/ directory\n")

# 5. Keyword importance heatmap (top 10 topics, top 10 keywords each)
print("Generating keyword importance heatmap...")
top_n_topics = min(10, len(topic_data))
top_topics = topic_data[:top_n_topics]

# Build matrix of keyword scores
max_keywords = 10
topic_labels = []
keyword_matrix = []

for topic in top_topics:
    topic_name = topic.get('name', f"T{topic['topic_id']}")
    topic_labels.append(topic_name[:30])  # Truncate long names
    
    if topic.get('keyword_scores'):
        # Use actual scores
        scores = list(topic['keyword_scores'].values())[:max_keywords]
        # Pad if fewer than max_keywords
        scores.extend([0] * (max_keywords - len(scores)))
    else:
        # Use rank-based weights
        num_kw = min(len(topic['keywords']), max_keywords)
        scores = list(range(num_kw, 0, -1))
        scores.extend([0] * (max_keywords - num_kw))
    
    keyword_matrix.append(scores)

keyword_matrix = np.array(keyword_matrix)

# Get keyword labels from first topic
all_keywords = []
for topic in top_topics:
    all_keywords.extend(topic['keywords'][:max_keywords])
unique_keywords = list(dict.fromkeys(all_keywords))[:max_keywords]  # Preserve order, remove duplicates

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(keyword_matrix, cmap='YlOrRd', aspect='auto')

# Set ticks and labels
ax.set_xticks(np.arange(max_keywords))
ax.set_yticks(np.arange(len(topic_labels)))
ax.set_xticklabels([f"KW{i+1}" for i in range(max_keywords)], fontsize=9)
ax.set_yticklabels(topic_labels, fontsize=9)

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Keyword Importance", rotation=-90, va="bottom")

ax.set_title("Top 10 Topics - Keyword Importance Heatmap", fontsize=12, fontweight='bold')
fig.tight_layout()
plt.savefig('visualisations/topics/keyword_heatmap.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved visualisations/topics/keyword_heatmap.png\n")
plt.close()

# Summary
print("=" * 70)
print("Done! Generated:")
print("  - visualisations/topics/wordclouds_all_topics.png (overview grid)")
print("  - visualisations/topics/topic_overview.png (bar chart)")
print("  - visualisations/topics/topic_distribution.png (pie chart)")
print("  - visualisations/topics/keyword_heatmap.png (heatmap)")
print("  - visualisations/topics/wordclouds/topic_XX.png (individual)")
print("=" * 70)
