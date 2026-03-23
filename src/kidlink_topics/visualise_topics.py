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
    Expects JSON file at: data/topic_model_results.json (or data/{PREFIX}_topic_model_results.json)
    Format: Array of topic objects with fields:
        - topic_id: Integer topic identifier
        - num_docs: Number of documents in topic
        - keywords: List of keyword strings
        - keyword_scores: Optional dict of {keyword: score}
        - name: Optional topic name/label

Output:
    All visualizations saved to visualisations/topics/ directory:
    - {PREFIX_}wordclouds_all_topics.png
    - {PREFIX_}topic_overview.png
    - {PREFIX_}topic_distribution.png
    - {PREFIX_}keyword_heatmap.png
    - wordclouds/{PREFIX_}topic_XX.png (one per topic)

Usage:
    python visualise_topics.py
    python visualise_topics.py --domain-prefix kidlink_org_dk
"""
import json
import argparse
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import os
import sys
from pathlib import Path
from utils import save_figure, get_topic_results_path


def find_cjk_font():
    """Find a font that supports CJK characters (Chinese, Japanese, Korean)."""
    # Font candidates for different platforms
    font_candidates = []
    
    if sys.platform == "darwin":  # macOS
        font_candidates = [
            "/Library/Fonts/AppleGothic.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Hiragino Sans W3.otf",
        ]
    elif sys.platform == "linux":
        font_candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf",
            "/usr/share/fonts/opentype/dejavu/DejaVuSans.ttf",
        ]
    elif sys.platform == "win32":  # Windows
        font_candidates = [
            "C:\\Windows\\Fonts\\msyh.ttc",  # Microsoft YaHei
            "C:\\Windows\\Fonts\\Arial.ttf",
        ]
    
    # Find first available font
    for font_path in font_candidates:
        if Path(font_path).exists():
            return font_path
    
    # Fallback: no font specified (system default)
    return None

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate topic model visualizations")
parser.add_argument("--domain-prefix", type=str, default=None, help="Optional domain prefix for input/output filenames (e.g., 'kidlink_org_dk')")
args = parser.parse_args()

domain_prefix = args.domain_prefix

# Derive a displayable domain name from the domain prefix.
# Keep `output_prefix` (used for filenames) unchanged; only alter the
# human-readable `display_domain` that appears inside the visuals.
import re

display_domain = None
if domain_prefix:
    # If the prefix starts with 'att' or 'p' + digits followed by an underscore,
    # strip that leading portion for display (e.g. 'att16_refugees-dk' -> 'refugees-dk').
    m = re.match(r'^(?:att|p)\d+_(.+)$', domain_prefix)
    if m:
        domain_only = m.group(1)
    else:
        # Fallback: if there is an underscore, take the part after it; otherwise use full prefix
        if "_" in domain_prefix:
            domain_only = domain_prefix.split("_", 1)[1]
        else:
            domain_only = domain_prefix

    # Replace dashes with dots for nicer display
    display_domain = domain_only.replace("-", ".")
else:
    display_domain = None

# Determine input and output paths based on domain prefix
input_json = get_topic_results_path(domain_prefix)
output_prefix = f"{domain_prefix}_" if domain_prefix else ""

print("Loading topic model results...")
with open(input_json, 'r', encoding='utf-8') as f:
    topic_data = json.load(f)
print(f"✓ Loaded {len(topic_data)} topics from {input_json}\n")

# Find CJK font for proper character rendering
cjk_font = find_cjk_font()
if cjk_font:
    print(f"✓ Using CJK-compatible font: {cjk_font}\n")
    # Configure matplotlib to use the CJK font for all text rendering
    plt.rcParams['font.sans-serif'] = [cjk_font]
    plt.rcParams['axes.unicode_minus'] = False
else:
    print("⚠ No CJK font found - wordclouds may not display non-Latin characters properly\n")

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
                   colormap='plasma', relative_scaling=0.5,
                   font_path=cjk_font).generate_from_frequencies(word_freq)
    
    axes[i].imshow(wc, interpolation='bilinear')
    topic_name = topic.get('name', f"Topic {topic['topic_id']}")
    title_lines = [f"{topic_name}", f"({topic['num_docs']} docs)"]
    if display_domain:
        title_lines.append(display_domain)
    axes[i].set_title("\n".join(title_lines), fontsize=9, fontweight='bold')
    axes[i].axis('off')

# Hide unused subplots
for i in range(num_topics, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
save_figure(f'visualisations/topics/{output_prefix}wordclouds_all_topics.png', fig, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualisations/topics/{output_prefix}wordclouds_all_topics.png\n")
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
title = 'Topic Sizes and Top 5 Keywords'
if display_domain:
    title = f"{title} — {display_domain}"
ax.set_title(title, fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
save_figure(f'visualisations/topics/{output_prefix}topic_overview.png', fig, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualisations/topics/{output_prefix}topic_overview.png\n")
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

title = 'Topic Distribution (Top 10 Topics)'
if display_domain:
    title = f"{title} — {display_domain}"
ax.set_title(title, fontsize=14, fontweight='bold')
plt.tight_layout()
save_figure(f'visualisations/topics/{output_prefix}topic_distribution.png', fig, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualisations/topics/{output_prefix}topic_distribution.png\n")
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
                   colormap='plasma', relative_scaling=0.5,
                   font_path=cjk_font).generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(10, 7))
    plt.imshow(wc, interpolation='bilinear')
    topic_name = topic.get('name', f"Topic {topic['topic_id']}")
    title = f"{topic_name} - {topic['num_docs']} documents"
    if display_domain:
        title = f"{title} — {display_domain}"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    fig = plt.gcf()
    save_figure(f"visualisations/topics/wordclouds/{output_prefix}topic_{topic['topic_id']:02d}.png", 
                fig, dpi=150, bbox_inches='tight')
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

title = "Top 10 Topics - Keyword Importance Heatmap"
if display_domain:
    title = f"{title} — {display_domain}"
ax.set_title(title, fontsize=12, fontweight='bold')
fig.tight_layout()
save_figure(f'visualisations/topics/{output_prefix}keyword_heatmap.png', fig, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualisations/topics/{output_prefix}keyword_heatmap.png\n")
plt.close()

# Summary
print("=" * 70)
print("Done! Generated:")
print(f"  - visualisations/topics/{output_prefix}wordclouds_all_topics.png (overview grid)")
print(f"  - visualisations/topics/{output_prefix}topic_overview.png (bar chart)")
print(f"  - visualisations/topics/{output_prefix}topic_distribution.png (pie chart)")
print(f"  - visualisations/topics/{output_prefix}keyword_heatmap.png (heatmap)")
print(f"  - visualisations/topics/wordclouds/{output_prefix}topic_XX.png (individual)")
print("=" * 70)
