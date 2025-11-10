# KIDLINKTopics

Multilingual topic modeling analysis of archived webpages from KIDLINK extracted from the Internet Archive.
The topic model is made using BERTopic and transformer-based embeddings.

## Overview

This project performs topic modeling on historical KIDLINK content archived by the Internet Archive and loaded into a local instance of Solrwayback. KIDLINK was a global educational network that connected youth (up to age 15) from around the world for collaborative learning and cultural exchange. The archive contains multilingual content in all participating languagues including Danish, Norwegian, English, Spanish, German, Italian, Portuguese, and other languages.

Using state-of-the-art NLP techniques (BERTopic with multilingual sentence transformers), this project identifies and visualizes the main themes and topics discussed across the KIDLINK network during its active years. The topics are used as a keyword discovery tool, so that manual queries in SolrWayback can be created from a better starting point.

## Features

- **Multilingual Topic Modeling**: Handles content in multiple languages using `paraphrase-multilingual-MiniLM-L12-v2` embeddings
- **GPU Acceleration**: Utilizes Apple Silicon GPU (MPS) when available for faster processing
- **Comprehensive Visualizations**: Generates multiple visualization types to explore topic distributions
- **Flexible Configuration**: Adjustable parameters for topic granularity and keyword extraction
- **Custom Stopword Management**: Supports multilingual and domain-specific stopword filtering

## Project Structure

```
KIDLINKTopics/
├── data/
│   ├── solrwayback_kidlink-org-dk.csv           # Source data
│   ├── custom_stopwords.txt                     # Domain-specific stopwords
├── models/
│   └── topic_model_bertopic                     # Saved BERTopic model
├── src/
│   └── kidlink_topics/
│       ├── topic_model.py                       # Main topic modeling script - creates the model
│       ├── utils.py                             # Utility functions
│       ├── visualise_topics.py                  # Generate visualizations for the topic model
│       ├── visualise_keywords.py                # Keyword visualizations for the clustering analysis
│       ├── get_keywords_from_topic.py           # Extract keywords from specific topic
│       ├── search_topics.py                     # Search topics by keywords
│       └── simple_cluster.py                    # Simple clustering analysis
├── visualisations/
│   ├── topics/                                  # Topic visualizations are saved here. E.g the results from visualise_topics.py
│   │   └── wordclouds/                          # Individual topic wordclouds
│   └── wordclouds/                              # General wordclouds from the simple_cluster.py
├── MODEL_CHANGES.md                             # Model iteration history
└── README.md                                    
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd KIDLINKTopics
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install pandas numpy torch bertopic sentence-transformers scikit-learn nltk matplotlib wordcloud
```

4. Download NLTK stopwords (first run only):
```python
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

### 1. Generate Topic Model

Run the main topic modeling script to analyze the KIDLINK extraction from SolrWayback:

```bash
python src/kidlink_topics/topic_model.py
```

**Configuration options** (edit in `topic_model.py`):
- `MIN_DOCUMENTS_PR_TOPIC`: Minimum documents required to form a topic (default: 80)
- `AMOUNT_OF_KEYWORDS_PR_TOPIC`: Number of keywords to extract per topic (default: 50)

**Output:**
- `data/topic_model_results.json`: Complete topic analysis with keywords and statistics
- `models/topic_model_bertopic/`: Saved model for later use

### 2. Visualize Topics

Generate comprehensive visualizations of the topic model results:

```bash
python src/kidlink_topics/visualise_topics.py
```

**Generates:**
- `wordclouds_all_topics.png`: Grid overview of all topics
- `topic_overview.png`: Bar chart with document counts and top keywords
- `topic_distribution.png`: Pie chart showing topic proportions
- `keyword_heatmap.png`: Heatmap of keyword importance across topics
- `wordclouds/topic_XX.png`: Individual high-resolution wordclouds

### 3. Explore Specific Topics

View all keywords for a specific topic:

```bash
python src/kidlink_topics/get_keywords_from_topic.py <topic_number>
```

Example:
```bash
python src/kidlink_topics/get_keywords_from_topic.py 0
python src/kidlink_topics/get_keywords_from_topic.py 5 data/topic_model_results.json
```

### 4. Search Topics

Search for topics containing specific keywords:

```bash
python src/kidlink_topics/search_topics.py
```

## Model Configuration

### Current Configuration

- **Minimum documents per topic**: 80
- **Keywords per topic**: 50
- **Embedding model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Vectorizer**: CountVectorizer with multilingual stopwords
- **N-gram range**: (1, 2) - includes unigrams and bigrams
- **Topics generated**: ~525 topics with ~146K outliers

### Stopword Management

Custom stopwords are defined in `data/custom_stopwords.txt` and include:

- Web-related terms (html, http, www)
- KIDLINK founder name (Odd Presno)
- Years and dates
- Search-related terms (advanced, search)
- Language-specific common words

The system combines these with NLTK stopwords for:
- Danish, Norwegian, English, Spanish, German, Italian, Portuguese

### Iteration History

See `MODEL_CHANGES.md` for detailed history of model refinements and parameter adjustments.

## Data Format

### Input CSV

Expected format for `data/solrwayback_kidlink-org-dk.csv`. The file should be created from a SolrWayback export with the fields CSV and content selected:
```csv
content
"Document text content here..."
"Another document..."
```

### Output JSON

Topic model results are saved in JSON format:

```json
[
    {
        "topic_id": 0,
        "num_docs": 11377,
        "keywords": ["keyword1", "keyword2", ...],
        "keyword_scores": {"keyword1": 0.85, "keyword2": 0.72, ...},
        "name": "Topic 0",
        "sample": "Sample document preview..."
    }
]
```

## Hardware Acceleration

The project automatically detects and uses Apple Silicon GPU (MPS) when available:

```
✓ Using Apple Silicon GPU (MPS) for acceleration
```

On non-Apple Silicon systems, it falls back to CPU processing.

## Contributing

When modifying the model configuration:

1. Document changes in `MODEL_CHANGES.md`
2. Note parameter values and resulting topic counts

