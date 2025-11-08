# Darija Word Embeddings: NLP for Low-Resource Arabic Dialects

A neural word embedding system for Moroccan Darija using Word2Vec Skip-gram with negative sampling, trained on 3 million sentences to advance natural language processing capabilities for under-resourced North African languages.

## 🎯 Research Motivation

Moroccan Darija, spoken by over 30 million people, remains critically under-resourced in NLP research despite being the primary language of Morocco. This project addresses the fundamental challenge of creating distributional semantic representations for dialectal Arabic, enabling computational linguistics research and practical applications for Moroccan digital infrastructure.

**Research Gap:** While Modern Standard Arabic has substantial NLP resources, Maghrebi dialects like Darija lack basic tools including word embeddings, trained language models, and annotated corpora.

**Contribution:** First large-scale Word2Vec embeddings for Darija trained on curated web corpus, establishing baseline semantic representations for downstream NLP tasks.

## 🛠️ Technical Architecture

### Neural Language Model

**Word2Vec Skip-gram with Negative Sampling:**
- **Architecture:** Predicts context words given target word
- **Optimization:** Negative sampling (k=10) for computational efficiency
- **Complexity Reduction:** O(V) → O(k log V) per training example
- **Training Objective:** Maximize log probability of context words

**Mathematical Foundation:**
```
Objective: maximize Σ log σ(v'_w · v_w) + Σ log σ(-v'_neg · v_w)
where σ is sigmoid, v_w is word vector, v'_w is context vector
```

**Why Skip-gram:**
- Superior performance on rare words vs CBOW
- Better semantic relationship capture
- Handles morphologically rich languages effectively
- Proven effectiveness for low-resource languages

### Computational Infrastructure

**Software Stack:**
- **Gensim 4.x:** Optimized C implementations for training
- **Python 3.10:** Development environment
- **NumPy/SciPy:** Vector operations and similarity computations
- **Regex:** Arabic text tokenization and preprocessing
- **Jupyter:** Interactive experimentation and analysis

**Custom Preprocessing Pipeline:**
```python
class DarijaTokenizer:
    - Unicode normalization for Arabic script
    - Punctuation removal preserving semantics
    - Case normalization (critical for mixed scripts)
    - Whitespace tokenization with Arabic handling
    - Stop word preservation (context-critical)
```

### Dataset & Corpus Engineering

**Corpus Specifications:**
- **Source:** Multi-domain Darija web scraping
- **Scale:** 8.7M sentences total (3M used for training)
- **Languages:** Romanized Darija, Arabic script Darija
- **Domains:** Social media, news, forums, literature
- **Quality:** Curated and filtered for linguistic validity

**Data Sampling Strategy:**
- Random sampling for domain diversity
- Length filtering (5-100 tokens per sentence)
- Language identification to ensure Darija (not MSA)
- Deduplication using MinHash LSH

## 💡 Experimental Design & Results

### Hardware Performance Study

**Comparative Analysis - Understanding Computational Bottlenecks:**

| Configuration | CPU | RAM | Storage | 1M Sentences | 3M Sentences | Status |
|--------------|-----|-----|---------|--------------|--------------|---------|
| System A | 4-core | 8GB | HDD | Failed (OOM) | - |   Bottleneck |
| System B | 8-core | 32GB | SSD RAID | 25 min | 90 min |   Optimal |

**Key Findings:**
1. **Memory Hierarchy Critical:** SSD vs HDD resulted in 10x performance difference
2. **RAM Requirements:** Minimum 4x corpus size needed for stable training
3. **CPU Utilization:** Multi-threading essential; single-thread training impractical
4. **I/O Bottleneck:** Disk speed directly impacts training time for large corpora

**System B Performance Metrics:**
- Peak memory: 30GB (~94% utilization)
- CPU usage: 100% sustained (parallel workers)
- Training throughput: ~33K sentences/minute
- Convergence: Stable loss decrease over 10 epochs

### Model Hyperparameter Exploration

**Optimized Configuration:**
```python
Word2Vec Parameters:
- vector_size: 200        # Dimensionality of embeddings
- window: 5               # Context window (±5 words)
- min_count: 5            # Vocabulary frequency threshold
- workers: 8              # Parallel training threads
- sg: 1                   # Skip-gram (vs CBOW)
- negative: 10            # Negative samples per positive
- epochs: 10              # Training iterations
- alpha: 0.025            # Initial learning rate
- sample: 1e-3            # Downsampling frequent words
```

**Parameter Rationale:**
- **Vector size (200):** Balance between expressiveness and generalization
- **Window (5):** Captures phrasal semantics without noise
- **Negative samples (10):** Optimal for vocabulary size ~500K
- **Min count (5):** Filters noise while preserving rare morphological variants

### Embedding Quality Evaluation

**Intrinsic Evaluation Metrics:**
1. **Semantic Similarity:** Cosine similarity for related words
2. **Analogy Tasks:** Darija-specific king-queen:man-woman patterns
3. **Nearest Neighbors:** Contextually appropriate word clusters
4. **Vocabulary Coverage:** 500K+ unique tokens learned

**Qualitative Analysis:**
- Embeddings capture dialectal variations
- Morphological relationships preserved (Arabic root patterns)
- Code-
