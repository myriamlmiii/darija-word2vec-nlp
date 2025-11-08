# Darija Word Embeddings: Neural NLP for Low-Resource Arabic Dialects

A neural word embedding system for Moroccan Darija using Word2Vec Skip-gram with negative sampling, trained on 3 million sentences from a curated 8.7M sentence corpus to advance natural language processing capabilities for under-resourced North African languages.

## 🎯 Research Motivation

Moroccan Darija, spoken by over 30 million people, remains critically under-resourced in NLP research despite being the primary language of Morocco. This project addresses the fundamental challenge of creating distributional semantic representations for dialectal Arabic, enabling computational linguistics research and practical applications for Moroccan digital infrastructure.

**Research Gap:** While Modern Standard Arabic has substantial NLP resources, Maghrebi dialects like Darija lack basic tools including word embeddings, trained language models, and annotated corpora.

**Contribution:** Large-scale Word2Vec embeddings for Darija trained on web-curated corpus, establishing baseline semantic representations for downstream NLP tasks in Moroccan Arabic dialect processing.

## 🛠️ Technical Architecture

### Neural Language Model

**Word2Vec Skip-gram with Negative Sampling:**
- **Architecture:** Predicts context words given target word
- **Optimization:** Negative sampling (k=10) for computational efficiency  
- **Complexity Reduction:** O(V) → O(k log V) per training example
- **Training Objective:** Maximize log probability of context words

**Mathematical Foundation:**
```
Objective: maximize Σ log σ(v'_c · v_w) + Σ_i log σ(-v'_ni · v_w)
where σ = sigmoid, v_w = target word vector, v'_c = context vector
```

**Why Skip-gram Over CBOW:**
- Superior performance on rare words and morphological variants
- Better semantic relationship capture in low-resource settings
- Handles morphologically rich languages more effectively
- Proven effectiveness for dialectal and under-resourced languages

### Computational Infrastructure

**Software Stack:**
- **Gensim 4.x** - Optimized C implementations for efficient training
- **Python 3.10** - Development and experimentation environment
- **NumPy/SciPy** - Vector operations and similarity computations
- **Regex** - Arabic script tokenization and text preprocessing
- **Jupyter Notebook** - Interactive development and analysis
- **Virtual Environment** - Isolated dependency management (`darija_env`)

**Custom Preprocessing Pipeline:**
```python
Darija Text Preprocessing:
├── Unicode normalization for Arabic script
├── Punctuation removal preserving semantic markers
├── Case normalization (critical for mixed-script text)
├── Whitespace tokenization with Arabic handling
├── Special character filtering
└── Sentence boundary detection
```

### Dataset & Corpus Engineering

**Corpus Specifications:**
- **Total Scale:** 8.7 million sentences
- **Training Subset:** 3 million sentences (optimal configuration)
- **Source:** Multi-domain Darija web scraping and curation
- **Script Types:** Romanized Darija, Arabic script Darija, mixed
- **Domains:** Social media, news, forums, blogs, literature
- **Quality Control:** Linguistic validation and filtering

**Corpus Sampling Strategy:**
- Stratified random sampling for domain diversity
- Length filtering (5-100 tokens per sentence)
- Language identification ensuring Darija (not MSA contamination)
- Deduplication and quality validation
- Vocabulary size: 500K+ unique tokens

## 💡 Experimental Design & Performance Analysis

### Computational Performance Study

**Hardware Configuration Comparison:**

| System | CPU Cores | RAM | Storage | 1M Sentences | 3M Sentences | Result |
|--------|-----------|-----|---------|--------------|--------------|---------|
| Config A | 4 cores | 8GB | HDD | System freeze | - | ❌ Failed |
| Config B | 8 cores | 32GB | SSD RAID | ~25 minutes | ~90 minutes | ✅ Success |

**Performance Bottleneck Analysis:**

**Configuration A Failure Symptoms:**
- CPU utilization dropped to 0% during training (process stall)
- Memory peaked at 94-97% causing system freeze
- HDD I/O bottleneck with slow disk read speeds
- Python process unresponsive during vocabulary building
- Unable to complete even 1M sentence training

**Configuration B Success Factors:**
- SSD provided 10x faster I/O for corpus loading
- 32GB RAM supported in-memory operations
- 8-core CPU enabled Gensim's multi-threaded training
- RAID configuration accelerated data access
- Stable training at 100% CPU, 94% memory utilization

**Key Technical Insights:**
1. **Memory Hierarchy Critical:** SSD vs HDD resulted in order-of-magnitude performance difference
2. **RAM Requirements:** Minimum 3-4x corpus size needed for stable training (3M sentences ≈ 30GB)
3. **Parallel Processing:** Multi-threading essential; Gensim scales near-linearly with cores
4. **I/O Dominates:** Disk speed directly impacts training time for large text corpora

### Model Hyperparameter Configuration

**Optimized Training Parameters:**
```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=darija_corpus,
    vector_size=200,      # Embedding dimensionality
    window=5,             # Context window (±5 words)
    min_count=5,          # Vocabulary frequency threshold
    workers=8,            # Parallel training threads
    sg=1,                 # Skip-gram architecture
    negative=10,          # Negative samples per positive
    epochs=10,            # Training iterations
    alpha=0.025,          # Initial learning rate
    sample=1e-3,          # Subsampling threshold
    hs=0                  # Negative sampling (not hierarchical softmax)
)
```

**Parameter Selection Rationale:**
- **vector_size=200:** Balances expressiveness with generalization for 500K vocabulary
- **window=5:** Captures phrasal semantics without incorporating noise
- **negative=10:** Optimal for vocabulary size based on Mikolov et al. recommendations
- **min_count=5:** Filters noise while preserving rare morphological variants
- **workers=8:** Maximizes multi-core utilization on available hardware

### Training Performance Metrics

**System B Training Results:**
- **Throughput:** ~33,000 sentences/minute
- **Peak Memory:** 30GB (~94% of available RAM)
- **CPU Utilization:** 100% sustained across all cores
- **Training Time (3M):** 90 minutes (~5.4K sentences/second)
- **Convergence:** Stable loss decrease across 10 epochs
- **Final Vocabulary:** 500K+ unique Darija tokens

## 📊 Embedding Quality & Applications

### Vector Space Properties

**Learned Representations:**
- 200-dimensional dense vectors for each vocabulary word
- Cosine similarity captures semantic relationships
- Vector arithmetic reveals linguistic patterns
- Context-sensitive meaning preservation

**Qualitative Observations:**
- Dialectal variations correctly clustered
- Morphological relationships preserved (Arabic root patterns)
- Code-switching patterns between Darija-French captured
- Semantic fields (family, food, emotions) form coherent clusters

### Potential Applications

**Downstream NLP Tasks:**
1. **Sentiment Analysis** - Moroccan social media opinion mining
2. **Machine Translation** - Darija ↔ French/English/MSA translation
3. **Text Classification** - Topic modeling and document categorization
4. **Named Entity Recognition** - Person/location/organization extraction
5. **Information Retrieval** - Darija document search and ranking
6. **Conversational AI** - Chatbots and virtual assistants for Morocco
7. **Semantic Similarity** - Document clustering and recommendation

**Research Applications:**
- Computational dialectology and linguistic analysis
- Cross-dialectal Arabic NLP transfer learning
- Low-resource language processing methodology
- Sociolinguistic pattern discovery in Maghrebi dialects

## 🚀 Technical Contributions & Skills

**Natural Language Processing:**
- Neural embedding architectures (Word2Vec, Skip-gram)
- Large-scale language model training and optimization
- Text preprocessing for morphologically rich languages
- Evaluation methodologies for word embeddings

**Machine Learning Engineering:**
- Hyperparameter tuning and model selection
- Computational resource optimization
- Scalability analysis and performance benchmarking
- Memory management for large-scale ML

**Low-Resource NLP:**
- Working with under-resourced and dialectal languages
- Corpus curation and quality control
- Transfer learning from high-resource to low-resource languages
- Evaluation without standard benchmarks

**Systems & Performance:**
- Hardware-software performance trade-off analysis
- Distributed computing and parallel processing
- I/O optimization for large text corpora
- Resource constraint problem-solving

## 🌍 Impact & Future Directions

### Addressing Language Technology Gaps

**Moroccan Darija Challenges:**
- Under-resourced compared to Modern Standard Arabic
- Mixed orthography (Arabic script + Romanization)
- Significant dialectal variation across regions
- Limited annotated datasets and NLP tools
- Code-switching with French and MSA

**This Work Contributes:**
- First publicly available Darija word embeddings at scale
- Baseline for future Moroccan NLP research
- Methodology for other Maghrebi dialects (Algerian, Tunisian)
- Foundation for Darija language technology development

### Future Research Directions

**Model Enhancements:**
- Subword embeddings (FastText) for morphological robustness
- Contextualized embeddings (BERT-style transformers for Darija)
- Multi-lingual models incorporating MSA, French, Berber
- Domain-specific fine-tuning (social media, news, formal)

**Scaling Strategies:**
- Cloud computing (AWS/GCP) for full 8.7M sentence training
- GPU acceleration with custom CUDA implementations
- Distributed training across compute clusters
- Incremental learning for continuous corpus updates

**Evaluation & Validation:**
- Creation of Darija analogy datasets
- Human evaluation of semantic similarity
- Extrinsic evaluation on downstream tasks
- Cross-lingual evaluation with MSA embeddings

## 📄 Technical Documentation

For complete implementation details, training logs, hyperparameter experiments, performance analysis, and hardware configuration specifications, see the [full project documentation](darija-word-embeddings.pdf).

## 🎓 Project Context

Developed to advance computational linguistics for Moroccan Darija and contribute to low-resource NLP research. This work demonstrates practical machine learning engineering on resource-constrained problems while addressing real-world language technology needs for North African digital infrastructure.

The project showcases end-to-end ML pipeline development: from data collection and preprocessing, through model architecture selection and training, to performance optimization and evaluation - all while navigating the unique challenges of dialectal Arabic and computational resource limitations.

## 🔗 Connect

**Meriem Lmoubariki**
- GitHub: [@myriamlmiii](https://github.com/myriamlmiii)

---

*Advancing NLP for low-resource languages through neural embeddings and computational linguistics research.*
