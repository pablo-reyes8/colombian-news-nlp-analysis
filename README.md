# 📰 Colombian News NLP Analysis

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![NLP](https://img.shields.io/badge/NLP-Spanish%20News-orange)]()
![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/colombian-news-nlp-analysis)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/colombian-news-nlp-analysis)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/colombian-news-nlp-analysis)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/colombian-news-nlp-analysis)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/colombian-news-nlp-analysis?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/colombian-news-nlp-analysis?style=social)

A Natural Language Processing (NLP) project focused on **scraping, cleaning, and analyzing Colombian news** from multiple sources (_El Espectador, Semana, El Colombiano_) to uncover patterns in **topics, sentiment, and named entities**.  
The goal is to transform a large and fragmented corpus of news into a **structured map of trends, emotions, and key actors** in the Colombian media landscape.

## 📌 Project Overview

Every day, hundreds of news articles are published in Colombia across multiple outlets, making it difficult to obtain a clear and unbiased overview of the national agenda.  
This project centralizes that information by:

- **Scraping** thousands of news articles.
- **Cleaning & preprocessing** the text corpus.
- Applying **NLP techniques** for:
  - Thematic classification (10 main categories).
  - Sentiment analysis (BETO, 5+1 classes).
  - Named Entity Recognition (NER).

---

## 🚀 Features

- **News Scraping** → Automated collection of news articles from different sources.  
- **Preprocessing Pipeline** → Tokenization, lemmatization, stopword removal, handling of accents and casing.  
- **Text Representations**:
  - Bag of Words (BoW)
  - TF-IDF
  - Sentence-BERT embeddings  
- **Supervised Classification**:
  - MLP (Multi-Layer Perceptron)
  - Linear SVM
  - Random Forest  
- **Sentiment Analysis**:
  - BETO pre-trained model (`ignacio-ave/beto-sentiment-analysis-spanish`) with HuggingFace
  - Sliding window strategy for long news (>512 tokens)
- **NER Exploration**:
  - Entity extraction (PER, ORG, LOC, MISC)
  - Interactive filtering by category & time period  

---

## 📊 Key Findings

- **Topic Distribution**: News are heavily biased towards politics and society (~40%), while economy and regions are underrepresented.  
- **Model Performance**:
  - **TF-IDF + SVM** achieved the best accuracy (**76%**).
  - BoW and BERT embeddings performed lower (61–72%).  
- **Sentiment**: The corpus shows a **predominantly negative tone**, especially towards political figures (e.g., Gustavo Petro, Miguel Uribe Turbay).  
- **NER**: Key entities include politicians, institutions (Fiscalía, Centro Democrático), and major cities (Bogotá, Medellín, Washington).

---

## 📂 Repository Structure


```plaintext
colombian-news-nlp-analysis/
├── data/
│ ├── data_full/ # Raw & preprocessed corpus
│ ├── scrapping_elcolombiano/ # Scrapers + data (El Colombiano)
│ ├── scrapping_espectador/ # Scrapers + data (El Espectador)
│ └── scrapping_semana/ # Scrapers + data (Semana)
│
├── src/
│ ├── preprocessing/ # Cleaning pipeline (main_preprocessing.py)
│ ├── embeddings/ # BoW, TF-IDF, BERT + viz
│ ├── modeling/ # MLP, SVM, RF, sentiment, NER
│ └── corpus_analysis/ # EDA + visualization helpers
│
├── notebooks/ # Exploratory/analysis notebooks
├── figures/ # Generated plots
├── requirements.txt
├── pyproject.toml / poetry.lock
├── .gitattributes
└── README.md
```



---
## 📈 Example Visualizations

- **Sentiment evolution over time** → smoothed with moving averages.  
- **Word clouds, bigrams & trigrams** → to uncover frequent narratives.  
- **Confusion matrices** → model performance by class.  
- **Violin plots** → distribution of sentiment confidence scores.  
- **Interactive NER explorer** → entity frequency by category and time.  

---

## 🔮 Future Work

- Fine-tuning transformer-based models (BETO, mBERT, RoBERTa).  
- Explore topic modeling (LDA, BERTopic) for unsupervised theme discovery.  
- Expand corpus with social media (Twitter, Reddit, forums).  
- Apply **active learning** for iterative and more robust labeling.  
- Integrate visualization dashboards (Streamlit or Plotly Dash).  

---

## 👤 Author

This project was developed as part of the **Machine Learning Diploma at Universidad Nacional de Colombia (UNAL)**.  

🔗 [GitHub](https://github.com/pablo-reyes8)
🔗 [Linkedin](https://www.linkedin.com/in/pablo-alejandro-reyes-granados/)

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

