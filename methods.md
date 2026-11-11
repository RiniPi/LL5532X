---
title: Data & methods
nav_order: 3
description: "Corpus, preprocessing, experimental design and evaluation for the ECtHR Article 8 NLP project."
---

# Data & methods
{: .no_toc }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Corpus

| | |
|:--|:--|
| **Source** | [HUDOC](https://hudoc.echr.coe.int/), the European Court of Human Rights case-law database |
| **Documents** | English-language ECtHR judgments involving Article 8 |
| **Period** | 2010–2023, the period in which Article 8 was increasingly applied to digital spaces |
| **Unit of analysis** | One judgment |
| **Target** | Binary: violation vs no violation of Article 8 |
| **Size** | 1,265 usable judgments: 982 violation, 283 no violation |
| **Fields** | Full text split into Facts and Law sections, respondent state, year, importance level, structured conclusion |
| **File** | [`data/echr_art8_filtered.parquet`](https://github.com/RiniPi/LL5532X/blob/main/data/echr_art8_filtered.parquet) |

## What the exploratory analysis found

The EDA changed the project more than any model did. Each finding became a design rule.

| Finding | Evidence | Design rule |
|:--|:--|:--|
| **Severe class imbalance** | 78% violation. Admissibility filtering means weak cases never reach a judgment | Use **macro-F1**, not accuracy, and balanced class weights throughout |
| **Geographic concentration** | Russia alone is about 27% of the corpus, with about 94% violation | Treat respondent state as a **bias risk**, not just a feature |
| **Structural leakage** | Article 41 (just satisfaction) appears only when a violation is found; 8 + 41 is the most common article pairing | Separate remedy language from reasoning, and test Facts and Law separately |
| **Genuine narrative signal** | Violation facts use coercive state vocabulary (*detention, surveillance, seizure, expulsion*); no-violation facts use procedural, domestic-remedy language | Test the Facts section alone; build a custom stop-word list for universal doctrinal terms |

![Shifterator word shift between violation and no-violation Facts sections](results/final/eda/eda_shifterator_facts.png)

## Pipeline

1. **Section extraction.** Split each judgment into Facts and Law.
2. **Preprocessing.** Lower-casing, lemmatisation, a custom legal stop-word list, and bigram detection (e.g. `margin_appreciation`, `private_life`).
3. **Representation.** TF-IDF n-grams (RQ1, RQ3), NMF topic weights (RQ2, RQ3) and region-specific Word2Vec embeddings (RQ4).
4. **Classification.** Linear SVM, Logistic Regression and Multinomial Naive Bayes.
5. **Evaluation.** 5-fold stratified cross-validation and macro-F1, with confusion matrices and per-class recall.
6. **Interpretation.** Linear feature weights, topic descriptors, nearest-neighbour tables, t-SNE and concordances.

## Key design choices

| Choice | Why |
|:--|:--|
| **Macro-F1** | Averages per-class F1, so the model is held equally accountable for predicting state *wins* |
| **Linear, interpretable models** | The research question is about *reasons*. Feature weights are how the leakage, the Russia effect and the regional divergence were found |
| **Three text variants (V1 / V2 / V3)** | Turns the leakage question into a controlled experiment |
| **SMOTE only on low-dimensional features** | Applied to metadata and topic features, not raw text, to avoid overfitting to synthetic points |
| **NMF k = 40 (RQ2), k = 20 (RQ3)** | Coherence peaked at k = 7, but that merged distinct legal areas; k = 20 keeps regional topic models stable |
| **East / West split with zero country overlap** | Moves the question from "who is being sued" towards "what the legal reasoning is"; Turkey held out as a boundary state |
| **Separate Word2Vec models per region** | Any difference in a word's position reflects regional usage, not text volume |
| **CPU-feasible methods only** | Contextual models (DistilBERT, Legal-BERT) moved to future work |

## Tools

Python on Google Colab: pandas, scikit-learn, imbalanced-learn (SMOTE), PyCaret, NLTK, spaCy, gensim (Word2Vec), ScatterText, Shifterator, matplotlib, seaborn and Plotly. The full list is in [`requirements.txt`](https://github.com/RiniPi/LL5532X/blob/main/requirements.txt).

The team also used Gemini and Claude models in Colab to help generate and debug code, analyse data and review drafts, and Gemini 2.5 Flash Lite for the RQ3 misclassification analysis. All outputs were verified by the group, who remain responsible for the work.
