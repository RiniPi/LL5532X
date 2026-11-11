---
title: Home
nav_order: 1
description: "Can the outcome of a European Court of Human Rights Article 8 judgment be predicted from its text, and what does the model actually learn? An NLP study of 1,265 judgments (2010–2023)."
permalink: /
---

# Predicting Privacy
{: .fs-9 }

An NLP analysis of European Court of Human Rights judgments under **Article 8**, the right to respect for private and family life, home and correspondence.
{: .fs-6 .fw-300 }

[Read the findings](findings.md){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Open the full notebook](web/final/LL5532X_Pt_5_Final_Project_Notebook_Group_1_v4.md){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/RiniPi/LL5532X){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## The question

The Court decides thousands of cases a year for 46 member states. Article 8 is a **qualified** right: a state may interfere with it when that is "necessary in a democratic society". Outcomes therefore turn on proportionality and the margin of appreciation, which is exactly where people question whether the Court is consistent.

We asked whether a judgment's **text** predicts a finding of violation, and then what the models are really picking up when they succeed.

{: .key }
If outcomes are predictable from text, the Court's reasoning looks coherent. If they are not, that is itself a finding about how determinate Article 8 is. A failure to predict counts as a legally meaningful result.

## At a glance

| | |
|:--|:--|
| **Corpus** | 1,265 ECtHR judgments involving Article 8, 2010–2023, from [HUDOC](https://hudoc.echr.coe.int/) |
| **Outcomes** | 982 violation (78%) · 283 no violation (22%) |
| **Largest respondent** | Russia: 342 cases (27%), about 94% violation |
| **Metric** | Macro-F1 with 5-fold stratified cross-validation (a "always violation" dummy already scores ~78% accuracy) |
| **Models** | TF-IDF with Linear SVM, Logistic Regression and Naive Bayes · NMF topic models · Word2Vec embeddings |
| **Best score** | Macro-F1 **0.904** (Linear SVM on the Law section), partly inflated by remedy-language leakage |

## Four research questions

Each answer raised the next question.

| | Question | Method | Short answer |
|:--|:--|:--|:--|
| **[RQ1](findings.md#rq1-does-text-beat-metadata)** | Does judgment text beat structured metadata, and which section carries the signal? | TF-IDF, SVM / LR / NB, three-arm design | **Yes.** Text beats metadata, but the metadata model learns *country*, not law, and the Law section leaks the remedy |
| **[RQ2](findings.md#rq2-what-themes-drive-outcomes)** | What themes run through Article 8 case law, and do they predict outcomes? | NMF topic modelling (k = 40) | Some themes are **artefacts** (remedies, Russia), some are settled, and some are **genuinely contested** (close to 50/50) |
| **[RQ3](findings.md#rq3-do-patterns-transfer-across-regions)** | Does a model trained on one region generalise to the other? | Train East → test West and the reverse | **Asymmetrically.** West → East 0.85, East → West 0.77 |
| **[RQ4](findings.md#rq4-do-the-same-words-mean-the-same-thing)** | Do core doctrinal terms mean the same thing in both regions? | Word2Vec semantic drift, t-SNE, concordances | **Same words, different work.** *Home* and *correspondence* sit in different neighbourhoods East and West |

## Headline results

Macro-F1 under 5-fold stratified cross-validation, Linear SVM unless stated.

| Model | Macro-F1 | What it tells us |
|:--|--:|:--|
| Metadata only (country, year, importance) | 0.649 | A floor above chance, driven largely by respondent state |
| Text: Facts section (V1) | 0.694 | The most honest, leakage-free estimate |
| Text: Law section, full (V2) | **0.904** | Highest, but contains Article 41 "just satisfaction" language |
| Text: Law section, cleaned (V3) | 0.870 | Removing remedy words is not enough: leakage is structural |
| Metadata + Law Clean | 0.828 | Adding country *hurts*: no-violation recall falls from 0.74 to 0.62 |
| NMF topic weights (Logistic Regression) | 0.857 | Topics predict well, often for the wrong reasons |
| Cross-region: West → East / East → West | 0.855 / 0.774 | The Eastern model cannot recognise Western proportionality reasoning |

![Macro-F1 by text variant and classifier, with the metadata floor at 0.649](results/final/rq1/rq1_variant_comparison.png)

## Three takeaways

1. **The no-violation paradox.** Rights protection and judicial restraint use the same words. *Schalk and Kopf v. Austria* discusses "family life", "proportionality" and "discrimination" and still finds no violation. Bag-of-words models cannot see the difference, because the outcome depends on how the words are *arranged*, not on whether they appear.
2. **Geographic epistemology.** The model does not learn one universal Article 8. It learns the docket it was trained on: physical state coercion in the East, proportionality balancing in the West.
3. **"Neutral" legal AI is not on offer.** A perfectly balanced, clean, large legal corpus does not exist, because the docket itself is shaped by history and litigation patterns. Building legal AI means encoding a region's socio-political history.

## Explore the project

| Page | What you will find |
|:--|:--|
| [Findings](findings.md) | Each research question with its result, key figures and interpretation, plus implications and limitations |
| [Data & methods](methods.md) | Corpus, preprocessing, experimental design and evaluation |
| [Research journey](research-journey.md) | How the questions changed from proposal to final findings, and why each design decision was made |
| [Full analysis notebook](web/final/LL5532X_Pt_5_Final_Project_Notebook_Group_1_v4.md) | Every code output and figure from the final notebook |
| [PyCaret model comparison](web/pycaret_web_presentation_v1.md) | 14 classifiers benchmarked and grouped by interpretability |
| [Code & data](code-and-data.md) | Repository layout, notebooks, datasets, models and how to run them |
| [References](references.md) | Papers and ECtHR cases cited |

## Team

**LL5532X Group Project (Data Investigation), Group 1**

Amanda Lim Jin Mei · Nicha Nutchayangkul · Rini Pinasthika · Sophia Emily Lin

{: .note }
This site presents coursework for LL5532X. All rights are reserved by the authors unless stated otherwise; please contact the group before reusing any material. Judgment texts are public documents of the European Court of Human Rights, published on HUDOC.
