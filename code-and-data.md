---
title: Code & data
nav_order: 8
description: "Repository layout, notebooks, datasets, trained models and how to run the analysis."
---

# Code & data
{: .no_toc }

Everything behind the analysis lives in the [GitHub repository](https://github.com/RiniPi/LL5532X). Large files (data, models, notebooks, slides) are linked to GitHub rather than served by this site.
{: .fs-5 .fw-300 }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## Notebooks

The notebooks are numbered by project stage and were written for Google Colab.

| Notebook | Stage | Contents |
|:--|:--|:--|
| [`01_eda_v3.ipynb`](https://github.com/RiniPi/LL5532X/blob/main/notebooks/01_eda_v3.ipynb) | Part 2 | Exploratory data analysis |
| [`03_modeling_prelim_v5.ipynb`](https://github.com/RiniPi/LL5532X/blob/main/notebooks/03_modeling_prelim_v5.ipynb) | Part 3 | Modelling and preliminary results |
| [`04_presentation_visuals.ipynb`](https://github.com/RiniPi/LL5532X/blob/main/notebooks/04_presentation_visuals.ipynb) | Part 4 | Charts for the in-class presentation |
| [`05_final_project_v4.ipynb`](https://github.com/RiniPi/LL5532X/blob/main/notebooks/05_final_project_v4.ipynb) | Part 5 | **Final project notebook** ([read it on this site](web/final/LL5532X_Pt_5_Final_Project_Notebook_Group_1_v4.md)) |
| [`05_final_project_pycaret_v1.ipynb`](https://github.com/RiniPi/LL5532X/blob/main/notebooks/05_final_project_pycaret_v1.ipynb) | Part 5 | PyCaret model comparison ([read it on this site](web/pycaret_web_presentation_v1.md)) |

## Data

| File | Description |
|:--|:--|
| [`data/echr_art8_filtered.parquet`](https://github.com/RiniPi/LL5532X/blob/main/data/echr_art8_filtered.parquet) | Article 8 judgments, 2010–2023: full text, structured conclusion, respondent state, year, importance level, court, separate opinions |
| [`data/east_sents_phrased.pkl`](https://github.com/RiniPi/LL5532X/blob/main/data/east_sents_phrased.pkl) | Tokenised, phrase-joined sentences from Eastern Law sections (RQ4) |
| [`data/west_sents_phrased.pkl`](https://github.com/RiniPi/LL5532X/blob/main/data/west_sents_phrased.pkl) | The same for Western Law sections (RQ4) |

Judgments are public documents of the European Court of Human Rights, published on [HUDOC](https://hudoc.echr.coe.int/).

{: .caution }
`.pkl` files can run code when loaded. Only open them from a source you trust, such as a fresh clone of this repository.

## Models

| Folder | Contents |
|:--|:--|
| [`models/final/`](https://github.com/RiniPi/LL5532X/tree/main/models/final) | Eastern and Western Word2Vec models used in the final notebook (RQ4) |
| [`models/prelim/`](https://github.com/RiniPi/LL5532X/tree/main/models/prelim) | Word2Vec models from the preliminary stage (Part 3) |

## Results

| Folder | Contents |
|:--|:--|
| [`results/final/eda/`](https://github.com/RiniPi/LL5532X/tree/main/results/final/eda) | Shifterator and fighting-words figures |
| [`results/final/rq1/`](https://github.com/RiniPi/LL5532X/tree/main/results/final/rq1) | Metadata leaderboard, text-arm scores, confusion matrices, top predictive terms |
| [`results/final/rq2/`](https://github.com/RiniPi/LL5532X/tree/main/results/final/rq2) | NMF topic descriptors, heatmap, topic-SVM results and feature importance |
| [`results/final/rq3/`](https://github.com/RiniPi/LL5532X/tree/main/results/final/rq3) | Cross-region scores, confusion matrices, misclassification analysis |
| [`results/final/rq4/`](https://github.com/RiniPi/LL5532X/tree/main/results/final/rq4) | Drift scores, t-SNE plots, nearest-neighbour tables, concordances |
| [`results/prelim/`](https://github.com/RiniPi/LL5532X/tree/main/results/prelim) | Outputs from the preliminary stage, including interactive t-SNE pages |

## Presentation

- [Slide deck (.pptx)](https://github.com/RiniPi/LL5532X/blob/main/docs/presentation/presentation_echr_art8.pptx)
- [Presentation outline (.docx)](https://github.com/RiniPi/LL5532X/blob/main/docs/presentation/presentation_outline.docx)
- [Chart data (.xlsx)](https://github.com/RiniPi/LL5532X/blob/main/docs/presentation/presentation_charts.xlsx)

## Repository layout

```
├── notebooks/                 Jupyter notebooks, numbered by project stage
├── data/                      Filtered Article 8 dataset and phrased sentences
├── models/{prelim,final}/     Word2Vec models
├── results/{prelim,final}/    Figures, CSVs and interactive outputs
├── docs/                      Proposal, references, presentation, course documents
├── web/                       Notebooks exported to Markdown (rendered on this site)
├── index.md, findings.md, …   Website pages
├── _config.yml                Website configuration
└── requirements.txt
```

## Running the notebooks

1. Open a notebook in Google Colab and mount your Google Drive.
2. Copy `data/` (and `models/` for RQ4) into the Drive folder the notebook expects, or change the path variables at the top of the notebook to point at this repository.
3. Install dependencies with `pip install -r requirements.txt`. Most notebooks also install what they need in their first cells.

Some large interactive outputs (ScatterText pages for RQ2 and RQ3) are not stored in the repository because of their size. Re-run the relevant notebook cells to regenerate them.
