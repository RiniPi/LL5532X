# Predicting Privacy: An NLP Analysis of ECHR Article 8 Judgments

**LL5532X Group Project (Data Investigation), Group 1**
Amanda Lim Jin Mei · Nicha Nutchayangkul · Rini Pinasthika · Sophia Emily Lin

Can the outcome of a European Court of Human Rights judgment under **Article 8** (right to respect for private and family life, home and correspondence) be predicted from its text? This project studies 1,265 Article 8 judgments from 2010 to 2023 (982 Violation, 283 No Violation) using supervised classification, topic modelling and word embeddings.

**Start here:** the [final web presentation](web/final/LL5532X_Pt_5_Final_Project_Notebook_Group_1_v4.md), which is the final notebook rendered as Markdown.

## Research questions

| | Question | Methods |
|---|---|---|
| **RQ1** | Does judgment text outperform structured metadata (respondent country, year, importance level) in predicting outcomes, and which section (Facts vs Law) carries the stronger signal? | TF-IDF, SVM / logistic regression, SMOTE, macro F1 |
| **RQ2** | What are the dominant themes across the Article 8 corpus, and can topic weights act as predictive features? | NMF topic modelling, ScatterText, Shifterator, fighting words |
| **RQ3** | Do models trained on one regional bloc (Eastern or Western Europe) generalise to the other? | Cross-region train/test, misclassification analysis |
| **RQ4** | Why is there a generalisation gap? Do core doctrinal terms carry different meanings in each region? | Word2vec embeddings, semantic drift, t-SNE, concordances |

## Repository structure

```
├── notebooks/                       Jupyter notebooks, numbered by project stage
│   ├── 01_eda_v3.ipynb                  Part 2: exploratory data analysis
│   ├── 03_modeling_prelim_v5.ipynb      Part 3: modelling and preliminary results
│   ├── 04_presentation_visuals.ipynb    Part 4: charts for the in-class presentation
│   ├── 05_final_project_v4.ipynb        Part 5: final project notebook (main)
│   └── 05_final_project_pycaret_v1.ipynb Part 5: PyCaret model comparison
├── data/                            Filtered Article 8 dataset and phrased sentences
├── models/
│   ├── prelim/                          Word2vec models from Part 3
│   └── final/                           Word2vec models used in the final notebook
├── results/
│   ├── prelim/                          Outputs from Part 3 (figures, CSVs, t-SNE HTML)
│   └── final/{eda,rq1,rq2,rq3,rq4}/     Outputs from the final notebook
├── docs/
│   ├── proposal/                        Project proposal and stop-word list
│   ├── references/                      Papers cited in the proposal (ref1–ref6)
│   ├── presentation/                    In-class slides, outline and chart data
│   └── course-admin/                    Grading criteria, feedback and meeting notes
├── web/                             Notebooks exported to Markdown (Part 6)
│   ├── final/                           Final notebook + its figure folder
│   └── pycaret_web_presentation_v1.md
└── requirements.txt
```

## Running the notebooks

The notebooks were written for **Google Colab** and read and write files through a mounted Google Drive (`/content/drive/MyDrive/LL…`). To run them:

1. Open a notebook in Colab and mount your Google Drive.
2. Copy `data/` (and `models/` for RQ4) into the Drive folder the notebook expects, or change the path variables at the top of the notebook to point at this repo.
3. Install dependencies: `pip install -r requirements.txt` (most notebooks also `pip install` what they need in their first cells).

Some large interactive outputs (ScatterText HTML for RQ2/RQ3) are not stored in the repo because of their size; re-run the relevant notebook cells to regenerate them.

## Data

The dataset (`data/echr_art8_filtered.parquet`) contains ECtHR judgments involving Article 8, filtered to 2010–2023, with Facts and Law sections, respondent state, importance level and outcome. Judgments are public documents published by the Court on [HUDOC](https://hudoc.echr.coe.int/).

## Note

This repository holds coursework for LL5532X. All rights reserved by the authors unless stated otherwise; please contact the group before reusing any material.
