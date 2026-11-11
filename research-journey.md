---
title: Research journey
nav_order: 4
description: "How the project's questions and design decisions changed from the original proposal to the final findings."
---

# Research journey: from proposal to findings
{: .no_toc }

How the team's thinking changed: what we planned, what the data showed us, and which decisions followed. Read it alongside the [findings](findings.md).
{: .fs-5 .fw-300 }

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

---

## 1. Where we started

The proposal asked one fairly classic legal-judgment-prediction (LJP) question:

> *How do textual features in Article 8 judgments predict findings of violation?*

The plan followed Aletras et al. (2016) and Medvedeva et al. (2020): extract sections, vectorise n-grams, fit an interpretable linear model (Logistic Regression), compare it against a DistilBERT baseline, and report F1. Two "comparative dimensions" were set aside as possible extensions: **temporal** (pre- vs. post-GDPR) and **jurisdictional** (by respondent state).

The motivation was jurisprudential, not only technical. **If outcomes are predictable from text, the Court's reasoning is coherent. If they are not, that is itself a finding about how determinate Article 8 is.** The final deck keeps this framing as its *Key Assumption*: a failure to predict counts as a legally meaningful result.

## 2. What the data told us (the EDA turn)

Exploratory analysis changed the project more than any model did. It turned up four findings, and each one became a design rule:

| EDA finding | Evidence | Design rule it produced |
|---|---|---|
| **Severe class imbalance** | 78% violation (982 V / 283 NV). The admissibility filter produces survivor bias, because weak cases never reach a judgment | A dummy "always violation" classifier scores about 78% accuracy, so **use macro-F1, not accuracy**. Use balanced class weights throughout |
| **Geographic concentration** | Russia: 342 cases (27%), about 94% violation, more than twice Ukraine's count | **Treat respondent country as a bias risk**, not just a feature. Watch for a "Russia effect" |
| **Structural leakage** | Most common article pairings: 8+41 (91 cases) and 8+35+41 (76). Article 41 (just satisfaction) appears only when a violation is found | The document already encodes its outcome. **Remedy language must be separated from reasoning**, and Facts and Law must be tested separately |
| **Genuine narrative signal** | Shifterator word-shift and ScatterText: violation facts use coercive state vocabulary (*detention, surveillance, seizure, expulsion*). No-violation facts use procedural and domestic-remedy language | There is a real signal to find. **Test the Facts section alone** as an honest, leakage-free estimate. Build a **custom stop-word list** for universal doctrinal terms ("necessary", "margin") |

## 3. How the questions multiplied

Each answer exposed the next question:

```
RQ1  Can we predict outcomes? Does text beat metadata?
 │   → Yes, but metadata learns geography, and the Law section leaks the remedy.
 ▼
RQ2  What themes actually drive outcomes?          (NMF topic modelling)
 │   → Some themes are artefacts (remedies, Russia), some are settled law,
 │     some are genuinely contested (about 50/50).
 ▼
RQ3  Do those patterns transfer across regions?     (train East → test West, and the reverse)
 │   → Only asymmetrically. Eastern and Western dockets argue about different things.
 ▼
RQ4  Do the same doctrinal words mean the same thing across regions?  (Word2Vec drift)
     → No. Shared vocabulary, different semantic neighbourhoods.
```

The proposal's *jurisdictional variance* extension grew into RQ3 and RQ4, which became the analytical centre of the project. The *temporal (GDPR)* extension was set aside, although the corpus was still limited to 2010–2023, a period chosen as the era when Article 8 was applied to digital spaces.

## 4. Key decisions and their rationale

### 4.1 Metric: macro-F1
With 78% of cases in one class, accuracy rewards ignoring the minority. Macro-F1 averages the per-class F1 scores, so the model is held equally accountable for predicting state *wins*.

### 4.2 Three-arm experiment for RQ1
To test "text vs. metadata" fairly, the team ran **metadata only**, **text only** (three variants) and **metadata + text**. Splitting text into **V1 Facts / V2 Law Full / V3 Law Clean** turned the leakage question into a controlled experiment. V1 has no remedy language and gives an honest lower bound. V2 is the full Law section. V3 has the remedy vocabulary stripped out to test whether lexical cleaning is enough.

**Lesson:** it is not. V3 still reaches 0.870. The model finds secondary leakage terms (*claimed, finds protocols, thereto law, central bank*). **Leakage is structural, not lexical.** The fix is to cut whole paragraphs out before feature extraction.

### 4.3 The unexpected result: metadata hurts
The hypothesis was *Combined > Text > Metadata*. The result was the reverse ordering for Combined vs. Text (0.828 < 0.870). The respondent-country prior overrode the text model's no-violation signal, and no-violation recall fell from 74% to 62%. The team read this as a lesson about **imbalance amplification**, not as evidence that metadata is useless (0.649 still beats chance).

### 4.4 Choosing k for NMF
The coherence curve peaked at **k = 7** (0.0524), but 7 topics merged legally distinct areas, for example surveillance and deportation. The team **deliberately chose k = 40**, trading coherence for legal granularity, and recorded it as a conscious trade-off rather than an optimum. For RQ3, NMF was fixed at **k = 20** to keep topics stable across regional sub-corpora.

### 4.5 Naive Bayes as a diagnostic, not a contender
NB was included in RQ2 to show *algorithmic fit*. Its weaker performance on NMF features is consistent with topic weights being correlated, which breaks NB's conditional-independence assumption.

### 4.6 Interpretability over raw performance
Linear models (LR, Linear SVM) were kept throughout because the **research question is about reasons**, not scores. Feature weights let the team check *why* a prediction was made, and that check is how leakage, the Russia effect and the regional divergence were found.

### 4.7 SMOTE, used sparingly
SMOTE was applied **only** to the low-dimensional metadata and topic features. It was avoided on high-dimensional raw text to prevent overfitting to synthetic points.

### 4.8 East / West split
To get past "who is being sued" toward "what the legal reasoning is", the Court's docket was split into Eastern and Western blocs following an external country list (see [RQ3](findings.md#rq3-do-patterns-transfer-across-regions)). Country overlap between train and test was checked and is zero. **Turkey** was held out as a *boundary state* side experiment.

### 4.9 Why DistilBERT was dropped
The final scope used **CPU-feasible** techniques. Contextual models moved from "baseline" to "future work". The RQ2 and RQ4 findings, especially the No-Violation paradox, became the argument for *why* contextual models are needed.

## 5. The intellectual arc

The project began by asking **"Can we predict?"** and ended by asking **"What exactly does prediction reveal?"** Three ideas run through it:

1. **The No-Violation paradox.** The vocabulary of rights protection and the vocabulary of judicial restraint are the same words. *Schalk and Kopf v. Austria* talks about "family life", "proportionality" and "discrimination" and still finds no violation. Bag-of-words cannot see the difference, because the outcome depends on the **arrangement** of words, not their presence (*performative syntax*).
2. **Geographic epistemology.** The algorithm does not learn a universal Article 8. It learns the docket it was trained on, whether that is physical state coercion in the East or proportionality balancing in the West.
3. **Technology as revealer (Heidegger).** The model exposes patterns already present in the law and also changes how we see them. Building legal AI encodes a region's socio-political history, so "neutral" law is not achievable.
