Group No: 1 
Group Members: Amanda Lim Jin Mei, Nicha Nutchayangkul, Rini Pinasthika, Sophia Emily Lin
Course/Module: LL5532X
Title: Predicting Privacy: An NLP Analysis of ECHR Article 8 Judgments

# PROJECT OVERVIEW

PyCaret's compare_models() evaluates 14 classifiers by default. We categorise them into three tiers based on their native interpretability for legal research:

**Tier 1 — Fully Interpretable (White-Box)**
These models produce signed coefficients directly mappable to feature contributions, without any post-hoc tool:

* *Logistic Regression*	: Log-odds coefficients per feature; signed and class-specific
* *SVM — Linear Kernel* :Weight vector coefficients; signed, directly interpretable as feature importance per class
* *Ridge Classifier*	: Regularised linear coefficients; signed, similar to Logistic Regression
  Linear Discriminant Analysis	: Linear combination of features; directional coefficients available

**Tier 2 — Partially Interpretable (Grey-Box)**
These models offer some structural transparency, but the explanation is either unsigned, non-directional, or requires aggregation across many sub-models:

- *Decision Tree*	: Single tree is traceable, but depth and complexity make legal interpretation impractical beyond shallow trees
- *Naive Bayes*	: Feature likelihoods are interpretable per class, but the conditional independence assumption is legally unrealistic and distorts importance rankings

**Tier 3 — Black-Box (Not Interpretable for Our Purposes)**
These models may offer post-hoc approximations via tools like SHAP or permutation importance, but none produce the kind of native, directional, per-feature explanation required for legal interpretation:

* *Random Forest*	: Aggregates hundreds of trees by majority vote; no single coherent decision path; feature importance is unsigned (always positive)
* *Extra Trees*	: Same as Random Forest; higher variance reduction does not improve interpretability
* *Gradient Boosting*	: Sequential ensemble of weak learners; final prediction is a sum of hundreds of corrections with no traceable legal meaning
* *Light Gradient Boosting Machine (LightGBM)*	: Same as Gradient Boosting; faster training, equally opaque
* *Ada Boost Classifier*	: Weighted ensemble of weak classifiers; aggregation destroys directional transparency
* *K Neighbors Classifier*	: Instance-based; prediction depends on local neighbourhood, not a global weight structure
* *Quadratic Discriminant Analysis*	: Non-linear decision boundary; no signed linear coefficients
* *Dummy Classifier*	: Baseline only; no informative features

**The Random Forest Case — A Common Misconception**
Random Forest deserves specific attention because it is frequently described as "partially interpretable" through its built-in feature importance scores. This claim is technically true but misleading for legal research. Random Forest feature importance — whether measured by Gini impurity or permutation — tells you how much a feature contributed to reducing prediction error across all trees. What it cannot tell you is whether that feature increases or decreases the probability of a Violation finding, because importance scores are always non-negative.

In legal interpretation, direction is everything. Knowing that "Topic 3 — Surveillance and Data Retention" is important is not useful unless we know whether its presence makes a violation more or less likely. SVM and Logistic Regression answer this directly through their weight vectors. Random Forest cannot, and SHAP-based post-hoc explanations introduce a layer of approximation that weakens the analytical rigour expected in legal scholarship.

**Implication for Model Selection**
Of PyCaret's 14 default models, only 4 are fully interpretable by our definition — Logistic Regression, SVM (Linear Kernel), Ridge Classifier, and Linear Discriminant Analysis. Among these, SVM and Logistic Regression are our primary models, selected on both theoretical grounds (Aletras et al., 2016; Medvedeva et al., 2020) and, as the PyCaret results will confirm, on empirical grounds as the top-performing interpretable models across all feature variants. The remaining 10 models serve a single purpose in this notebook: to establish the performance ceiling against which our interpretable models are benchmarked, and to quantify the cost of the interpretability-performance tradeoff we deliberately accept.



    


# **RQ1 : Text vs. Metadata**

## **Metadata**

**Step 1: Preprocessing**

Features consist of four categorical columns (`respondent_country`, `importance`, `court`, `typedescription`) and three numeric columns (`year`, `has_dissent`, `is_english`). PyCaret's `setup()` is told explicitly which columns belong to each type, triggering one-hot encoding for categoricals and standard scaling for numerics. SMOTE (`fix_imbalance=True`) is applied to oversample the minority class on the training folds, since metadata features are low-dimensional and SMOTE can safely interpolate between real samples without distorting feature semantics. 

In this notebook, We use a Linear SVM implemented as CalibratedClassifierCV(LinearSVC), which produces the same linear decision boundary as a standard LinearSVC but adds probability calibration required by PyCaret's AUC computation



### Pycaret

**Step 2: Compare Models**

<style type="text/css">
#T_1150f_row9_col8, #T_1150f_row10_col8, #T_1150f_row11_col8 {
  background-color: #ffe066;
  font-weight: bold;
}
</style>
<table id="T_1150f">
  <caption>Metadata — All Models sorted by F1 Macro (highlighted: NB · LR · SVM)</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_1150f_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_1150f_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_1150f_level0_col2" class="col_heading level0 col2" >AUC</th>
      <th id="T_1150f_level0_col3" class="col_heading level0 col3" >Recall</th>
      <th id="T_1150f_level0_col4" class="col_heading level0 col4" >Prec.</th>
      <th id="T_1150f_level0_col5" class="col_heading level0 col5" >F1</th>
      <th id="T_1150f_level0_col6" class="col_heading level0 col6" >Kappa</th>
      <th id="T_1150f_level0_col7" class="col_heading level0 col7" >MCC</th>
      <th id="T_1150f_level0_col8" class="col_heading level0 col8" >F1 Macro</th>
      <th id="T_1150f_level0_col9" class="col_heading level0 col9" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1150f_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_1150f_row0_col0" class="data row0 col0" >Gradient Boosting Classifier</td>
      <td id="T_1150f_row0_col1" class="data row0 col1" >0.7842</td>
      <td id="T_1150f_row0_col2" class="data row0 col2" >0.7555</td>
      <td id="T_1150f_row0_col3" class="data row0 col3" >0.7842</td>
      <td id="T_1150f_row0_col4" class="data row0 col4" >0.7844</td>
      <td id="T_1150f_row0_col5" class="data row0 col5" >0.7828</td>
      <td id="T_1150f_row0_col6" class="data row0 col6" >0.3731</td>
      <td id="T_1150f_row0_col7" class="data row0 col7" >0.3760</td>
      <td id="T_1150f_row0_col8" class="data row0 col8" >0.8612</td>
      <td id="T_1150f_row0_col9" class="data row0 col9" >0.0340</td>
    </tr>
    <tr>
      <th id="T_1150f_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_1150f_row1_col0" class="data row1 col0" >Light Gradient Boosting Machine</td>
      <td id="T_1150f_row1_col1" class="data row1 col1" >0.7593</td>
      <td id="T_1150f_row1_col2" class="data row1 col2" >0.7186</td>
      <td id="T_1150f_row1_col3" class="data row1 col3" >0.7593</td>
      <td id="T_1150f_row1_col4" class="data row1 col4" >0.7590</td>
      <td id="T_1150f_row1_col5" class="data row1 col5" >0.7576</td>
      <td id="T_1150f_row1_col6" class="data row1 col6" >0.3010</td>
      <td id="T_1150f_row1_col7" class="data row1 col7" >0.3035</td>
      <td id="T_1150f_row1_col8" class="data row1 col8" >0.8451</td>
      <td id="T_1150f_row1_col9" class="data row1 col9" >0.2960</td>
    </tr>
    <tr>
      <th id="T_1150f_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_1150f_row2_col0" class="data row2 col0" >Random Forest Classifier</td>
      <td id="T_1150f_row2_col1" class="data row2 col1" >0.7559</td>
      <td id="T_1150f_row2_col2" class="data row2 col2" >0.7181</td>
      <td id="T_1150f_row2_col3" class="data row2 col3" >0.7559</td>
      <td id="T_1150f_row2_col4" class="data row2 col4" >0.7608</td>
      <td id="T_1150f_row2_col5" class="data row2 col5" >0.7564</td>
      <td id="T_1150f_row2_col6" class="data row2 col6" >0.3055</td>
      <td id="T_1150f_row2_col7" class="data row2 col7" >0.3084</td>
      <td id="T_1150f_row2_col8" class="data row2 col8" >0.8415</td>
      <td id="T_1150f_row2_col9" class="data row2 col9" >0.0560</td>
    </tr>
    <tr>
      <th id="T_1150f_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_1150f_row3_col0" class="data row3 col0" >Extra Trees Classifier</td>
      <td id="T_1150f_row3_col1" class="data row3 col1" >0.7525</td>
      <td id="T_1150f_row3_col2" class="data row3 col2" >0.6647</td>
      <td id="T_1150f_row3_col3" class="data row3 col3" >0.7525</td>
      <td id="T_1150f_row3_col4" class="data row3 col4" >0.7656</td>
      <td id="T_1150f_row3_col5" class="data row3 col5" >0.7566</td>
      <td id="T_1150f_row3_col6" class="data row3 col6" >0.3172</td>
      <td id="T_1150f_row3_col7" class="data row3 col7" >0.3211</td>
      <td id="T_1150f_row3_col8" class="data row3 col8" >0.8372</td>
      <td id="T_1150f_row3_col9" class="data row3 col9" >0.0460</td>
    </tr>
    <tr>
      <th id="T_1150f_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_1150f_row4_col0" class="data row4 col0" >Ada Boost Classifier</td>
      <td id="T_1150f_row4_col1" class="data row4 col1" >0.7537</td>
      <td id="T_1150f_row4_col2" class="data row4 col2" >0.7595</td>
      <td id="T_1150f_row4_col3" class="data row4 col3" >0.7537</td>
      <td id="T_1150f_row4_col4" class="data row4 col4" >0.7793</td>
      <td id="T_1150f_row4_col5" class="data row4 col5" >0.7605</td>
      <td id="T_1150f_row4_col6" class="data row4 col6" >0.3439</td>
      <td id="T_1150f_row4_col7" class="data row4 col7" >0.3538</td>
      <td id="T_1150f_row4_col8" class="data row4 col8" >0.8347</td>
      <td id="T_1150f_row4_col9" class="data row4 col9" >0.0220</td>
    </tr>
    <tr>
      <th id="T_1150f_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_1150f_row5_col0" class="data row5 col0" >Decision Tree Classifier</td>
      <td id="T_1150f_row5_col1" class="data row5 col1" >0.7299</td>
      <td id="T_1150f_row5_col2" class="data row5 col2" >0.6037</td>
      <td id="T_1150f_row5_col3" class="data row5 col3" >0.7299</td>
      <td id="T_1150f_row5_col4" class="data row5 col4" >0.7461</td>
      <td id="T_1150f_row5_col5" class="data row5 col5" >0.7363</td>
      <td id="T_1150f_row5_col6" class="data row5 col6" >0.2645</td>
      <td id="T_1150f_row5_col7" class="data row5 col7" >0.2668</td>
      <td id="T_1150f_row5_col8" class="data row5 col8" >0.8216</td>
      <td id="T_1150f_row5_col9" class="data row5 col9" >0.0300</td>
    </tr>
    <tr>
      <th id="T_1150f_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_1150f_row6_col0" class="data row6 col0" >K Neighbors Classifier</td>
      <td id="T_1150f_row6_col1" class="data row6 col1" >0.7073</td>
      <td id="T_1150f_row6_col2" class="data row6 col2" >0.6766</td>
      <td id="T_1150f_row6_col3" class="data row6 col3" >0.7073</td>
      <td id="T_1150f_row6_col4" class="data row6 col4" >0.7334</td>
      <td id="T_1150f_row6_col5" class="data row6 col5" >0.7168</td>
      <td id="T_1150f_row6_col6" class="data row6 col6" >0.2243</td>
      <td id="T_1150f_row6_col7" class="data row6 col7" >0.2284</td>
      <td id="T_1150f_row6_col8" class="data row6 col8" >0.8038</td>
      <td id="T_1150f_row6_col9" class="data row6 col9" >0.0340</td>
    </tr>
    <tr>
      <th id="T_1150f_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_1150f_row7_col0" class="data row7 col0" >Linear Discriminant Analysis</td>
      <td id="T_1150f_row7_col1" class="data row7 col1" >0.7085</td>
      <td id="T_1150f_row7_col2" class="data row7 col2" >0.7661</td>
      <td id="T_1150f_row7_col3" class="data row7 col3" >0.7085</td>
      <td id="T_1150f_row7_col4" class="data row7 col4" >0.7724</td>
      <td id="T_1150f_row7_col5" class="data row7 col5" >0.7259</td>
      <td id="T_1150f_row7_col6" class="data row7 col6" >0.3021</td>
      <td id="T_1150f_row7_col7" class="data row7 col7" >0.3223</td>
      <td id="T_1150f_row7_col8" class="data row7 col8" >0.7939</td>
      <td id="T_1150f_row7_col9" class="data row7 col9" >0.0320</td>
    </tr>
    <tr>
      <th id="T_1150f_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_1150f_row8_col0" class="data row8 col0" >Ridge Classifier</td>
      <td id="T_1150f_row8_col1" class="data row8 col1" >0.7051</td>
      <td id="T_1150f_row8_col2" class="data row8 col2" >0.7663</td>
      <td id="T_1150f_row8_col3" class="data row8 col3" >0.7051</td>
      <td id="T_1150f_row8_col4" class="data row8 col4" >0.7702</td>
      <td id="T_1150f_row8_col5" class="data row8 col5" >0.7234</td>
      <td id="T_1150f_row8_col6" class="data row8 col6" >0.2971</td>
      <td id="T_1150f_row8_col7" class="data row8 col7" >0.3167</td>
      <td id="T_1150f_row8_col8" class="data row8 col8" >0.7913</td>
      <td id="T_1150f_row8_col9" class="data row8 col9" >0.2740</td>
    </tr>
    <tr>
      <th id="T_1150f_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_1150f_row9_col0" class="data row9 col0" >Logistic Regression</td>
      <td id="T_1150f_row9_col1" class="data row9 col1" >0.7040</td>
      <td id="T_1150f_row9_col2" class="data row9 col2" >0.7620</td>
      <td id="T_1150f_row9_col3" class="data row9 col3" >0.7040</td>
      <td id="T_1150f_row9_col4" class="data row9 col4" >0.7674</td>
      <td id="T_1150f_row9_col5" class="data row9 col5" >0.7220</td>
      <td id="T_1150f_row9_col6" class="data row9 col6" >0.2911</td>
      <td id="T_1150f_row9_col7" class="data row9 col7" >0.3097</td>
      <td id="T_1150f_row9_col8" class="data row9 col8" >0.7910</td>
      <td id="T_1150f_row9_col9" class="data row9 col9" >0.3720</td>
    </tr>
    <tr>
      <th id="T_1150f_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_1150f_row10_col0" class="data row10 col0" >Naive Bayes</td>
      <td id="T_1150f_row10_col1" class="data row10 col1" >0.6949</td>
      <td id="T_1150f_row10_col2" class="data row10 col2" >0.7094</td>
      <td id="T_1150f_row10_col3" class="data row10 col3" >0.6949</td>
      <td id="T_1150f_row10_col4" class="data row10 col4" >0.7452</td>
      <td id="T_1150f_row10_col5" class="data row10 col5" >0.7120</td>
      <td id="T_1150f_row10_col6" class="data row10 col6" >0.2461</td>
      <td id="T_1150f_row10_col7" class="data row10 col7" >0.2562</td>
      <td id="T_1150f_row10_col8" class="data row10 col8" >0.7888</td>
      <td id="T_1150f_row10_col9" class="data row10 col9" >0.5420</td>
    </tr>
    <tr>
      <th id="T_1150f_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_1150f_row11_col0" class="data row11 col0" >LinearSVC</td>
      <td id="T_1150f_row11_col1" class="data row11 col1" >0.5955</td>
      <td id="T_1150f_row11_col2" class="data row11 col2" >0.7008</td>
      <td id="T_1150f_row11_col3" class="data row11 col3" >0.5955</td>
      <td id="T_1150f_row11_col4" class="data row11 col4" >0.7027</td>
      <td id="T_1150f_row11_col5" class="data row11 col5" >0.6273</td>
      <td id="T_1150f_row11_col6" class="data row11 col6" >0.1143</td>
      <td id="T_1150f_row11_col7" class="data row11 col7" >0.1285</td>
      <td id="T_1150f_row11_col8" class="data row11 col8" >0.7007</td>
      <td id="T_1150f_row11_col9" class="data row11 col9" >0.0920</td>
    </tr>
  </tbody>
</table>
Across all 14 models, tree ensemble methods dominate the leaderboard. Gradient Boosting achieves the highest F1 Macro (0.8612). These models benefit from their ability to learn non-linear interactions between metadata features — for example, the combination of country, court type, and case importance can jointly signal outcome in ways that a single linear boundary cannot capture.

Among the three interpretable models of interest, Logistic Regression (0.7910) and Naive Bayes (0.7888) perform similarly, while the custom LinearSVC (0.7007) perform the worst. This occurs because LinearSVC on a mixed high-dimensional OHE + numeric feature space, without feature scaling at the estimator level, fails to find a valid separating hyperplane. LR handles this better because its L2 regularisation via `lbfgs` is more numerically stable under this setup.

NB performing comparably to LR on metadata is expected: after SMOTE resampling, class distributions are more balanced, partially closing the gap that typically disadvantages NB. However, the independence assumption still limits NB's ceiling, which is why it trails LR slightly.



**Step 3: Fine Tune Model**


    STEP 3: Tuning LR, NB, SVM (optimize = F1 Macro)...
    
    ── Metadata: Tuning Summary (F1 Macro, CV) ──
                  Model  Tuned F1 Macro  Δ F1 Macro Improved
    Logistic Regression          0.8561      0.0200        ✓
            Naive Bayes          0.8453      0.0000        —
    Linear SVM (custom)          0.9247      0.0619        ✓

After tuning, LR improves meaningfully from to 0.856. SVM improves biggest to 0.9247. Meanwhile Naive Bayes does not improve from the base result. 

Tuned-SVM holds the best model. 



**Step 4: Hyperparameter Extraction**


    =======================================================
      Logistic Regression  (LogisticRegression)
    =======================================================
      C                                   2.214
      class_weight                        balanced
      dual                                False
      fit_intercept                       True
      intercept_scaling                   1
      l1_ratio                            None
      max_iter                            1000
      multi_class                         auto
      n_jobs                              None
      penalty                             l2
      random_state                        42
      solver                              lbfgs
      tol                                 0.0001
      verbose                             0
      warm_start                          False
    
    =======================================================
      Naive Bayes  (GaussianNB)
    =======================================================
      priors                              None
      var_smoothing                       1e-09
    
    =======================================================
      Linear SVM (custom)  (CalibratedClassifierCV → LinearSVC)
    =======================================================
      C                                   0.001
      class_weight                        balanced
      dual                                warn
      fit_intercept                       True
      intercept_scaling                   1
      loss                                squared_hinge
      max_iter                            3000
      multi_class                         ovr
      penalty                             l2
      random_state                        42
      tol                                 0.0001
      verbose                             0

The most impactful tuned parameters are: 
**LR** — `C=2.214` (stronger regularisation than default `C=1.0` reduces overfitting on sparse OHE features) and `class_weight='balanced'` (corrects residual class imbalance not fully resolved by SMOTE). 
**NB** — `var_smoothing=1e-09` (default retained, confirming GaussianNB was already well-calibrated). 
**SVM** — `C=0.001` (tuned c-value improved performance). 



## Text


**Step 1: Preprocessing**

Raw text from the facts section (V1) and the law clean section (V3) are each vectorised using TF-IDF with `max_features=10,000`, unigram and bigram tokens (`ngram_range=(1,2)`), and `sublinear_tf=True` to apply log-normalisation and dampen the dominance of very frequent legal terms. Custom domain stop words specific to each section are applied. Imbalance is handled via `class_weight='balanced'` inside the SVM estimator, and LR uses its default balanced weighting. No SMOTE is applied since interpolating across 10,000 sparse TF-IDF dimensions risks generating unrealistic synthetic documents. 

    TF-IDF matrices ready.
      V1 Facts  → shape: (1265, 10000)
      V3 Law Clean → shape: (1265, 10000)



### Pycaret - Facts

**Step 2: Compare Models — V1 Facts**

<style type="text/css">
#T_eec8d_row0_col6 {
  background-color: #90ee90;
}
#T_eec8d_row1_col6 {
  background-color: #90ee90;
  background-color: #ffe066;
  font-weight: bold;
}
#T_eec8d_row5_col6, #T_eec8d_row11_col6 {
  background-color: #ffe066;
  font-weight: bold;
}
</style>
<table id="T_eec8d">
  <caption>V1 — Facts Section — All Models sorted by F1 Macro | 🟡 NB · LR · SVM (interpretable)  🟢 Best overall</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_eec8d_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_eec8d_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_eec8d_level0_col2" class="col_heading level0 col2" >AUC</th>
      <th id="T_eec8d_level0_col3" class="col_heading level0 col3" >Recall</th>
      <th id="T_eec8d_level0_col4" class="col_heading level0 col4" >Prec.</th>
      <th id="T_eec8d_level0_col5" class="col_heading level0 col5" >F1</th>
      <th id="T_eec8d_level0_col6" class="col_heading level0 col6" >F1 Macro</th>
      <th id="T_eec8d_level0_col7" class="col_heading level0 col7" >Kappa</th>
      <th id="T_eec8d_level0_col8" class="col_heading level0 col8" >MCC</th>
      <th id="T_eec8d_level0_col9" class="col_heading level0 col9" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_eec8d_level0_row0" class="row_heading level0 row0" >2</th>
      <td id="T_eec8d_row0_col0" class="data row0 col0" >Ridge Classifier</td>
      <td id="T_eec8d_row0_col1" class="data row0 col1" >0.8124</td>
      <td id="T_eec8d_row0_col2" class="data row0 col2" >0.8224</td>
      <td id="T_eec8d_row0_col3" class="data row0 col3" >0.8124</td>
      <td id="T_eec8d_row0_col4" class="data row0 col4" >0.7994</td>
      <td id="T_eec8d_row0_col5" class="data row0 col5" >0.7781</td>
      <td id="T_eec8d_row0_col6" class="data row0 col6" >0.8889</td>
      <td id="T_eec8d_row0_col7" class="data row0 col7" >0.3090</td>
      <td id="T_eec8d_row0_col8" class="data row0 col8" >0.3599</td>
      <td id="T_eec8d_row0_col9" class="data row0 col9" >0.4300</td>
    </tr>
    <tr>
      <th id="T_eec8d_level0_row1" class="row_heading level0 row1" >11</th>
      <td id="T_eec8d_row1_col0" class="data row1 col0" >LinearSVC</td>
      <td id="T_eec8d_row1_col1" class="data row1 col1" >0.8158</td>
      <td id="T_eec8d_row1_col2" class="data row1 col2" >0.8238</td>
      <td id="T_eec8d_row1_col3" class="data row1 col3" >0.8158</td>
      <td id="T_eec8d_row1_col4" class="data row1 col4" >0.8023</td>
      <td id="T_eec8d_row1_col5" class="data row1 col5" >0.7912</td>
      <td id="T_eec8d_row1_col6" class="data row1 col6" >0.8889</td>
      <td id="T_eec8d_row1_col7" class="data row1 col7" >0.3577</td>
      <td id="T_eec8d_row1_col8" class="data row1 col8" >0.3904</td>
      <td id="T_eec8d_row1_col9" class="data row1 col9" >0.5600</td>
    </tr>
    <tr>
      <th id="T_eec8d_level0_row2" class="row_heading level0 row2" >6</th>
      <td id="T_eec8d_row2_col0" class="data row2 col0" >Extra Trees Classifier</td>
      <td id="T_eec8d_row2_col1" class="data row2 col1" >0.8000</td>
      <td id="T_eec8d_row2_col2" class="data row2 col2" >0.8088</td>
      <td id="T_eec8d_row2_col3" class="data row2 col3" >0.8000</td>
      <td id="T_eec8d_row2_col4" class="data row2 col4" >0.7852</td>
      <td id="T_eec8d_row2_col5" class="data row2 col5" >0.7467</td>
      <td id="T_eec8d_row2_col6" class="data row2 col6" >0.8841</td>
      <td id="T_eec8d_row2_col7" class="data row2 col7" >0.2042</td>
      <td id="T_eec8d_row2_col8" class="data row2 col8" >0.2784</td>
      <td id="T_eec8d_row2_col9" class="data row2 col9" >0.5460</td>
    </tr>
    <tr>
      <th id="T_eec8d_level0_row3" class="row_heading level0 row3" >5</th>
      <td id="T_eec8d_row3_col0" class="data row3 col0" >Random Forest Classifier</td>
      <td id="T_eec8d_row3_col1" class="data row3 col1" >0.7944</td>
      <td id="T_eec8d_row3_col2" class="data row3 col2" >0.8075</td>
      <td id="T_eec8d_row3_col3" class="data row3 col3" >0.7944</td>
      <td id="T_eec8d_row3_col4" class="data row3 col4" >0.7753</td>
      <td id="T_eec8d_row3_col5" class="data row3 col5" >0.7363</td>
      <td id="T_eec8d_row3_col6" class="data row3 col6" >0.8812</td>
      <td id="T_eec8d_row3_col7" class="data row3 col7" >0.1725</td>
      <td id="T_eec8d_row3_col8" class="data row3 col8" >0.2459</td>
      <td id="T_eec8d_row3_col9" class="data row3 col9" >0.5320</td>
    </tr>
    <tr>
      <th id="T_eec8d_level0_row4" class="row_heading level0 row4" >9</th>
      <td id="T_eec8d_row4_col0" class="data row4 col0" >Light Gradient Boosting Machine</td>
      <td id="T_eec8d_row4_col1" class="data row4 col1" >0.7955</td>
      <td id="T_eec8d_row4_col2" class="data row4 col2" >0.7724</td>
      <td id="T_eec8d_row4_col3" class="data row4 col3" >0.7955</td>
      <td id="T_eec8d_row4_col4" class="data row4 col4" >0.7713</td>
      <td id="T_eec8d_row4_col5" class="data row4 col5" >0.7596</td>
      <td id="T_eec8d_row4_col6" class="data row4 col6" >0.8787</td>
      <td id="T_eec8d_row4_col7" class="data row4 col7" >0.2512</td>
      <td id="T_eec8d_row4_col8" class="data row4 col8" >0.2924</td>
      <td id="T_eec8d_row4_col9" class="data row4 col9" >1.1340</td>
    </tr>
    <tr>
      <th id="T_eec8d_level0_row5" class="row_heading level0 row5" >1</th>
      <td id="T_eec8d_row5_col0" class="data row5 col0" >Logistic Regression</td>
      <td id="T_eec8d_row5_col1" class="data row5 col1" >0.7831</td>
      <td id="T_eec8d_row5_col2" class="data row5 col2" >0.8146</td>
      <td id="T_eec8d_row5_col3" class="data row5 col3" >0.7831</td>
      <td id="T_eec8d_row5_col4" class="data row5 col4" >0.7428</td>
      <td id="T_eec8d_row5_col5" class="data row5 col5" >0.7014</td>
      <td id="T_eec8d_row5_col6" class="data row5 col6" >0.8768</td>
      <td id="T_eec8d_row5_col7" class="data row5 col7" >0.0662</td>
      <td id="T_eec8d_row5_col8" class="data row5 col8" >0.1324</td>
      <td id="T_eec8d_row5_col9" class="data row5 col9" >0.4600</td>
    </tr>
    <tr>
      <th id="T_eec8d_level0_row6" class="row_heading level0 row6" >10</th>
      <td id="T_eec8d_row6_col0" class="data row6 col0" >K Neighbors Classifier</td>
      <td id="T_eec8d_row6_col1" class="data row6 col1" >0.7740</td>
      <td id="T_eec8d_row6_col2" class="data row6 col2" >0.5398</td>
      <td id="T_eec8d_row6_col3" class="data row6 col3" >0.7740</td>
      <td id="T_eec8d_row6_col4" class="data row6 col4" >0.6247</td>
      <td id="T_eec8d_row6_col5" class="data row6 col5" >0.6794</td>
      <td id="T_eec8d_row6_col6" class="data row6 col6" >0.8724</td>
      <td id="T_eec8d_row6_col7" class="data row6 col7" >0.0011</td>
      <td id="T_eec8d_row6_col8" class="data row6 col8" >-0.0017</td>
      <td id="T_eec8d_row6_col9" class="data row6 col9" >0.4360</td>
    </tr>
    <tr>
      <th id="T_eec8d_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_eec8d_row7_col0" class="data row7 col0" >Gradient Boosting Classifier</td>
      <td id="T_eec8d_row7_col1" class="data row7 col1" >0.7853</td>
      <td id="T_eec8d_row7_col2" class="data row7 col2" >0.7737</td>
      <td id="T_eec8d_row7_col3" class="data row7 col3" >0.7853</td>
      <td id="T_eec8d_row7_col4" class="data row7 col4" >0.7537</td>
      <td id="T_eec8d_row7_col5" class="data row7 col5" >0.7520</td>
      <td id="T_eec8d_row7_col6" class="data row7 col6" >0.8719</td>
      <td id="T_eec8d_row7_col7" class="data row7 col7" >0.2302</td>
      <td id="T_eec8d_row7_col8" class="data row7 col8" >0.2588</td>
      <td id="T_eec8d_row7_col9" class="data row7 col9" >3.0780</td>
    </tr>
    <tr>
      <th id="T_eec8d_level0_row8" class="row_heading level0 row8" >3</th>
      <td id="T_eec8d_row8_col0" class="data row8 col0" >Linear Discriminant Analysis</td>
      <td id="T_eec8d_row8_col1" class="data row8 col1" >0.7763</td>
      <td id="T_eec8d_row8_col2" class="data row8 col2" >0.7790</td>
      <td id="T_eec8d_row8_col3" class="data row8 col3" >0.7763</td>
      <td id="T_eec8d_row8_col4" class="data row8 col4" >0.7776</td>
      <td id="T_eec8d_row8_col5" class="data row8 col5" >0.7632</td>
      <td id="T_eec8d_row8_col6" class="data row8 col6" >0.8567</td>
      <td id="T_eec8d_row8_col7" class="data row8 col7" >0.3107</td>
      <td id="T_eec8d_row8_col8" class="data row8 col8" >0.3341</td>
      <td id="T_eec8d_row8_col9" class="data row8 col9" >0.6200</td>
    </tr>
    <tr>
      <th id="T_eec8d_level0_row9" class="row_heading level0 row9" >8</th>
      <td id="T_eec8d_row9_col0" class="data row9 col0" >Ada Boost Classifier</td>
      <td id="T_eec8d_row9_col1" class="data row9 col1" >0.7537</td>
      <td id="T_eec8d_row9_col2" class="data row9 col2" >0.7011</td>
      <td id="T_eec8d_row9_col3" class="data row9 col3" >0.7537</td>
      <td id="T_eec8d_row9_col4" class="data row9 col4" >0.7293</td>
      <td id="T_eec8d_row9_col5" class="data row9 col5" >0.7380</td>
      <td id="T_eec8d_row9_col6" class="data row9 col6" >0.8475</td>
      <td id="T_eec8d_row9_col7" class="data row9 col7" >0.2109</td>
      <td id="T_eec8d_row9_col8" class="data row9 col8" >0.2158</td>
      <td id="T_eec8d_row9_col9" class="data row9 col9" >0.8840</td>
    </tr>
    <tr>
      <th id="T_eec8d_level0_row10" class="row_heading level0 row10" >4</th>
      <td id="T_eec8d_row10_col0" class="data row10 col0" >Decision Tree Classifier</td>
      <td id="T_eec8d_row10_col1" class="data row10 col1" >0.7311</td>
      <td id="T_eec8d_row10_col2" class="data row10 col2" >0.5740</td>
      <td id="T_eec8d_row10_col3" class="data row10 col3" >0.7311</td>
      <td id="T_eec8d_row10_col4" class="data row10 col4" >0.7234</td>
      <td id="T_eec8d_row10_col5" class="data row10 col5" >0.7266</td>
      <td id="T_eec8d_row10_col6" class="data row10 col6" >0.8287</td>
      <td id="T_eec8d_row10_col7" class="data row10 col7" >0.2016</td>
      <td id="T_eec8d_row10_col8" class="data row10 col8" >0.2027</td>
      <td id="T_eec8d_row10_col9" class="data row10 col9" >0.4860</td>
    </tr>
    <tr>
      <th id="T_eec8d_level0_row11" class="row_heading level0 row11" >0</th>
      <td id="T_eec8d_row11_col0" class="data row11 col0" >Naive Bayes</td>
      <td id="T_eec8d_row11_col1" class="data row11 col1" >0.7017</td>
      <td id="T_eec8d_row11_col2" class="data row11 col2" >0.5452</td>
      <td id="T_eec8d_row11_col3" class="data row11 col3" >0.7017</td>
      <td id="T_eec8d_row11_col4" class="data row11 col4" >0.6867</td>
      <td id="T_eec8d_row11_col5" class="data row11 col5" >0.6921</td>
      <td id="T_eec8d_row11_col6" class="data row11 col6" >0.8110</td>
      <td id="T_eec8d_row11_col7" class="data row11 col7" >0.0947</td>
      <td id="T_eec8d_row11_col8" class="data row11 col8" >0.0962</td>
      <td id="T_eec8d_row11_col9" class="data row11 col9" >0.4180</td>
    </tr>
  </tbody>
</table>
On the facts section, SVM and Ridge Classifier share the top F1 Macro at 0.8889, with LR close at 0.8768. This is a strong result for the interpretable models. This behaviour is well-known in text classification: TF-IDF features are sparse and high-dimensional, which is exactly the condition where linear models thrive. Each feature is an independent token count, closely matching the assumptions of linear classifiers.

NB scores 0.8110 — clearly the lowest among the three target models. This confirms the expected baseline behaviour. NB's lower ceiling on TF-IDF data relative to SVM is because it models each feature independently without accounting for the relative discriminative power between features, while SVM finds a global optimal margin.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
**Step 3: Fine Tune Model**

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model ID</th>
      <th>Sklearn Class</th>
      <th>Tuned F1 Macro</th>
      <th>Kappa</th>
      <th>MCC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nb</td>
      <td>Naive Bayes (ComplementNB)</td>
      <td>0.8306</td>
      <td>-0.1065</td>
      <td>-0.1317</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lr</td>
      <td>Logistic Regression</td>
      <td>0.8880</td>
      <td>0.3091</td>
      <td>0.3563</td>
    </tr>
    <tr>
      <th>2</th>
      <td>svm</td>
      <td>Linear SVM (custom)</td>
      <td>0.8935</td>
      <td>0.3551</td>
      <td>0.4030</td>
    </tr>
  </tbody>
</table>
On V1 Facts, tuning yields NB: 0.8306, LR: 0.8880, SVM: 0.8935. NB benefits most from switching to ComplementNB , which is designed for imbalanced sparse text and substantially outperforms the default GaussianNB on TF-IDF features. LR and SVM were already near-optimal at default settings.



**Step 4: Hyperparameter Extraction**


    ==============================================================
      HYPERPARAMETERS — V1 — Facts Section
    ==============================================================
    
      ── Naive Bayes  (ComplementNB)
         alpha                                    0.01
         class_prior                              None
         fit_prior                                True
         force_alpha                              True
         norm                                     False
    
      ── Logistic Regression  (LogisticRegression)
         C                                        9.885
         class_weight                             {}
         dual                                     False
         fit_intercept                            True
         intercept_scaling                        1
         l1_ratio                                 None
         max_iter                                 1000
         multi_class                              auto
         n_jobs                                   None
         penalty                                  l2
         random_state                             42
         solver                                   lbfgs
         tol                                      0.0001
         verbose                                  0
         warm_start                               False
    
      ── Linear SVM (custom)  (CalibratedClassifierCV → LinearSVC)
         C                                        100
         class_weight                             balanced
         dual                                     warn
         fit_intercept                            True
         intercept_scaling                        1
         loss                                     squared_hinge
         max_iter                                 2000
         multi_class                              ovr
         penalty                                  l2
         random_state                             42
         tol                                      0.0001
         verbose                                  0

Key parameters to transfer: **V1 NB** — change to ComplementNB alpha0.01 is the most impactful change; it departs significantly from the default and drove the largest relative gain. **V1 LR** — `C=9.885`; higher C reduces regularisation and fits the large vocabulary better. **V1 SVM** — `C=100`; higher C creates a tighter margin, beneficial for clean sparse text.



### Pycaret - Law Clean

**Step 2: Compare Models — V3 Law Clean**

<style type="text/css">
#T_4f476_row0_col6 {
  background-color: #90ee90;
}
#T_4f476_row5_col6, #T_4f476_row9_col6, #T_4f476_row10_col6 {
  background-color: #ffe066;
  font-weight: bold;
}
</style>
<table id="T_4f476">
  <caption>V3 — Law Clean — All Models sorted by F1 Macro | 🟡 NB · LR · SVM (interpretable)  🟢 Best overall</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_4f476_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_4f476_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_4f476_level0_col2" class="col_heading level0 col2" >AUC</th>
      <th id="T_4f476_level0_col3" class="col_heading level0 col3" >Recall</th>
      <th id="T_4f476_level0_col4" class="col_heading level0 col4" >Prec.</th>
      <th id="T_4f476_level0_col5" class="col_heading level0 col5" >F1</th>
      <th id="T_4f476_level0_col6" class="col_heading level0 col6" >F1 Macro</th>
      <th id="T_4f476_level0_col7" class="col_heading level0 col7" >Kappa</th>
      <th id="T_4f476_level0_col8" class="col_heading level0 col8" >MCC</th>
      <th id="T_4f476_level0_col9" class="col_heading level0 col9" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_4f476_level0_row0" class="row_heading level0 row0" >9</th>
      <td id="T_4f476_row0_col0" class="data row0 col0" >Light Gradient Boosting Machine</td>
      <td id="T_4f476_row0_col1" class="data row0 col1" >0.9480</td>
      <td id="T_4f476_row0_col2" class="data row0 col2" >0.9447</td>
      <td id="T_4f476_row0_col3" class="data row0 col3" >0.9480</td>
      <td id="T_4f476_row0_col4" class="data row0 col4" >0.9486</td>
      <td id="T_4f476_row0_col5" class="data row0 col5" >0.9470</td>
      <td id="T_4f476_row0_col6" class="data row0 col6" >0.9670</td>
      <td id="T_4f476_row0_col7" class="data row0 col7" >0.8447</td>
      <td id="T_4f476_row0_col8" class="data row0 col8" >0.8483</td>
      <td id="T_4f476_row0_col9" class="data row0 col9" >1.4540</td>
    </tr>
    <tr>
      <th id="T_4f476_level0_row1" class="row_heading level0 row1" >5</th>
      <td id="T_4f476_row1_col0" class="data row1 col0" >Random Forest Classifier</td>
      <td id="T_4f476_row1_col1" class="data row1 col1" >0.9480</td>
      <td id="T_4f476_row1_col2" class="data row1 col2" >0.9468</td>
      <td id="T_4f476_row1_col3" class="data row1 col3" >0.9480</td>
      <td id="T_4f476_row1_col4" class="data row1 col4" >0.9491</td>
      <td id="T_4f476_row1_col5" class="data row1 col5" >0.9473</td>
      <td id="T_4f476_row1_col6" class="data row1 col6" >0.9668</td>
      <td id="T_4f476_row1_col7" class="data row1 col7" >0.8462</td>
      <td id="T_4f476_row1_col8" class="data row1 col8" >0.8498</td>
      <td id="T_4f476_row1_col9" class="data row1 col9" >0.4560</td>
    </tr>
    <tr>
      <th id="T_4f476_level0_row2" class="row_heading level0 row2" >7</th>
      <td id="T_4f476_row2_col0" class="data row2 col0" >Gradient Boosting Classifier</td>
      <td id="T_4f476_row2_col1" class="data row2 col1" >0.9480</td>
      <td id="T_4f476_row2_col2" class="data row2 col2" >0.9374</td>
      <td id="T_4f476_row2_col3" class="data row2 col3" >0.9480</td>
      <td id="T_4f476_row2_col4" class="data row2 col4" >0.9487</td>
      <td id="T_4f476_row2_col5" class="data row2 col5" >0.9475</td>
      <td id="T_4f476_row2_col6" class="data row2 col6" >0.9667</td>
      <td id="T_4f476_row2_col7" class="data row2 col7" >0.8475</td>
      <td id="T_4f476_row2_col8" class="data row2 col8" >0.8498</td>
      <td id="T_4f476_row2_col9" class="data row2 col9" >3.9700</td>
    </tr>
    <tr>
      <th id="T_4f476_level0_row3" class="row_heading level0 row3" >8</th>
      <td id="T_4f476_row3_col0" class="data row3 col0" >Ada Boost Classifier</td>
      <td id="T_4f476_row3_col1" class="data row3 col1" >0.9367</td>
      <td id="T_4f476_row3_col2" class="data row3 col2" >0.9488</td>
      <td id="T_4f476_row3_col3" class="data row3 col3" >0.9367</td>
      <td id="T_4f476_row3_col4" class="data row3 col4" >0.9379</td>
      <td id="T_4f476_row3_col5" class="data row3 col5" >0.9357</td>
      <td id="T_4f476_row3_col6" class="data row3 col6" >0.9596</td>
      <td id="T_4f476_row3_col7" class="data row3 col7" >0.8125</td>
      <td id="T_4f476_row3_col8" class="data row3 col8" >0.8168</td>
      <td id="T_4f476_row3_col9" class="data row3 col9" >1.0920</td>
    </tr>
    <tr>
      <th id="T_4f476_level0_row4" class="row_heading level0 row4" >6</th>
      <td id="T_4f476_row4_col0" class="data row4 col0" >Extra Trees Classifier</td>
      <td id="T_4f476_row4_col1" class="data row4 col1" >0.9288</td>
      <td id="T_4f476_row4_col2" class="data row4 col2" >0.9537</td>
      <td id="T_4f476_row4_col3" class="data row4 col3" >0.9288</td>
      <td id="T_4f476_row4_col4" class="data row4 col4" >0.9298</td>
      <td id="T_4f476_row4_col5" class="data row4 col5" >0.9262</td>
      <td id="T_4f476_row4_col6" class="data row4 col6" >0.9552</td>
      <td id="T_4f476_row4_col7" class="data row4 col7" >0.7810</td>
      <td id="T_4f476_row4_col8" class="data row4 col8" >0.7892</td>
      <td id="T_4f476_row4_col9" class="data row4 col9" >0.4300</td>
    </tr>
    <tr>
      <th id="T_4f476_level0_row5" class="row_heading level0 row5" >11</th>
      <td id="T_4f476_row5_col0" class="data row5 col0" >LinearSVC</td>
      <td id="T_4f476_row5_col1" class="data row5 col1" >0.9073</td>
      <td id="T_4f476_row5_col2" class="data row5 col2" >0.9490</td>
      <td id="T_4f476_row5_col3" class="data row5 col3" >0.9073</td>
      <td id="T_4f476_row5_col4" class="data row5 col4" >0.9060</td>
      <td id="T_4f476_row5_col5" class="data row5 col5" >0.9033</td>
      <td id="T_4f476_row5_col6" class="data row5 col6" >0.9420</td>
      <td id="T_4f476_row5_col7" class="data row5 col7" >0.7123</td>
      <td id="T_4f476_row5_col8" class="data row5 col8" >0.7207</td>
      <td id="T_4f476_row5_col9" class="data row5 col9" >0.6020</td>
    </tr>
    <tr>
      <th id="T_4f476_level0_row6" class="row_heading level0 row6" >4</th>
      <td id="T_4f476_row6_col0" class="data row6 col0" >Decision Tree Classifier</td>
      <td id="T_4f476_row6_col1" class="data row6 col1" >0.8960</td>
      <td id="T_4f476_row6_col2" class="data row6 col2" >0.8753</td>
      <td id="T_4f476_row6_col3" class="data row6 col3" >0.8960</td>
      <td id="T_4f476_row6_col4" class="data row6 col4" >0.9037</td>
      <td id="T_4f476_row6_col5" class="data row6 col5" >0.8982</td>
      <td id="T_4f476_row6_col6" class="data row6 col6" >0.9316</td>
      <td id="T_4f476_row6_col7" class="data row6 col7" >0.7142</td>
      <td id="T_4f476_row6_col8" class="data row6 col8" >0.7185</td>
      <td id="T_4f476_row6_col9" class="data row6 col9" >0.5080</td>
    </tr>
    <tr>
      <th id="T_4f476_level0_row7" class="row_heading level0 row7" >2</th>
      <td id="T_4f476_row7_col0" class="data row7 col0" >Ridge Classifier</td>
      <td id="T_4f476_row7_col1" class="data row7 col1" >0.8791</td>
      <td id="T_4f476_row7_col2" class="data row7 col2" >0.9395</td>
      <td id="T_4f476_row7_col3" class="data row7 col3" >0.8791</td>
      <td id="T_4f476_row7_col4" class="data row7 col4" >0.8824</td>
      <td id="T_4f476_row7_col5" class="data row7 col5" >0.8660</td>
      <td id="T_4f476_row7_col6" class="data row7 col6" >0.9267</td>
      <td id="T_4f476_row7_col7" class="data row7 col7" >0.5887</td>
      <td id="T_4f476_row7_col8" class="data row7 col8" >0.6239</td>
      <td id="T_4f476_row7_col9" class="data row7 col9" >0.4100</td>
    </tr>
    <tr>
      <th id="T_4f476_level0_row8" class="row_heading level0 row8" >3</th>
      <td id="T_4f476_row8_col0" class="data row8 col0" >Linear Discriminant Analysis</td>
      <td id="T_4f476_row8_col1" class="data row8 col1" >0.8802</td>
      <td id="T_4f476_row8_col2" class="data row8 col2" >0.9245</td>
      <td id="T_4f476_row8_col3" class="data row8 col3" >0.8802</td>
      <td id="T_4f476_row8_col4" class="data row8 col4" >0.8809</td>
      <td id="T_4f476_row8_col5" class="data row8 col5" >0.8750</td>
      <td id="T_4f476_row8_col6" class="data row8 col6" >0.9247</td>
      <td id="T_4f476_row8_col7" class="data row8 col7" >0.6302</td>
      <td id="T_4f476_row8_col8" class="data row8 col8" >0.6435</td>
      <td id="T_4f476_row8_col9" class="data row8 col9" >0.6240</td>
    </tr>
    <tr>
      <th id="T_4f476_level0_row9" class="row_heading level0 row9" >1</th>
      <td id="T_4f476_row9_col0" class="data row9 col0" >Logistic Regression</td>
      <td id="T_4f476_row9_col1" class="data row9 col1" >0.8305</td>
      <td id="T_4f476_row9_col2" class="data row9 col2" >0.9218</td>
      <td id="T_4f476_row9_col3" class="data row9 col3" >0.8305</td>
      <td id="T_4f476_row9_col4" class="data row9 col4" >0.8456</td>
      <td id="T_4f476_row9_col5" class="data row9 col5" >0.7913</td>
      <td id="T_4f476_row9_col6" class="data row9 col6" >0.9009</td>
      <td id="T_4f476_row9_col7" class="data row9 col7" >0.3472</td>
      <td id="T_4f476_row9_col8" class="data row9 col8" >0.4350</td>
      <td id="T_4f476_row9_col9" class="data row9 col9" >0.4680</td>
    </tr>
    <tr>
      <th id="T_4f476_level0_row10" class="row_heading level0 row10" >0</th>
      <td id="T_4f476_row10_col0" class="data row10 col0" >Naive Bayes</td>
      <td id="T_4f476_row10_col1" class="data row10 col1" >0.8113</td>
      <td id="T_4f476_row10_col2" class="data row10 col2" >0.6432</td>
      <td id="T_4f476_row10_col3" class="data row10 col3" >0.8113</td>
      <td id="T_4f476_row10_col4" class="data row10 col4" >0.7930</td>
      <td id="T_4f476_row10_col5" class="data row10 col5" >0.7871</td>
      <td id="T_4f476_row10_col6" class="data row10 col6" >0.8863</td>
      <td id="T_4f476_row10_col7" class="data row10 col7" >0.3439</td>
      <td id="T_4f476_row10_col8" class="data row10 col8" >0.3725</td>
      <td id="T_4f476_row10_col9" class="data row10 col9" >0.4080</td>
    </tr>
    <tr>
      <th id="T_4f476_level0_row11" class="row_heading level0 row11" >10</th>
      <td id="T_4f476_row11_col0" class="data row11 col0" >K Neighbors Classifier</td>
      <td id="T_4f476_row11_col1" class="data row11 col1" >0.7989</td>
      <td id="T_4f476_row11_col2" class="data row11 col2" >0.7613</td>
      <td id="T_4f476_row11_col3" class="data row11 col3" >0.7989</td>
      <td id="T_4f476_row11_col4" class="data row11 col4" >0.8066</td>
      <td id="T_4f476_row11_col5" class="data row11 col5" >0.7490</td>
      <td id="T_4f476_row11_col6" class="data row11 col6" >0.8821</td>
      <td id="T_4f476_row11_col7" class="data row11 col7" >0.2204</td>
      <td id="T_4f476_row11_col8" class="data row11 col8" >0.2921</td>
      <td id="T_4f476_row11_col9" class="data row11 col9" >0.4600</td>
    </tr>
  </tbody>
</table>



On the cleaned law section, all models perform substantially better than on facts. Tree ensembles lead (LightGBM, RF, GBC all at 0.9667–0.9670), followed closely by SVM at 0.9420. LR ranks lower at 0.9009. The law section contains the court's formal legal reasoning, which is more structured and predictive than the facts narrative — this explains the overall performance lift.

SVM outperforms LR here because LinearSVC's margin-maximisation is especially effective on clean, sparse legal language with consistent terminology. NB (0.8863) again plays its expected baseline role, ranking second-lowest — confirming that despite the richer signal in the law section, NB's independence assumption still limits it relative to margin-based and ensemble methods.



**Step 3: Fine Tune model — V3 Law Clean**

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model ID</th>
      <th>Sklearn Class</th>
      <th>Tuned F1 Macro</th>
      <th>Kappa</th>
      <th>MCC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nb</td>
      <td>Naive Bayes (ComplementNB)</td>
      <td>0.9051</td>
      <td>0.4749</td>
      <td>0.5012</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lr</td>
      <td>Logistic Regression</td>
      <td>0.9406</td>
      <td>0.7292</td>
      <td>0.7314</td>
    </tr>
    <tr>
      <th>2</th>
      <td>svm</td>
      <td>Linear SVM (custom)</td>
      <td>0.9420</td>
      <td>0.7123</td>
      <td>0.7207</td>
    </tr>
  </tbody>
</table>
On V3 Law Clean, NB 0.9051 (tuned from ComplementNB base), LR: 0.9406 (+0.0397), SVM: 0.9420 (unchanged). LR gains the most here — tuning finds `C=8.957`, higher than default, allowing less regularisation and better fitting the structured legal text. SVM reaches its ceiling at default `C=1.0`. NB remains the lowest performer across both text arms, confirming its baseline role as expected.




    ==============================================================
      HYPERPARAMETERS — V3 — Law Clean
    ==============================================================
    
      ── Naive Bayes  (ComplementNB)
         alpha                                    1.0
         class_prior                              None
         fit_prior                                True
         force_alpha                              True
         norm                                     False
    
      ── Logistic Regression  (LogisticRegression)
         C                                        8.956999999999999
         class_weight                             balanced
         dual                                     False
         fit_intercept                            True
         intercept_scaling                        1
         l1_ratio                                 None
         max_iter                                 1000
         multi_class                              auto
         n_jobs                                   None
         penalty                                  l2
         random_state                             42
         solver                                   lbfgs
         tol                                      0.0001
         verbose                                  0
         warm_start                               False
    
      ── Linear SVM (custom)  (CalibratedClassifierCV → LinearSVC)
         C                                        1.0
         class_weight                             balanced
         dual                                     warn
         fit_intercept                            True
         intercept_scaling                        1
         loss                                     squared_hinge
         max_iter                                 2000
         multi_class                              ovr
         penalty                                  l2
         random_state                             42
         tol                                      0.0001
         verbose                                  0



## Metadata + Text (Law Clean)

**Step 1: Preprocessing**

V4 combines 10,000 TF-IDF columns from the cleaned law text with four categorical and three numeric metadata columns. All TF-IDF column names are prefixed with `tfidf_` to prevent name collisions with metadata columns. Categorical metadata columns are declared explicitly in `setup()` to trigger one-hot encoding, while numeric columns undergo standard scaling. SMOTE (`fix_imbalance=True`) is applied at the data level, though this is a practical compromise — interpolating across 10,000+ sparse dimensions is less theoretically clean than for low-dimensional features.

    ✓ No duplicate columns
    V4 shape: (1265, 10008)
    TF-IDF cols: 10000 | Meta cols: 7



### Pycaret

**Step 2: Compare Models**

<style type="text/css">
#T_1460b_row0_col6 {
  background-color: #90ee90;
}
#T_1460b_row8_col6, #T_1460b_row9_col6, #T_1460b_row10_col6 {
  background-color: #ffe066;
  font-weight: bold;
}
</style>
<table id="T_1460b">
  <caption>V4 — Law Clean + Metadata — All Models sorted by F1 Macro | 🟡 NB · LR · SVM (interpretable)  🟢 Best overall</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_1460b_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_1460b_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_1460b_level0_col2" class="col_heading level0 col2" >AUC</th>
      <th id="T_1460b_level0_col3" class="col_heading level0 col3" >Recall</th>
      <th id="T_1460b_level0_col4" class="col_heading level0 col4" >Prec.</th>
      <th id="T_1460b_level0_col5" class="col_heading level0 col5" >F1</th>
      <th id="T_1460b_level0_col6" class="col_heading level0 col6" >F1 Macro</th>
      <th id="T_1460b_level0_col7" class="col_heading level0 col7" >Kappa</th>
      <th id="T_1460b_level0_col8" class="col_heading level0 col8" >MCC</th>
      <th id="T_1460b_level0_col9" class="col_heading level0 col9" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1460b_level0_row0" class="row_heading level0 row0" >7</th>
      <td id="T_1460b_row0_col0" class="data row0 col0" >Gradient Boosting Classifier</td>
      <td id="T_1460b_row0_col1" class="data row0 col1" >0.9435</td>
      <td id="T_1460b_row0_col2" class="data row0 col2" >0.9496</td>
      <td id="T_1460b_row0_col3" class="data row0 col3" >0.9435</td>
      <td id="T_1460b_row0_col4" class="data row0 col4" >0.9438</td>
      <td id="T_1460b_row0_col5" class="data row0 col5" >0.9428</td>
      <td id="T_1460b_row0_col6" class="data row0 col6" >0.9639</td>
      <td id="T_1460b_row0_col7" class="data row0 col7" >0.8336</td>
      <td id="T_1460b_row0_col8" class="data row0 col8" >0.8358</td>
      <td id="T_1460b_row0_col9" class="data row0 col9" >21.9980</td>
    </tr>
    <tr>
      <th id="T_1460b_level0_row1" class="row_heading level0 row1" >9</th>
      <td id="T_1460b_row1_col0" class="data row1 col0" >Light Gradient Boosting Machine</td>
      <td id="T_1460b_row1_col1" class="data row1 col1" >0.9412</td>
      <td id="T_1460b_row1_col2" class="data row1 col2" >0.9594</td>
      <td id="T_1460b_row1_col3" class="data row1 col3" >0.9412</td>
      <td id="T_1460b_row1_col4" class="data row1 col4" >0.9418</td>
      <td id="T_1460b_row1_col5" class="data row1 col5" >0.9405</td>
      <td id="T_1460b_row1_col6" class="data row1 col6" >0.9624</td>
      <td id="T_1460b_row1_col7" class="data row1 col7" >0.8269</td>
      <td id="T_1460b_row1_col8" class="data row1 col8" >0.8296</td>
      <td id="T_1460b_row1_col9" class="data row1 col9" >16.4700</td>
    </tr>
    <tr>
      <th id="T_1460b_level0_row2" class="row_heading level0 row2" >8</th>
      <td id="T_1460b_row2_col0" class="data row2 col0" >Ada Boost Classifier</td>
      <td id="T_1460b_row2_col1" class="data row2 col1" >0.9333</td>
      <td id="T_1460b_row2_col2" class="data row2 col2" >0.9446</td>
      <td id="T_1460b_row2_col3" class="data row2 col3" >0.9333</td>
      <td id="T_1460b_row2_col4" class="data row2 col4" >0.9339</td>
      <td id="T_1460b_row2_col5" class="data row2 col5" >0.9331</td>
      <td id="T_1460b_row2_col6" class="data row2 col6" >0.9572</td>
      <td id="T_1460b_row2_col7" class="data row2 col7" >0.8066</td>
      <td id="T_1460b_row2_col8" class="data row2 col8" >0.8081</td>
      <td id="T_1460b_row2_col9" class="data row2 col9" >15.3900</td>
    </tr>
    <tr>
      <th id="T_1460b_level0_row3" class="row_heading level0 row3" >5</th>
      <td id="T_1460b_row3_col0" class="data row3 col0" >Random Forest Classifier</td>
      <td id="T_1460b_row3_col1" class="data row3 col1" >0.9333</td>
      <td id="T_1460b_row3_col2" class="data row3 col2" >0.9536</td>
      <td id="T_1460b_row3_col3" class="data row3 col3" >0.9333</td>
      <td id="T_1460b_row3_col4" class="data row3 col4" >0.9345</td>
      <td id="T_1460b_row3_col5" class="data row3 col5" >0.9334</td>
      <td id="T_1460b_row3_col6" class="data row3 col6" >0.9570</td>
      <td id="T_1460b_row3_col7" class="data row3 col7" >0.8085</td>
      <td id="T_1460b_row3_col8" class="data row3 col8" >0.8098</td>
      <td id="T_1460b_row3_col9" class="data row3 col9" >10.2500</td>
    </tr>
    <tr>
      <th id="T_1460b_level0_row4" class="row_heading level0 row4" >6</th>
      <td id="T_1460b_row4_col0" class="data row4 col0" >Extra Trees Classifier</td>
      <td id="T_1460b_row4_col1" class="data row4 col1" >0.9322</td>
      <td id="T_1460b_row4_col2" class="data row4 col2" >0.9529</td>
      <td id="T_1460b_row4_col3" class="data row4 col3" >0.9322</td>
      <td id="T_1460b_row4_col4" class="data row4 col4" >0.9328</td>
      <td id="T_1460b_row4_col5" class="data row4 col5" >0.9314</td>
      <td id="T_1460b_row4_col6" class="data row4 col6" >0.9567</td>
      <td id="T_1460b_row4_col7" class="data row4 col7" >0.8002</td>
      <td id="T_1460b_row4_col8" class="data row4 col8" >0.8033</td>
      <td id="T_1460b_row4_col9" class="data row4 col9" >10.0080</td>
    </tr>
    <tr>
      <th id="T_1460b_level0_row5" class="row_heading level0 row5" >4</th>
      <td id="T_1460b_row5_col0" class="data row5 col0" >Decision Tree Classifier</td>
      <td id="T_1460b_row5_col1" class="data row5 col1" >0.9006</td>
      <td id="T_1460b_row5_col2" class="data row5 col2" >0.8601</td>
      <td id="T_1460b_row5_col3" class="data row5 col3" >0.9006</td>
      <td id="T_1460b_row5_col4" class="data row5 col4" >0.9017</td>
      <td id="T_1460b_row5_col5" class="data row5 col5" >0.9010</td>
      <td id="T_1460b_row5_col6" class="data row5 col6" >0.9358</td>
      <td id="T_1460b_row5_col7" class="data row5 col7" >0.7156</td>
      <td id="T_1460b_row5_col8" class="data row5 col8" >0.7160</td>
      <td id="T_1460b_row5_col9" class="data row5 col9" >10.6680</td>
    </tr>
    <tr>
      <th id="T_1460b_level0_row6" class="row_heading level0 row6" >2</th>
      <td id="T_1460b_row6_col0" class="data row6 col0" >Ridge Classifier</td>
      <td id="T_1460b_row6_col1" class="data row6 col1" >0.8983</td>
      <td id="T_1460b_row6_col2" class="data row6 col2" >0.9285</td>
      <td id="T_1460b_row6_col3" class="data row6 col3" >0.8983</td>
      <td id="T_1460b_row6_col4" class="data row6 col4" >0.8969</td>
      <td id="T_1460b_row6_col5" class="data row6 col5" >0.8968</td>
      <td id="T_1460b_row6_col6" class="data row6 col6" >0.9351</td>
      <td id="T_1460b_row6_col7" class="data row6 col7" >0.6993</td>
      <td id="T_1460b_row6_col8" class="data row6 col8" >0.7012</td>
      <td id="T_1460b_row6_col9" class="data row6 col9" >17.4620</td>
    </tr>
    <tr>
      <th id="T_1460b_level0_row7" class="row_heading level0 row7" >3</th>
      <td id="T_1460b_row7_col0" class="data row7 col0" >Linear Discriminant Analysis</td>
      <td id="T_1460b_row7_col1" class="data row7 col1" >0.8644</td>
      <td id="T_1460b_row7_col2" class="data row7 col2" >0.9083</td>
      <td id="T_1460b_row7_col3" class="data row7 col3" >0.8644</td>
      <td id="T_1460b_row7_col4" class="data row7 col4" >0.8686</td>
      <td id="T_1460b_row7_col5" class="data row7 col5" >0.8621</td>
      <td id="T_1460b_row7_col6" class="data row7 col6" >0.9129</td>
      <td id="T_1460b_row7_col7" class="data row7 col7" >0.6021</td>
      <td id="T_1460b_row7_col8" class="data row7 col8" >0.6119</td>
      <td id="T_1460b_row7_col9" class="data row7 col9" >13.5420</td>
    </tr>
    <tr>
      <th id="T_1460b_level0_row8" class="row_heading level0 row8" >1</th>
      <td id="T_1460b_row8_col0" class="data row8 col0" >Logistic Regression</td>
      <td id="T_1460b_row8_col1" class="data row8 col1" >0.8520</td>
      <td id="T_1460b_row8_col2" class="data row8 col2" >0.9088</td>
      <td id="T_1460b_row8_col3" class="data row8 col3" >0.8520</td>
      <td id="T_1460b_row8_col4" class="data row8 col4" >0.8605</td>
      <td id="T_1460b_row8_col5" class="data row8 col5" >0.8547</td>
      <td id="T_1460b_row8_col6" class="data row8 col6" >0.9029</td>
      <td id="T_1460b_row8_col7" class="data row8 col7" >0.5909</td>
      <td id="T_1460b_row8_col8" class="data row8 col8" >0.5944</td>
      <td id="T_1460b_row8_col9" class="data row8 col9" >11.1120</td>
    </tr>
    <tr>
      <th id="T_1460b_level0_row9" class="row_heading level0 row9" >0</th>
      <td id="T_1460b_row9_col0" class="data row9 col0" >Naive Bayes</td>
      <td id="T_1460b_row9_col1" class="data row9 col1" >0.8147</td>
      <td id="T_1460b_row9_col2" class="data row9 col2" >0.6406</td>
      <td id="T_1460b_row9_col3" class="data row9 col3" >0.8147</td>
      <td id="T_1460b_row9_col4" class="data row9 col4" >0.7991</td>
      <td id="T_1460b_row9_col5" class="data row9 col5" >0.7875</td>
      <td id="T_1460b_row9_col6" class="data row9 col6" >0.8890</td>
      <td id="T_1460b_row9_col7" class="data row9 col7" >0.3426</td>
      <td id="T_1460b_row9_col8" class="data row9 col8" >0.3796</td>
      <td id="T_1460b_row9_col9" class="data row9 col9" >6.7620</td>
    </tr>
    <tr>
      <th id="T_1460b_level0_row10" class="row_heading level0 row10" >11</th>
      <td id="T_1460b_row10_col0" class="data row10 col0" >LinearSVC</td>
      <td id="T_1460b_row10_col1" class="data row10 col1" >0.6023</td>
      <td id="T_1460b_row10_col2" class="data row10 col2" >0.7444</td>
      <td id="T_1460b_row10_col3" class="data row10 col3" >0.6023</td>
      <td id="T_1460b_row10_col4" class="data row10 col4" >0.7112</td>
      <td id="T_1460b_row10_col5" class="data row10 col5" >0.6341</td>
      <td id="T_1460b_row10_col6" class="data row10 col6" >0.7048</td>
      <td id="T_1460b_row10_col7" class="data row10 col7" >0.1352</td>
      <td id="T_1460b_row10_col8" class="data row10 col8" >0.1511</td>
      <td id="T_1460b_row10_col9" class="data row10 col9" >20.6380</td>
    </tr>
    <tr>
      <th id="T_1460b_level0_row11" class="row_heading level0 row11" >10</th>
      <td id="T_1460b_row11_col0" class="data row11 col0" >K Neighbors Classifier</td>
      <td id="T_1460b_row11_col1" class="data row11 col1" >0.4158</td>
      <td id="T_1460b_row11_col2" class="data row11 col2" >0.6279</td>
      <td id="T_1460b_row11_col3" class="data row11 col3" >0.4158</td>
      <td id="T_1460b_row11_col4" class="data row11 col4" >0.7630</td>
      <td id="T_1460b_row11_col5" class="data row11 col5" >0.4201</td>
      <td id="T_1460b_row11_col6" class="data row11 col6" >0.4242</td>
      <td id="T_1460b_row11_col7" class="data row11 col7" >0.0919</td>
      <td id="T_1460b_row11_col8" class="data row11 col8" >0.1704</td>
      <td id="T_1460b_row11_col9" class="data row11 col9" >11.4160</td>
    </tr>
  </tbody>
</table>
Tree ensembles continue to dominate, led by Gradient Boosting (F1 Macro 0.9639). Among interpretable models, LR (0.9029) and NB (0.8890) are in a similar range. PyCaret's default LinearSVC scores 0.7048. The custom CalibratedClassifierCV SVM collapses in this mixed feature space, confirming the failure is not wrapper-specific but a property of linear SVM geometry on high-dimensional mixed features. This is the same issue seen in Section 1 Metadata — LinearSVC cannot handle a mixed 10,000+ dimensional feature space of sparse TF-IDF and OHE columns without explicit feature normalisation at the estimator level. Adding metadata to text features does not improve the interpretable models relative to text-only, suggesting metadata adds noise rather than signal in this combined representation. NB performs as expected in the baseline role, trailing LR.



**Step 3: Fine Tune Model**

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model ID</th>
      <th>Sklearn Class</th>
      <th>Tuned F1 Macro</th>
      <th>Kappa</th>
      <th>MCC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nb</td>
      <td>Naive Bayes (GaussianNB)</td>
      <td>0.8961</td>
      <td>0.4839</td>
      <td>0.4896</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lr</td>
      <td>Logistic Regression</td>
      <td>0.9337</td>
      <td>0.6924</td>
      <td>0.6948</td>
    </tr>
    <tr>
      <th>2</th>
      <td>svm</td>
      <td>Linear SVM (custom)</td>
      <td>0.7059</td>
      <td>0.1365</td>
      <td>0.1523</td>
    </tr>
  </tbody>
</table>
LR improves to 0.9337 (tuned), a gain of +0.0308. NB improves to 0.8961 (+0.0071). The SVM collapse persists across all tuned values, confirming this is an architectural incompatibility with this mixed feature space rather than a hyperparameter problem. NB is not the worst model — SVM holds that position. 



**Step 4: Hyperparameter Extraction**


    ==============================================================
      HYPERPARAMETERS — V4 — Law Clean + Meta
    ==============================================================
    
      ── Naive Bayes  (GaussianNB)
         priors                                   None
         var_smoothing                            1e-07
    
      ── Logistic Regression  (LogisticRegression)
         C                                        9.56
         class_weight                             balanced
         dual                                     False
         fit_intercept                            True
         intercept_scaling                        1
         l1_ratio                                 None
         max_iter                                 1000
         multi_class                              auto
         n_jobs                                   None
         penalty                                  l2
         random_state                             42
         solver                                   lbfgs
         tol                                      0.0001
         verbose                                  0
         warm_start                               False
    
      ── Linear SVM (custom)  (CalibratedClassifierCV → LinearSVC)
         C                                        0.01
         class_weight                             balanced
         dual                                     warn
         fit_intercept                            True
         intercept_scaling                        1
         loss                                     squared_hinge
         max_iter                                 2000
         multi_class                              ovr
         penalty                                  l2
         random_state                             42
         tol                                      0.0001
         verbose                                  0

Transfer **LR** `C=9.56` and `class_weight='balanced'` — the higher C allows the model to fit the combined feature space with less shrinkage, and balanced weighting corrects residual class imbalance. **NB** `var_smoothing=1e-07` (near-default, minimal impact). The most impactful parameter overall is LR's C value, which drives the largest single improvement in this section.




# **RQ2 : Topic**

**Step 1: Preprocessing**

The cleaned law text is first vectorised with TF-IDF (`max_features=10,000`, bigrams, `sublinear_tf=True`), then decomposed into 40 topic dimensions using NMF (`n_components=40`, `random_state=42`). Each document is represented as a 40-dimensional vector of non-negative topic weights. SMOTE (`fix_imbalance=True`) is appropriate here — the low-dimensional continuous NMF output is well-suited for SMOTE's interpolation between samples. GaussianNB is used because topic weights are continuous values, not sparse token counts, making the Gaussian distribution assumption more defensible than ComplementNB.

    V5 Topic Features → shape: (1265, 41)
    Topics: 40 | Docs: 1265



**Step 2: Compare Models**

<style type="text/css">
#T_f9738_row0_col6 {
  background-color: #90ee90;
}
#T_f9738_row7_col6, #T_f9738_row8_col6, #T_f9738_row9_col6 {
  background-color: #ffe066;
  font-weight: bold;
}
</style>
<table id="T_f9738">
  <caption>V5 — Topic Features (k=40) — All Models sorted by F1 Macro | 🟡 NB · LR · SVM (interpretable)  🟢 Best overall</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_f9738_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_f9738_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_f9738_level0_col2" class="col_heading level0 col2" >AUC</th>
      <th id="T_f9738_level0_col3" class="col_heading level0 col3" >Recall</th>
      <th id="T_f9738_level0_col4" class="col_heading level0 col4" >Prec.</th>
      <th id="T_f9738_level0_col5" class="col_heading level0 col5" >F1</th>
      <th id="T_f9738_level0_col6" class="col_heading level0 col6" >F1 Macro</th>
      <th id="T_f9738_level0_col7" class="col_heading level0 col7" >Kappa</th>
      <th id="T_f9738_level0_col8" class="col_heading level0 col8" >MCC</th>
      <th id="T_f9738_level0_col9" class="col_heading level0 col9" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f9738_level0_row0" class="row_heading level0 row0" >5</th>
      <td id="T_f9738_row0_col0" class="data row0 col0" >Random Forest Classifier</td>
      <td id="T_f9738_row0_col1" class="data row0 col1" >0.9232</td>
      <td id="T_f9738_row0_col2" class="data row0 col2" >0.9364</td>
      <td id="T_f9738_row0_col3" class="data row0 col3" >0.9232</td>
      <td id="T_f9738_row0_col4" class="data row0 col4" >0.9218</td>
      <td id="T_f9738_row0_col5" class="data row0 col5" >0.9220</td>
      <td id="T_f9738_row0_col6" class="data row0 col6" >0.9511</td>
      <td id="T_f9738_row0_col7" class="data row0 col7" >0.7718</td>
      <td id="T_f9738_row0_col8" class="data row0 col8" >0.7732</td>
      <td id="T_f9738_row0_col9" class="data row0 col9" >0.0560</td>
    </tr>
    <tr>
      <th id="T_f9738_level0_row1" class="row_heading level0 row1" >9</th>
      <td id="T_f9738_row1_col0" class="data row1 col0" >Light Gradient Boosting Machine</td>
      <td id="T_f9738_row1_col1" class="data row1 col1" >0.9119</td>
      <td id="T_f9738_row1_col2" class="data row1 col2" >0.9327</td>
      <td id="T_f9738_row1_col3" class="data row1 col3" >0.9119</td>
      <td id="T_f9738_row1_col4" class="data row1 col4" >0.9118</td>
      <td id="T_f9738_row1_col5" class="data row1 col5" >0.9108</td>
      <td id="T_f9738_row1_col6" class="data row1 col6" >0.9437</td>
      <td id="T_f9738_row1_col7" class="data row1 col7" >0.7404</td>
      <td id="T_f9738_row1_col8" class="data row1 col8" >0.7431</td>
      <td id="T_f9738_row1_col9" class="data row1 col9" >0.3120</td>
    </tr>
    <tr>
      <th id="T_f9738_level0_row2" class="row_heading level0 row2" >7</th>
      <td id="T_f9738_row2_col0" class="data row2 col0" >Gradient Boosting Classifier</td>
      <td id="T_f9738_row2_col1" class="data row2 col1" >0.9085</td>
      <td id="T_f9738_row2_col2" class="data row2 col2" >0.9329</td>
      <td id="T_f9738_row2_col3" class="data row2 col3" >0.9085</td>
      <td id="T_f9738_row2_col4" class="data row2 col4" >0.9097</td>
      <td id="T_f9738_row2_col5" class="data row2 col5" >0.9086</td>
      <td id="T_f9738_row2_col6" class="data row2 col6" >0.9409</td>
      <td id="T_f9738_row2_col7" class="data row2 col7" >0.7373</td>
      <td id="T_f9738_row2_col8" class="data row2 col8" >0.7384</td>
      <td id="T_f9738_row2_col9" class="data row2 col9" >0.1220</td>
    </tr>
    <tr>
      <th id="T_f9738_level0_row3" class="row_heading level0 row3" >6</th>
      <td id="T_f9738_row3_col0" class="data row3 col0" >Extra Trees Classifier</td>
      <td id="T_f9738_row3_col1" class="data row3 col1" >0.9051</td>
      <td id="T_f9738_row3_col2" class="data row3 col2" >0.9485</td>
      <td id="T_f9738_row3_col3" class="data row3 col3" >0.9051</td>
      <td id="T_f9738_row3_col4" class="data row3 col4" >0.9030</td>
      <td id="T_f9738_row3_col5" class="data row3 col5" >0.9033</td>
      <td id="T_f9738_row3_col6" class="data row3 col6" >0.9397</td>
      <td id="T_f9738_row3_col7" class="data row3 col7" >0.7165</td>
      <td id="T_f9738_row3_col8" class="data row3 col8" >0.7184</td>
      <td id="T_f9738_row3_col9" class="data row3 col9" >0.0340</td>
    </tr>
    <tr>
      <th id="T_f9738_level0_row4" class="row_heading level0 row4" >8</th>
      <td id="T_f9738_row4_col0" class="data row4 col0" >Ada Boost Classifier</td>
      <td id="T_f9738_row4_col1" class="data row4 col1" >0.8768</td>
      <td id="T_f9738_row4_col2" class="data row4 col2" >0.8902</td>
      <td id="T_f9738_row4_col3" class="data row4 col3" >0.8768</td>
      <td id="T_f9738_row4_col4" class="data row4 col4" >0.8807</td>
      <td id="T_f9738_row4_col5" class="data row4 col5" >0.8781</td>
      <td id="T_f9738_row4_col6" class="data row4 col6" >0.9199</td>
      <td id="T_f9738_row4_col7" class="data row4 col7" >0.6528</td>
      <td id="T_f9738_row4_col8" class="data row4 col8" >0.6545</td>
      <td id="T_f9738_row4_col9" class="data row4 col9" >0.0380</td>
    </tr>
    <tr>
      <th id="T_f9738_level0_row5" class="row_heading level0 row5" >4</th>
      <td id="T_f9738_row5_col0" class="data row5 col0" >Decision Tree Classifier</td>
      <td id="T_f9738_row5_col1" class="data row5 col1" >0.8667</td>
      <td id="T_f9738_row5_col2" class="data row5 col2" >0.8213</td>
      <td id="T_f9738_row5_col3" class="data row5 col3" >0.8667</td>
      <td id="T_f9738_row5_col4" class="data row5 col4" >0.8684</td>
      <td id="T_f9738_row5_col5" class="data row5 col5" >0.8672</td>
      <td id="T_f9738_row5_col6" class="data row5 col6" >0.9138</td>
      <td id="T_f9738_row5_col7" class="data row5 col7" >0.6192</td>
      <td id="T_f9738_row5_col8" class="data row5 col8" >0.6199</td>
      <td id="T_f9738_row5_col9" class="data row5 col9" >0.0160</td>
    </tr>
    <tr>
      <th id="T_f9738_level0_row6" class="row_heading level0 row6" >3</th>
      <td id="T_f9738_row6_col0" class="data row6 col0" >Linear Discriminant Analysis</td>
      <td id="T_f9738_row6_col1" class="data row6 col1" >0.8203</td>
      <td id="T_f9738_row6_col2" class="data row6 col2" >0.9078</td>
      <td id="T_f9738_row6_col3" class="data row6 col3" >0.8203</td>
      <td id="T_f9738_row6_col4" class="data row6 col4" >0.8663</td>
      <td id="T_f9738_row6_col5" class="data row6 col5" >0.8317</td>
      <td id="T_f9738_row6_col6" class="data row6 col6" >0.8745</td>
      <td id="T_f9738_row6_col7" class="data row6 col7" >0.5655</td>
      <td id="T_f9738_row6_col8" class="data row6 col8" >0.5897</td>
      <td id="T_f9738_row6_col9" class="data row6 col9" >0.0120</td>
    </tr>
    <tr>
      <th id="T_f9738_level0_row7" class="row_heading level0 row7" >11</th>
      <td id="T_f9738_row7_col0" class="data row7 col0" >LinearSVC</td>
      <td id="T_f9738_row7_col1" class="data row7 col1" >0.8000</td>
      <td id="T_f9738_row7_col2" class="data row7 col2" >0.8813</td>
      <td id="T_f9738_row7_col3" class="data row7 col3" >0.8000</td>
      <td id="T_f9738_row7_col4" class="data row7 col4" >0.8366</td>
      <td id="T_f9738_row7_col5" class="data row7 col5" >0.8108</td>
      <td id="T_f9738_row7_col6" class="data row7 col6" >0.8628</td>
      <td id="T_f9738_row7_col7" class="data row7 col7" >0.4991</td>
      <td id="T_f9738_row7_col8" class="data row7 col8" >0.5140</td>
      <td id="T_f9738_row7_col9" class="data row7 col9" >0.0180</td>
    </tr>
    <tr>
      <th id="T_f9738_level0_row8" class="row_heading level0 row8" >0</th>
      <td id="T_f9738_row8_col0" class="data row8 col0" >Naive Bayes</td>
      <td id="T_f9738_row8_col1" class="data row8 col1" >0.7842</td>
      <td id="T_f9738_row8_col2" class="data row8 col2" >0.8796</td>
      <td id="T_f9738_row8_col3" class="data row8 col3" >0.7842</td>
      <td id="T_f9738_row8_col4" class="data row8 col4" >0.8530</td>
      <td id="T_f9738_row8_col5" class="data row8 col5" >0.7999</td>
      <td id="T_f9738_row8_col6" class="data row8 col6" >0.8453</td>
      <td id="T_f9738_row8_col7" class="data row8 col7" >0.5025</td>
      <td id="T_f9738_row8_col8" class="data row8 col8" >0.5382</td>
      <td id="T_f9738_row8_col9" class="data row8 col9" >0.0200</td>
    </tr>
    <tr>
      <th id="T_f9738_level0_row9" class="row_heading level0 row9" >1</th>
      <td id="T_f9738_row9_col0" class="data row9 col0" >Logistic Regression</td>
      <td id="T_f9738_row9_col1" class="data row9 col1" >0.7684</td>
      <td id="T_f9738_row9_col2" class="data row9 col2" >0.8604</td>
      <td id="T_f9738_row9_col3" class="data row9 col3" >0.7684</td>
      <td id="T_f9738_row9_col4" class="data row9 col4" >0.8281</td>
      <td id="T_f9738_row9_col5" class="data row9 col5" >0.7839</td>
      <td id="T_f9738_row9_col6" class="data row9 col6" >0.8361</td>
      <td id="T_f9738_row9_col7" class="data row9 col7" >0.4517</td>
      <td id="T_f9738_row9_col8" class="data row9 col8" >0.4773</td>
      <td id="T_f9738_row9_col9" class="data row9 col9" >0.0160</td>
    </tr>
    <tr>
      <th id="T_f9738_level0_row10" class="row_heading level0 row10" >2</th>
      <td id="T_f9738_row10_col0" class="data row10 col0" >Ridge Classifier</td>
      <td id="T_f9738_row10_col1" class="data row10 col1" >0.7605</td>
      <td id="T_f9738_row10_col2" class="data row10 col2" >0.8723</td>
      <td id="T_f9738_row10_col3" class="data row10 col3" >0.7605</td>
      <td id="T_f9738_row10_col4" class="data row10 col4" >0.8302</td>
      <td id="T_f9738_row10_col5" class="data row10 col5" >0.7775</td>
      <td id="T_f9738_row10_col6" class="data row10 col6" >0.8282</td>
      <td id="T_f9738_row10_col7" class="data row10 col7" >0.4462</td>
      <td id="T_f9738_row10_col8" class="data row10 col8" >0.4772</td>
      <td id="T_f9738_row10_col9" class="data row10 col9" >0.0100</td>
    </tr>
    <tr>
      <th id="T_f9738_level0_row11" class="row_heading level0 row11" >10</th>
      <td id="T_f9738_row11_col0" class="data row11 col0" >K Neighbors Classifier</td>
      <td id="T_f9738_row11_col1" class="data row11 col1" >0.7469</td>
      <td id="T_f9738_row11_col2" class="data row11 col2" >0.8344</td>
      <td id="T_f9738_row11_col3" class="data row11 col3" >0.7469</td>
      <td id="T_f9738_row11_col4" class="data row11 col4" >0.8292</td>
      <td id="T_f9738_row11_col5" class="data row11 col5" >0.7663</td>
      <td id="T_f9738_row11_col6" class="data row11 col6" >0.8163</td>
      <td id="T_f9738_row11_col7" class="data row11 col7" >0.4292</td>
      <td id="T_f9738_row11_col8" class="data row11 col8" >0.4670</td>
      <td id="T_f9738_row11_col9" class="data row11 col9" >0.0100</td>
    </tr>
  </tbody>
</table>
****

Topic features produce the clearest ranking among all sections. SVM leads the interpretable models at F1 Macro 0.8626, surpassing LR (0.8361) by a meaningful margin. Tree ensembles remain the strongest overall (Random Forest 0.9511). SVM's strong performance on 40 dense continuous dimensions confirms that a max-margin linear classifier works extremely well when features are semantically compressed and non-sparse. 

NB (0.8453) ranks in the middle of the three interpretable models, fulfilling its expected baseline role compared to SVM. 



**Step 3: Fine Tune Model**

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model ID</th>
      <th>Sklearn Class</th>
      <th>Tuned F1 Macro</th>
      <th>Kappa</th>
      <th>MCC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nb</td>
      <td>Naive Bayes (GaussianNB)</td>
      <td>0.8460</td>
      <td>0.5058</td>
      <td>0.5423</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lr</td>
      <td>Logistic Regression</td>
      <td>0.8590</td>
      <td>0.5169</td>
      <td>0.5420</td>
    </tr>
    <tr>
      <th>2</th>
      <td>svm</td>
      <td>Linear SVM (custom)</td>
      <td>0.9247</td>
      <td>0.7087</td>
      <td>0.7189</td>
    </tr>
  </tbody>
</table>
On V5 Topics, LR gains modestly (0.8590) with C=9.56. SVM gain to 0.9247 because the 40-dimensional continuous NMF space, while tractable at default settings, is disrupted by the calibration wrapper at lower C values. 



**Step 4: Hyperparameter Extraction**


    ==============================================================
      HYPERPARAMETERS — V5 — Topic Features
    ==============================================================
    
      ── Naive Bayes  (GaussianNB)
         priors                                   None
         var_smoothing                            1e-05
    
      ── Logistic Regression  (LogisticRegression)
         C                                        9.56
         class_weight                             balanced
         dual                                     False
         fit_intercept                            True
         intercept_scaling                        1
         l1_ratio                                 None
         max_iter                                 1000
         multi_class                              auto
         n_jobs                                   None
         penalty                                  l2
         random_state                             42
         solver                                   lbfgs
         tol                                      0.0001
         verbose                                  0
         warm_start                               False
    
      ── Linear SVM (custom)  (CalibratedClassifierCV → LinearSVC)
         C                                        100
         class_weight                             balanced
         dual                                     warn
         fit_intercept                            True
         intercept_scaling                        1
         loss                                     squared_hinge
         max_iter                                 2000
         multi_class                              ovr
         penalty                                  l2
         random_state                             42
         tol                                      0.0001
         verbose                                  0

Since tuning produced no improvement, base model parameters are transferred directly. **SVM** — `C=100` (tuned to high value, allowing tight margin on dense topic features),  **LR** — `C=9.56`,  (SMOTE handles imbalance at data level). **NB** — `var_smoothing=1e-05`. The most meaningful parameter is SVM's `C=100`, which allows the model to maximise the margin on the well-separated topic feature space.



# **CONCLUSION** : 
**Optimization Effort**

The five feature arms vary substantially in how much they respond to hyperparameter tuning, which reflects the underlying geometry of each feature space rather than any limitation in the tuning procedure itself.

**Topic features** are the most stable arm — The 40-dimensional NMF topic space is geometrically well-conditioned: features are dense, continuous, and semantically compressed, so default hyperparameters already sit near the performance ceiling. Tuning effort here yields zero return.

**Law Clean text** requires moderate, targeted tuning. LR benefits most from a higher regularisation relaxation by fitting the structured legal vocabulary with less shrinkage. SVM's gain is modest because C=1.0 was already near-optimal for clean sparse text. ComplementNB benefits from a small alpha (0.01), which preserves fine-grained term frequency distinctions in formal legal language. Total tuning effort is low: a small grid over C and alpha is sufficient.

**Facts text** is the highest-reward arm for tuning effort, primarily because the base configuration used in the final notebook the wrong NB variant. Replacing GaussianNB with ComplementNB (alpha=0.01) drives NB's substantial gain from a single architectural correction. LR and SVM also benefits from a higher C, both reflecting that raw factual narratives require less regularisation because the discriminative signal is sparse but real.

**Metadata** shows meaningful gains for LR and NB but SVM is structurally resistant. LR improves through C with class_weight=balanced, correcting both regularisation and residual class imbalance not fully resolved by SMOTE. NB improves similarly because SMOTE balancing makes class distributions more symmetric, partially closing NB's usual disadvantage. SVM's small gain is not a tuning failure — LinearSVC's geometry collapses on the mixed OHE + numeric feature space regardless of C value.

**Metadata + Law Clean** is the most unstable arm. Combining 10,000 TF-IDF dimensions with 7 metadata columns creates a feature space that is simultaneously sparse and mixed-scale. LR recovers well with, but SVM collapses completely across all six grid values (C ∈ {0.001, 0.01, 0.1, 1, 10, 100}), confirming the collapse is architectural. Adding metadata to text features does not improve the interpretable models relative to text-only law section, indicating that metadata introduces noise rather than signal in this concatenated representation.



**Parting Thoughts**
The gains from PyCaret tuning sharpen the numbers but do not rewrite the story. The rank ordering of feature arms, the interpretability trade-offs, and the core findings on Article 8 classification should remain identical whether default or tuned parameters are used — confirming that **the final notebook's experimental design was analytically sound from the start.**

PyCaret's leaderboard does surface stronger models such as LightBGM and Gradient Boosting. But raw performance is not the only criterion here. **Legal research demands that predictions be traceable, auditable, and defensible** — conditions that tree ensembles cannot satisfy intrinsically. Logistic Regression and LinearSVM expose their decision weights directly, making every coefficient can offer claim or global explanation about what drives a classification outcome. Two out of three (incl NB as baseline) that we choose to apply in the final notebook are not the strongest models available, but they are proven as the strongest models that remain fit for legal research purposes.

