# Machine Learning Interview Crash Sheet

This is the short, high-yield version. Read it when the interview is close.

---

## 1. The 30-Second ML Story

Machine learning means teaching computers to learn patterns from data. We collect data, clean it, create useful features, split it into train and test sets, train models, evaluate them using the right metric, tune them, then deploy and monitor them. The most important practical risks are data leakage, wrong metrics, overfitting, poor data quality, and mismatch between training and real-world data.

---

## 2. AI vs ML vs DL

AI:

- Broad field of intelligent machines.

ML:

- Systems learn from data.

DL:

- ML using deep neural networks.

Data Science:

- Uses statistics, programming, ML, visualization, and domain knowledge to solve data problems.

---

## 3. Types of ML

Supervised:

- Labeled data.
- Regression and classification.

Unsupervised:

- No labels.
- Clustering, PCA, anomaly detection.

Semi-supervised:

- Few labels plus many unlabeled examples.

Reinforcement learning:

- Agent learns by reward from environment.

Batch learning:

- Train offline on full data.

Online learning:

- Update continuously as new data arrives.

Instance-based:

- Store examples and compare new data, like KNN.

Model-based:

- Learn parameters or model structure, like linear regression.

---

## 4. End-to-End ML Lifecycle

1. Define problem.
2. Define target.
3. Choose metric.
4. Collect data.
5. Explore data.
6. Clean missing values/outliers.
7. Engineer features.
8. Split train/validation/test.
9. Build baseline.
10. Train models.
11. Evaluate.
12. Tune.
13. Error analysis.
14. Save pipeline.
15. Deploy.
16. Monitor and retrain.

Interview sentence:

"A model is only one part of ML. The full lifecycle matters because bad data, leakage, or wrong metrics can make even a strong algorithm useless."

---

## 5. Data Leakage

Data leakage means the model uses information during training that will not be available during real prediction.

Examples:

- Scaling before train-test split.
- Imputing using full dataset.
- Target encoding using test labels.
- Using future information.
- Same customer appears in both train and test.

Fix:

- Split first.
- Fit preprocessing only on training data.
- Use pipelines.
- Respect time order.

---

## 6. EDA Checklist

Always check:

- `df.shape`
- `df.head()`
- `df.info()`
- `df.describe()`
- Missing values.
- Duplicates.
- Target distribution.
- Numerical distributions.
- Categorical value counts.
- Outliers.
- Correlations.
- Relationship with target.

Univariate:

- One column.

Bivariate:

- Two columns.

Multivariate:

- Multiple columns together.

---

## 7. Feature Engineering

Feature engineering means creating better inputs for the model.

Common techniques:

- Scaling.
- Encoding categorical values.
- Missing value imputation.
- Outlier treatment.
- Log/power transform.
- Binning.
- Date/time extraction.
- Feature construction.
- Feature splitting.
- Feature selection.
- Dimensionality reduction.

Best interview line:

"Feature engineering must use only information available at prediction time."

---

## 8. Scaling

Standardization:

- Mean 0, standard deviation 1.
- Good default.

MinMax scaling:

- Range 0 to 1.
- Sensitive to outliers.

Robust scaling:

- Uses median and IQR.
- Good with outliers.

Needs scaling:

- KNN, SVM, logistic regression, regularized linear models, K-Means, PCA, neural networks.

Usually does not need scaling:

- Decision tree, random forest, gradient boosting trees.

---

## 9. Encoding

Label encoding:

- Converts labels to integers.
- Best for target variable.

Ordinal encoding:

- For ordered categories like low, medium, high.

One-hot encoding:

- For unordered categories like city or color.

Target encoding:

- Uses target mean per category.
- Powerful but leakage-prone.

---

## 10. Missing Values

Options:

- Drop rows.
- Mean/median imputation.
- Mode imputation.
- Add "Missing" category.
- Missing indicator.
- KNN imputer.
- Iterative/MICE imputer.

Important:

- Fit imputer only on training data.

---

## 11. Outliers

Outliers may be errors or meaningful rare cases.

Methods:

- Z-score.
- IQR.
- Percentile capping.
- Winsorization.

Do not remove blindly.

Tree models are usually more robust than linear models to outliers.

---

## 12. Regression

Regression predicts continuous values.

Linear regression:

- Fits line/plane to minimize squared error.

Metrics:

- MAE: average absolute error.
- MSE: average squared error.
- RMSE: square root of MSE.
- R2: variance explained.
- Adjusted R2: R2 adjusted for number of features.

Assumptions:

- Linearity.
- Independent errors.
- Constant error variance.
- Normal residuals.
- No strong multicollinearity.

---

## 13. Regularization

Regularization reduces overfitting by penalizing large weights.

Ridge:

- L2 penalty.
- Shrinks coefficients.
- Good with multicollinearity.

Lasso:

- L1 penalty.
- Can set coefficients to zero.
- Performs feature selection.

ElasticNet:

- Combines L1 and L2.

---

## 14. Logistic Regression

Logistic regression is for classification.

It calculates:

- Linear score.
- Sigmoid probability.
- Class based on threshold.

Loss:

- Binary cross entropy / log loss.

For multiclass:

- Softmax regression.

Important:

- Threshold can be changed depending on precision/recall needs.

---

## 15. Classification Metrics

Accuracy:

- Overall correctness.
- Bad for imbalanced data.

Precision:

- Of predicted positives, how many are correct?
- Use when false positives are costly.

Recall:

- Of actual positives, how many did we find?
- Use when false negatives are costly.

F1:

- Balance between precision and recall.

ROC-AUC:

- Ranking quality across thresholds.

PR-AUC:

- Often better for imbalanced data.

Type 1 error:

- False positive.

Type 2 error:

- False negative.

---

## 16. Bias-Variance

High bias:

- Model too simple.
- Underfitting.

High variance:

- Model too sensitive.
- Overfitting.

Fix underfitting:

- More complex model.
- Better features.
- Train longer.
- Less regularization.

Fix overfitting:

- More data.
- Regularization.
- Simpler model.
- Cross-validation.
- Feature selection.

---

## 17. Algorithms in One Line

Linear Regression:

- Predicts continuous value using linear relationship.

Logistic Regression:

- Predicts class probability.

Naive Bayes:

- Probabilistic classifier with independence assumption; great for text baseline.

KNN:

- Predicts using nearest examples; needs scaling.

Decision Tree:

- Splits data using if-else rules; interpretable but overfits.

Random Forest:

- Many trees using bagging; reduces variance.

Gradient Boosting:

- Sequential trees that correct previous errors; strong for tabular data.

SVM:

- Finds maximum-margin boundary; kernels handle nonlinearity.

K-Means:

- Groups data into k clusters using centroids.

PCA:

- Reduces dimensions by preserving maximum variance.

---

## 18. Model Selection Quick Guide

Need simple interpretable regression:

- Linear regression, Ridge, Lasso.

Need simple interpretable classification:

- Logistic regression, shallow decision tree.

Need strong tabular performance:

- Random Forest, XGBoost, LightGBM, CatBoost.

Need text classification baseline:

- Naive Bayes, Logistic Regression, Linear SVM.

Need clustering:

- K-Means, DBSCAN, hierarchical clustering.

Need dimensionality reduction:

- PCA.

Need image/text advanced:

- Deep learning.

---

## 19. Cross-Validation

K-Fold:

- Split into k parts, train k times.

Stratified K-Fold:

- Keeps class proportions.

TimeSeriesSplit:

- Keeps time order.

Use pipeline inside CV to avoid leakage.

---

## 20. Hyperparameter Tuning

Grid Search:

- Tries all combinations.

Random Search:

- Tries random combinations, often more efficient.

Bayesian Optimization:

- Chooses next trial intelligently.

Common tuning parameters:

- Logistic regression: C, penalty.
- Tree: max_depth, min_samples_leaf.
- Random forest: n_estimators, max_depth, max_features.
- Boosting: learning_rate, n_estimators, max_depth.
- SVM: C, gamma, kernel.
- KNN: n_neighbors, distance metric.

---

## 21. Top Interview Q&A

### 1. What is machine learning?

Machine learning is a method where computers learn patterns from data to make predictions or decisions without explicit rule programming.

### 2. What is supervised learning?

Learning from labeled data where both input and correct output are given.

### 3. What is unsupervised learning?

Learning from unlabeled data to discover patterns or groups.

### 4. Regression vs classification?

Regression predicts continuous values. Classification predicts categories.

### 5. What is overfitting?

The model learns training data too specifically and performs poorly on unseen data.

### 6. What is underfitting?

The model is too simple and performs poorly on both train and test data.

### 7. How do you prevent overfitting?

Use more data, regularization, cross-validation, simpler model, pruning, feature selection, and early stopping for iterative models.

### 8. What is data leakage?

When information unavailable at prediction time enters training, giving fake high performance.

### 9. Why use train-test split?

To estimate performance on unseen data.

### 10. Why use validation set?

To tune hyperparameters and compare models before final test evaluation.

### 11. What is cross-validation?

A method of training and validating on multiple splits to get a more reliable performance estimate.

### 12. Why scale features?

To put features on similar ranges, helping distance-based and gradient-based algorithms.

### 13. Difference between standardization and normalization?

Standardization gives mean 0 and standard deviation 1. Normalization usually scales values to 0-1.

### 14. What is one-hot encoding?

It converts categories into binary columns.

### 15. What is PCA?

PCA reduces dimensionality by finding directions of maximum variance.

### 16. What is curse of dimensionality?

In high dimensions, data becomes sparse and distance-based learning becomes harder.

### 17. What is Ridge regression?

Linear regression with L2 penalty to shrink coefficients and reduce overfitting.

### 18. What is Lasso regression?

Linear regression with L1 penalty that can make some coefficients zero.

### 19. What is ElasticNet?

A combination of L1 and L2 regularization.

### 20. Why is logistic regression used for classification?

It maps linear output through sigmoid or softmax to produce class probabilities.

### 21. What is confusion matrix?

A table showing true positives, false positives, true negatives, and false negatives.

### 22. Precision vs recall?

Precision is correctness among predicted positives. Recall is how many actual positives were found.

### 23. When use precision?

When false positives are costly, like wrongly marking normal email as spam.

### 24. When use recall?

When false negatives are costly, like missing disease or fraud.

### 25. What is F1 score?

Harmonic mean of precision and recall.

### 26. What is ROC-AUC?

Metric that measures how well model ranks positives above negatives across thresholds.

### 27. Why accuracy is bad for imbalanced data?

Because predicting only the majority class can give high accuracy but useless model.

### 28. What is Naive Bayes?

A probabilistic classifier using Bayes theorem and assuming features are independent given the class.

### 29. What is KNN?

An instance-based algorithm that predicts using nearest training examples.

### 30. What is Decision Tree?

A model that splits data using feature-based rules.

### 31. Why do decision trees overfit?

They can keep splitting until they memorize training data.

### 32. What is Random Forest?

An ensemble of decision trees trained on random samples and feature subsets.

### 33. Bagging vs boosting?

Bagging trains models independently to reduce variance. Boosting trains sequentially to correct errors and reduce bias.

### 34. What is SVM?

A classifier that finds the maximum-margin decision boundary.

### 35. What is kernel trick?

A method to create nonlinear decision boundaries by computing similarities in higher-dimensional space.

### 36. What is K-Means?

An unsupervised clustering algorithm that assigns points to nearest centroids.

### 37. How choose k in K-Means?

Use elbow method, silhouette score, and domain knowledge.

### 38. What is imbalanced data?

When one class is much more frequent than another.

### 39. How handle imbalanced data?

Use class weights, resampling, SMOTE, threshold tuning, and appropriate metrics.

### 40. What is model deployment?

Making a trained model usable in real systems through API, batch job, app, or service.

---

## 22. Must-Remember Code Pattern

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", num_pipe, numeric_cols),
    ("cat", cat_pipe, categorical_cols)
])

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
```

This pattern is powerful because it prevents leakage and keeps preprocessing with the model.

---

## 23. Final Confidence Script

"Machine learning is an end-to-end workflow. I need to frame the problem, avoid leakage, preprocess correctly, choose the right metric, train a baseline, compare models, tune carefully, and monitor after deployment. Algorithms matter, but clean data and correct validation matter more."

