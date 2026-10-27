"""
Machine Learning Code Templates

These templates are written for interview prep and practical projects.
They use scikit-learn style pipelines so preprocessing is kept together
with the model and data leakage is avoided.

Install if needed:
    pip install pandas numpy scikit-learn matplotlib seaborn joblib

Optional:
    pip install xgboost lightgbm catboost imbalanced-learn
"""


# ============================================================
# 1. Basic Data Loading and First Inspection
# ============================================================

def load_and_inspect_csv(path, target_col=None):
    import pandas as pd

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("\nFirst rows:")
    print(df.head())
    print("\nInfo:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False))
    print("\nDuplicates:", df.duplicated().sum())
    print("\nNumerical summary:")
    print(df.describe())

    if target_col is not None:
        print("\nTarget distribution:")
        print(df[target_col].value_counts(dropna=False))

    return df


# ============================================================
# 2. Simple EDA Helpers
# ============================================================

def quick_eda_plots(df, target_col=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

    for col in numeric_cols[:8]:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()

    for col in categorical_cols[:8]:
        plt.figure(figsize=(7, 4))
        df[col].value_counts(dropna=False).head(20).plot(kind="bar")
        plt.title(f"Counts of {col}")
        plt.tight_layout()
        plt.show()

    if target_col is not None and target_col in df.columns:
        plt.figure(figsize=(6, 4))
        df[target_col].value_counts(dropna=False).plot(kind="bar")
        plt.title(f"Target distribution: {target_col}")
        plt.tight_layout()
        plt.show()


def correlation_heatmap(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# ============================================================
# 3. Train/Test Split
# ============================================================

def make_train_test_split(df, target_col, task="classification"):
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = y if task == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# 4. Preprocessing Pipeline
# ============================================================

def build_preprocessor(X):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


# ============================================================
# 5. Classification Pipeline
# ============================================================

def train_classification_pipeline(df, target_col):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score
    )
    from sklearn.pipeline import Pipeline

    X_train, X_test, y_train, y_test = make_train_test_split(
        df,
        target_col,
        task="classification"
    )

    preprocessor = build_preprocessor(X_train)

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    if hasattr(model.named_steps["classifier"], "predict_proba"):
        y_prob = model.predict_proba(X_test)
        if y_prob.shape[1] == 2:
            print("ROC-AUC:", roc_auc_score(y_test, y_prob[:, 1]))

    return model


# ============================================================
# 6. Regression Pipeline
# ============================================================

def train_regression_pipeline(df, target_col):
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    preprocessor = build_preprocessor(X_train)

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=300, random_state=42))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    return model


# ============================================================
# 7. Compare Multiple Classification Models
# ============================================================

def compare_classification_models(df, target_col):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import cross_val_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    X = df.drop(columns=[target_col])
    y = df[target_col]
    preprocessor = build_preprocessor(X)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "svm": SVC(kernel="rbf", class_weight="balanced"),
        "naive_bayes": GaussianNB()
    }

    results = {}

    for name, estimator in models.items():
        pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", estimator)
        ])

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=5,
            scoring="f1_weighted"
        )

        results[name] = scores.mean()
        print(name, "F1 weighted:", scores.mean())

    return results


# ============================================================
# 8. Compare Multiple Regression Models
# ============================================================

def compare_regression_models(df, target_col):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    X = df.drop(columns=[target_col])
    y = df[target_col]
    preprocessor = build_preprocessor(X)

    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.01),
        "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.5),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "svr": SVR(kernel="rbf")
    }

    results = {}

    for name, estimator in models.items():
        pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", estimator)
        ])

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=5,
            scoring="neg_root_mean_squared_error"
        )

        rmse = -scores.mean()
        results[name] = rmse
        print(name, "RMSE:", rmse)

    return results


# ============================================================
# 9. Hyperparameter Tuning
# ============================================================

def tune_random_forest_classifier(df, target_col):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.pipeline import Pipeline

    X = df.drop(columns=[target_col])
    y = df[target_col]
    preprocessor = build_preprocessor(X)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))
    ])

    param_distributions = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 5, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None]
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="f1_weighted",
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X, y)

    print("Best score:", search.best_score_)
    print("Best params:", search.best_params_)

    return search.best_estimator_


# ============================================================
# 10. PCA Template
# ============================================================

def pca_template(df, numeric_cols, n_components=2):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = df[numeric_cols]

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components))
    ])

    X_pca = pipeline.fit_transform(X)
    pca = pipeline.named_steps["pca"]

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total explained variance:", pca.explained_variance_ratio_.sum())

    if n_components == 2:
        plt.figure(figsize=(6, 5))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Projection")
        plt.tight_layout()
        plt.show()

    return pipeline, X_pca


# ============================================================
# 11. K-Means Clustering
# ============================================================

def kmeans_clustering_template(df, numeric_cols, k=3):
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = df[numeric_cols]

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=10))
    ])

    labels = pipeline.fit_predict(X)
    scaled_X = pipeline.named_steps["scaler"].transform(X)

    print("Silhouette score:", silhouette_score(scaled_X, labels))

    df_with_clusters = df.copy()
    df_with_clusters["cluster"] = labels

    return pipeline, df_with_clusters


def elbow_method_template(df, numeric_cols, max_k=10):
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = df[numeric_cols]
    X_scaled = StandardScaler().fit_transform(X)

    inertias = []

    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        inertias.append(model.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, max_k + 1), inertias, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.tight_layout()
    plt.show()

    return inertias


# ============================================================
# 12. Imbalanced Classification
# ============================================================

def imbalanced_classification_with_class_weight(df, target_col):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
    from sklearn.pipeline import Pipeline

    X_train, X_test, y_train, y_test = make_train_test_split(
        df,
        target_col,
        task="classification"
    )

    preprocessor = build_preprocessor(X_train)

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model


def tune_classification_threshold(model, X_valid, y_valid, desired_recall=0.8):
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    y_prob = model.predict_proba(X_valid)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_valid, y_prob)

    candidate_indices = np.where(recall[:-1] >= desired_recall)[0]

    if len(candidate_indices) == 0:
        print("No threshold reaches desired recall.")
        return 0.5

    best_index = candidate_indices[np.argmax(precision[candidate_indices])]
    best_threshold = thresholds[best_index]

    print("Best threshold:", best_threshold)
    print("Precision:", precision[best_index])
    print("Recall:", recall[best_index])

    return best_threshold


# ============================================================
# 13. Outlier Handling Helpers
# ============================================================

def cap_outliers_iqr(df, columns):
    capped = df.copy()

    for col in columns:
        q1 = capped[col].quantile(0.25)
        q3 = capped[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        capped[col] = capped[col].clip(lower=lower, upper=upper)

    return capped


def cap_outliers_percentile(df, columns, lower_q=0.01, upper_q=0.99):
    capped = df.copy()

    for col in columns:
        lower = capped[col].quantile(lower_q)
        upper = capped[col].quantile(upper_q)
        capped[col] = capped[col].clip(lower=lower, upper=upper)

    return capped


# ============================================================
# 14. Date Feature Engineering
# ============================================================

def add_date_features(df, date_col):
    import numpy as np
    import pandas as pd

    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")

    result[f"{date_col}_year"] = result[date_col].dt.year
    result[f"{date_col}_month"] = result[date_col].dt.month
    result[f"{date_col}_day"] = result[date_col].dt.day
    result[f"{date_col}_dayofweek"] = result[date_col].dt.dayofweek
    result[f"{date_col}_is_weekend"] = result[f"{date_col}_dayofweek"].isin([5, 6]).astype(int)

    month = result[f"{date_col}_month"]
    result[f"{date_col}_month_sin"] = np.sin(2 * np.pi * month / 12)
    result[f"{date_col}_month_cos"] = np.cos(2 * np.pi * month / 12)

    return result


# ============================================================
# 15. Model Saving and Loading
# ============================================================

def save_model(model, path="model_pipeline.joblib"):
    import joblib

    joblib.dump(model, path)
    print("Saved model to:", path)


def load_model(path="model_pipeline.joblib"):
    import joblib

    model = joblib.load(path)
    print("Loaded model from:", path)
    return model


# ============================================================
# 16. FastAPI Serving Sketch
# ============================================================

FASTAPI_SERVING_EXAMPLE = r'''
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load("model_pipeline.joblib")

class PredictionInput(BaseModel):
    age: float
    income: float
    city: str

@app.post("/predict")
def predict(input_data: PredictionInput):
    df = pd.DataFrame([input_data.dict()])
    prediction = model.predict(df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(df).max()
        return {"prediction": int(prediction), "probability": float(probability)}

    return {"prediction": float(prediction)}
'''


if __name__ == "__main__":
    print("Machine learning templates loaded.")
    print("Use these functions by importing this file into your notebook/script.")

