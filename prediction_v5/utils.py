import json
import re
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich
import rich.progress
import seaborn as sns
import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import GRU, LSTM, Dense, Dropout, Flatten, Input, Masking, SimpleRNN
from keras.metrics import AUC, Precision, Recall
from keras.optimizers.legacy import (  # the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs
    Adam,
)
from keras.regularizers import l2
from matplotlib.ticker import MultipleLocator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUM_CHAPTERS = 12
RANDOM_SEED = 42


@dataclass
class Fold:
    train: pd.DataFrame
    test: pd.DataFrame


class Dataset:
    def __init__(self, name: str):
        """Load dataset"""
        root = "../data/processed"

        # Load original data
        self.data = pd.read_csv(f"{root}/{name}.csv")
        self.feature_cols = self.data.columns.drop(
            ["label", "student_id", "final_grade"]
        )

    def generate_k_fold(self, k=5, random_state=0):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

        # split
        folds = []
        for train_index, test_index in skf.split(self.data, self.data["label"]):
            train = self.data.iloc[train_index]
            test = self.data.iloc[test_index]

            folds.append(Fold(train=train, test=test))

        return folds


class BaseModel:
    name = "Original"

    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.model = base_prediction_pipeline(feature_cols)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_scores(self, X):
        return self.model.predict_proba(X)[:, 1]

    def finetune(self, X_train, y_train, X_val, y_val):
        GRID = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
        }

        best_roc_auc = 0
        best_model = None
        best_params = None

        for param_values in product(*GRID.values()):
            params = dict(zip(GRID.keys(), param_values))
            model = base_prediction_pipeline(self.feature_cols, **params)

            model.fit(X_train, y_train)

            y_pred_scores = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_pred_scores)

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model = model
                best_params = params

        self.model = best_model
        self.best_params = best_params


def base_prediction_pipeline(
    feature_cols: list[str], n_estimators=100, max_depth=None
) -> Pipeline:
    """
    Create a base pipeline that does data processing and prediction with random forest
    Later we can add more steps or add update parameters to the existing steps.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("num_imputer", SimpleImputer(strategy="mean")),
            ("num_scaler", StandardScaler()),
        ],
    )

    column_transformer = ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, feature_cols)],
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", column_transformer),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                ),
            ),
        ]
    )
    return pipeline


def preprocessing_pipeline(feature_cols: list[str]) -> Pipeline:
    """
    Create a base pipeline that does data processing and prediction with random forest
    Later we can add more steps or add update parameters to the existing steps.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("num_imputer", SimpleImputer(strategy="mean")),
            ("num_scaler", StandardScaler()),
        ],
    )

    column_transformer = ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, feature_cols)],
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocessor", column_transformer)])
    return pipeline


def compute_chapter_metrics(model, X, y):
    features = X.columns

    chapter_metrics = []
    for chapter in range(1, 13):
        features_to_mask = [
            f for f in features if int(re.findall(r"\d+", f)[0]) > chapter
        ]
        X_masked = X.copy()
        X_masked[features_to_mask] = np.nan

        y_pred_binary = model.predict(X_masked)
        y_pred_score = model.predict_scores(X_masked)

        chapter_metrics.append(
            {
                "Chapter": chapter,
                "Precision": precision_score(y, y_pred_binary, zero_division=0),
                "Recall": recall_score(y, y_pred_binary),
                "F1": f1_score(y, y_pred_binary),
                "Accuracy": accuracy_score(y, y_pred_binary),
                "ROC_AUC": roc_auc_score(y, y_pred_score),
                "PR_AUC": average_precision_score(y, y_pred_score),
            }
        )

    return chapter_metrics


def create_sequence_model(
    feature_cols: list[str],
    rnn_cls: str = "rnn",
    rnn_units: int = 32,
    dropout_rate: float = 0.5,
    dense_units: int = 8,
):
    rnn_cls_by_name = {"rnn": SimpleRNN, "lstm": LSTM, "gru": GRU}
    rnn_cls = rnn_cls_by_name[rnn_cls]

    n_features = len(feature_cols) // 12

    model = Sequential()
    model.add(Input(shape=(12, n_features)))
    model.add(Masking(mask_value=np.nan))
    model.add(
        rnn_cls(
            units=rnn_units,
            dropout=dropout_rate,
            return_sequences=False,
            kernel_regularizer=l2(0.01),
        )
    )
    model.add(Dense(units=dense_units, activation="relu", kernel_regularizer=l2(0.01)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(), Recall(), AUC(name="roc_auc", curve="ROC")],
    )

    return model


def plot_metrics(metrics: pd.DataFrame, metrics_name: str):
    sns.set_theme(style="darkgrid")
    sns.set_context("talk")
    sns.set_palette("tab10")

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 8))

    g = sns.lineplot(
        data=metrics,
        x="Chapter",
        y=metrics_name,
        hue="Model" if "Model" in metrics.columns else None,
        linewidth=2,
        errorbar=None,
    )
    g.set_xticks(range(1, 13))
    g.set_xticklabels(range(1, 13))

    # Add a horizontal line at 0.7 for ROC_AUC
    # if metrics_name == "ROC_AUC":
    #     plt.axhline(y=0.7, color="r", linestyle="--")


def plot_metrics_subplots(
    metrics: pd.DataFrame, exclude_models: Optional[list[str]] = None, limit_y=True
):
    metrics = metrics.copy()

    sns.set_theme(style="darkgrid")
    sns.set_context("talk")
    sns.set_palette("tab10")

    metrics_names = ["ROC_AUC", "PR_AUC", "Accuracy", "Precision", "Recall", "F1"]

    metrics["Model"] = metrics["Model"].apply(
        lambda x: {
            "Original": "RF Original",
            "Augmented": "RF Augmented",
            "Sequence": "Sequence Original",
        }.get(x)
        or x
    )

    # Set up the subplots with 3 rows and 2 columns
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

    # Keep track of the lines and labels for the legend
    handles = []
    labels = []

    if exclude_models is not None:
        metrics = metrics[~metrics["Model"].isin(exclude_models)]

    for idx, metrics_name in enumerate(metrics_names):
        ax = axes[idx]

        sns.lineplot(
            data=metrics,
            x="Chapter",
            y=metrics_name,
            hue="Model" if "Model" in metrics.columns else None,
            linewidth=2,
            errorbar=None,
            ax=ax,  # Use the specific subplot axis
        )
        ax.set_title(metrics_name)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(range(1, 13))

        # set y limit to 0-1
        if limit_y:
            ax.set_ylim(-0.05, 1.05)
            ax.yaxis.set_major_locator(
                MultipleLocator(0.1)
            )  # Set y ticks at 0.1 intervals

        # Collect the legend handles and labels from the first plot (or any plot)
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()

        # Remove the legend from individual subplots
        ax.get_legend().remove()

        # Add a horizontal line at 0.7 for ROC_AUC
        # if metrics_name == "ROC_AUC":
        #     ax.axhline(y=0.7, color="r", linestyle="--")

    # Create a single legend outside the subplots
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.0),
    )

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for space for the legend


def evaluate_model(model_cls, data, trials=5):
    feature_cols = data.feature_cols
    results = []

    # The whole evaluation process is repeated 'trials' times
    for trial in range(trials):
        print("Trial:", trial)
        all_chapter_metrics: list[list[dict[str, float]]] = [[] for _ in range(12)]

        folds = data.generate_k_fold(5, random_state=trial)
        for fold in folds:
            model = model_cls(feature_cols)
            model.fit(fold.train[feature_cols], fold.train["label"])

            chapter_metrics = compute_chapter_metrics(
                model, fold.test[feature_cols], fold.test["label"]
            )
            for i, chapter_metric in enumerate(chapter_metrics):
                all_chapter_metrics[i].append(chapter_metric)

        # Compute the average and standard deviation of the each metric
        metric_names = all_chapter_metrics[0][0].keys() - {"Chapter"}
        chapter_average_metrics = []

        for chapter_metrics in all_chapter_metrics:
            average_metrics = {
                metric_name: np.mean([m[metric_name] for m in chapter_metrics])
                for metric_name in metric_names
            }
            chapter_average_metrics.append(average_metrics)

        # Average metrics table
        table = pd.DataFrame(chapter_average_metrics)
        table["Model"] = model_cls.name
        table["Chapter"] = table.index + 1
        table["Trial"] = trial + 1

        results.append(table)

    return pd.concat(results)


def evaluate_transfer(model_cls, train_data, test_data):
    feature_cols = train_data.feature_cols
    all_chapter_metrics: list[list[dict[str, float]]] = [[] for _ in range(12)]

    model = model_cls(feature_cols)
    model.fit(train_data.data, train_data.data["label"])

    chapter_metrics = compute_chapter_metrics(
        model, test_data.data[feature_cols], test_data.data["label"]
    )
    for i, chapter_metric in enumerate(chapter_metrics):
        all_chapter_metrics[i].append(chapter_metric)

    # Compute the average and standard deviation of the each metric
    metric_names = all_chapter_metrics[0][0].keys() - {"Chapter"}
    chapter_average_metrics = []
    chapter_std_metrics = []

    for chapter_metrics in all_chapter_metrics:
        average_metrics = {
            metric_name: np.mean([m[metric_name] for m in chapter_metrics])
            for metric_name in metric_names
        }
        std_metrics = {
            metric_name: np.std([m[metric_name] for m in chapter_metrics])
            for metric_name in metric_names
        }
        chapter_average_metrics.append(average_metrics)
        chapter_std_metrics.append(std_metrics)

    # Average metrics table
    table = pd.DataFrame(chapter_average_metrics)
    table["Model"] = model_cls.name
    table["Chapter"] = table.index + 1

    return table


def evaluate_model_finetune(model_cls, data, trials=5):
    feature_cols = data.feature_cols
    results = []

    # The whole evaluation process is repeated 'trials' times
    progress = rich.progress.Progress()
    with progress:
        task = progress.add_task(
            description="[bold yellow]processing:[/bold yellow] trial 1", total=trials
        )

        for trial in range(trials):

            try:
                all_chapter_metrics: list[list[dict[str, float]]] = [
                    [] for _ in range(12)
                ]

                folds = data.generate_k_fold(5, random_state=trial)
                for fold in folds:
                    model = model_cls(feature_cols)
                    model.finetune(
                        fold.train[feature_cols],
                        fold.train["label"],
                        fold.test[feature_cols],
                        fold.test["label"],
                    )

                    chapter_metrics = compute_chapter_metrics(
                        model, fold.test[feature_cols], fold.test["label"]
                    )
                    for i, chapter_metric in enumerate(chapter_metrics):
                        all_chapter_metrics[i].append(chapter_metric)

                # Compute the average and standard deviation of the each metric
                metric_names = all_chapter_metrics[0][0].keys() - {"Chapter"}
                chapter_average_metrics = []

                for chapter_metrics in all_chapter_metrics:
                    average_metrics = {
                        metric_name: np.mean([m[metric_name] for m in chapter_metrics])
                        for metric_name in metric_names
                    }
                    chapter_average_metrics.append(average_metrics)

                # Average metrics table
                table = pd.DataFrame(chapter_average_metrics)
                table["Model"] = model_cls.name
                table["Chapter"] = table.index + 1
                table["Trial"] = trial + 1

                results.append(table)
            except Exception as e:
                rich.print(f"Error in trial {trial}, skipped trial: {e}")

            progress.update(
                task,
                description=f"[bold yellow]Processing:[bold yellow] trial {trial + 1}",
                advance=1,
            )
            progress.refresh()

        progress.update(
            task,
            completed=trials,
            description=f"[bold green] Finished evaluating model: [/bold green]{model_cls.name}",
        )
        progress.refresh()

    return pd.concat(results)


def evaluate_transfer_finetune(
    model_cls, train_data, test_data, return_best_model=False
):
    feature_cols = train_data.feature_cols
    all_chapter_metrics: list[list[dict[str, float]]] = [[] for _ in range(12)]

    model = model_cls(feature_cols)
    model.finetune(
        train_data.data,
        train_data.data["label"],
        test_data.data,
        test_data.data["label"],
    )

    chapter_metrics = compute_chapter_metrics(
        model, test_data.data[feature_cols], test_data.data["label"]
    )
    for i, chapter_metric in enumerate(chapter_metrics):
        all_chapter_metrics[i].append(chapter_metric)

    # Compute the average and standard deviation of the each metric
    metric_names = all_chapter_metrics[0][0].keys() - {"Chapter"}
    chapter_average_metrics = []
    chapter_std_metrics = []

    for chapter_metrics in all_chapter_metrics:
        average_metrics = {
            metric_name: np.mean([m[metric_name] for m in chapter_metrics])
            for metric_name in metric_names
        }
        std_metrics = {
            metric_name: np.std([m[metric_name] for m in chapter_metrics])
            for metric_name in metric_names
        }
        chapter_average_metrics.append(average_metrics)
        chapter_std_metrics.append(std_metrics)

    # Average metrics table
    table = pd.DataFrame(chapter_average_metrics)
    table["Model"] = model_cls.name
    table["Chapter"] = table.index + 1
    table["Best Prams"] = json.dumps(model.best_params)

    if return_best_model:
        return table, model

    return table
