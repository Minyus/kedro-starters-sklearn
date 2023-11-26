"""
This is a boilerplate pipeline
generated using Kedro {{ cookiecutter.kedro_version }}
"""

import logging
from typing import List

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


logger = logging.getLogger(__name__)


def init_model():
    return LogisticRegression(
        C=1.23456,
        max_iter=987,
        random_state=42,
    )


def train_model(model, df: pd.DataFrame, cols_features: List[str], col_target: str):
    # TODO: Add validation (Hold-out or Cross-Validation)
    # TODO: Add error handling
    model.fit(df[cols_features], df[col_target])
    return model


def run_inference(model, df: pd.DataFrame, cols_features: List[str]):
    df["pred_proba"] = model.predict_proba(df[cols_features])[:, 1]
    return df


def evaluate_model(model, df: pd.DataFrame, cols_features: List[str], col_target: str):
    y_pred = model.predict(df[cols_features])
    score = float(f1_score(df[col_target], y_pred))
    logger.info("F1 score: {:.3f}".format(score))
    return score
