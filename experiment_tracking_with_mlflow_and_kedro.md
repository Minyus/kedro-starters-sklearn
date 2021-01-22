---
marp: true
headingDivider: 2
paginate: true
---

# Experiment Tracking & Model Management with MLflow and Kedro

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Yusuke Minami


## Agenda

1. Experiment Tracking & Model Maanagement
2. How MLflow resolves pain points
3. How Kedro resolves pain points 


## Experiment Tracking

- Experiment Tracking by storing metadata:
    - "parameters" (string)
        - inputs
            - model config/hyperparameters
            - Git commit hash/message
            - data versions
    - "metrics" (numeric)
        - inputs
            - model config/hyperparameters
        - outputs
            - model evaluation metrics
                - e.g. accuracy, F1 score
            - execution time


## Model Management

- Model Management by storing artifacts:
    - models (pickle, PyTorch pth, TensorFlow SavedModel, etc.)
    - visualization of model behaviors, e.g. confusion matrix (html, png)
    - samples with which the model did not work well (csv)


## Pain points

- Writing to text files
    - Writing to a log files?
        - hard to compare 2+ experiment runs
    - Writing to a CSV file and upload to Google Drive?
        - hard to manage the artifacts 
- Writing to a database and storage
    - We do not want to spend time on 
        - coding access to database & storage
        - developing an UI application

## Tools

```
- MLflow
- DVC
- Pachyderm
- Sacred
- Polyaxon
- Allegro Trains
- VertaAI ModelDB
- Guild AI
- Kubeflow Metadata
- Weights & Biases
- Neptune.ai
- Valohai
- Comet
```

Reference:
https://github.com/Minyus/Tools_for_ML_Lifecycle_Management


## Why MLflow?

- All features (except access control) are free for multiple users
- Support various backend databases (MySQL, PostgreSQL, SQLite) with SQLAlchemy
- Accessible to the backend database & storage without using MLflow server
- Intuitive API & web UI
- Support Auto-logging
- Active development & community


## MLflow Architecture

![](https://raw.githubusercontent.com/Minyus/Tools_for_ML_Lifecycle_Management/main/mlflow/mlflow_architecture.png)


## How MLflow works

![](https://raw.githubusercontent.com/Minyus/Tools_for_ML_Lifecycle_Management/main/mlflow/mlflow_experiment_tracking.png)


## MLflow Web UI

- Experiment Tracking
    - search/filter/visualize params/metrics
    - download experiment table CSV
- Model Management
    - view files: {text, image, html, pdf, geojson}
    - download artifacts (e.g. models)

![bg right:40%](https://raw.githubusercontent.com/Minyus/pipelinex_sklearn/master/img/mlflow_ui.png)


## MLflow Tracking Python API

```python
import mlflow


experiment_id = mlflow.get_experiment_by_name("experiment_name").experiment_id
mlflow.start_run(experiment_id)

mlflow.log_params({"param_name": "foo"})
mlflow.log_metrics({"metric_name": 123})
mlflow.log_artifact("local_path")

mlflow.end_run()
```

## MLflow Tracking Python code

```python
import time

enable_mlflow = True

if enable_mlflow:
    import os
    import mlflow

    experiment_name = "experiment_001"

    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    mlflow.start_run(experiment_id=experiment_id)

    mlflow.log_params(
        {
            "env..CI_COMMIT_SHA": os.environ.get("CI_COMMIT_SHA", ""),
            "env..CI_COMMIT_MESSAGE": os.environ.get("CI_COMMIT_MESSAGE", ""),
        }
    )

data = open(local_input_data_path).read()

time_begin = time.time()

# Run processing here

time = time.time() - time_begin
time_dict = {"__time": time}

open(local_output_data_path, "w").write(data)

if enable_mlflow:
    mlflow.log_metrics(time_dict)
    mlflow.log_artifact(local_output_data_path)
    mlflow.end_run()
```

## Problems of direct use of MLflow Python API

Messy!

- MLflow code would "contaminate" your processing code
    - becomes an obstacle to reuse your code in the live service
    - becomes an obstacle for unit testing
- No API for logging execution time
    - Need to add 2 separate lines (before and after) or use a Python decorator
    - Need to specify unique names for each subtask

## Kedro can resolve the mess

```
- Catalog
    - dataset1
        - data format (e.g. csv, pickle, etc.)
        - data location/path/URI {file/storage/DB/MLflow}
        - args
    - dataset2
- Pipeline (DAG; Python code or YAML)
    - Node1
        - inputs: [datasets1, datasets2, ...]
        - Python function for processing
        - outputs: [datasets3, ...]
    - Node2
- Hooks (Inject non-task code between Nodes)
    - Config for MLflow
```

![bg 100% right:35%](https://raw.githubusercontent.com/Minyus/kedro-starters-sklearn/master/_doc_images/kedro_viz.png)


## Kedro project directory tree

```
- conf
    - base
        - catalog.yml
        - logging.yml
        - parameters.yml
- src
    - <package>
        - catalogs
            - catalog.py
        - mlflow
            - mlflow_config.py
        - pipelines
            - <pipeline>
                - pipeline.py
                - <nodes>.py
- main.py
```

![bg 100% right:35%](https://raw.githubusercontent.com/Minyus/kedro-starters-sklearn/master/_doc_images/kedro_viz.png)

## Kedro Catalog (catalog.py)

- Configure inputs & outputs ("DataSets")
    - {file, database, storage, MLflow}

```python
from kedro.extras.datasets.pandas import CSVDataSet
from pipelinex import MLflowDataSet


catalog_dict = {
    "train_df": CSVDataSet(
        filepath="data/01_raw/train.csv",
    ),
    "test_df": CSVDataSet(
        filepath="data/01_raw/test.csv",
    ),
    "model": MLflowDataSet(dataset="pkl"),
    "pred_df": MLflowDataSet(dataset="csv"),
}
```

![bg 100% right:35%](https://raw.githubusercontent.com/Minyus/kedro-starters-sklearn/master/_doc_images/kedro_viz.png)


## Processing code (no Kedro/MLflow)

```python
from logging import getLogger
from typing import List

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


log = getLogger(__name__)


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
    log.info("F1 score: {:.3f}".format(score))
    return score
```

![bg 100% right:35%](https://raw.githubusercontent.com/Minyus/kedro-starters-sklearn/master/_doc_images/kedro_viz.png)


## Kedro Pipeline (pipeline.py)

- Map Kedro DataSets and Python functions
- Pipeline DAG will be automatically generated based on dependencies

```python
Pipeline(
    [
        node(func=init_model, inputs=None, outputs="init_model"),
        node(
            inputs=["init_model", "train_df", "params:features", "params:target"],
            func=train_model,
            outputs="model",
        ),
        node(
            inputs=["model", "train_df", "params:features", "params:target"],
            func=evaluate_model,
            outputs="score",
        ),
        node(
            inputs=["model", "test_df", "params:features"],
            func=run_inference,
            outputs="pred_df",
        ),
    ]
)
```

![bg 100% right:35%](https://raw.githubusercontent.com/Minyus/kedro-starters-sklearn/master/_doc_images/kedro_viz.png)


## Config (parameters.yml)

- Optionally, values in YAML config file can be used as input datasets. 

```yaml
# Columns used as features (can specify "params:features" as an input dataset)
features: 
  - sepal_length

# Column used as the target (can specify "params:target" as an input dataset)
target: species
```

![bg 100% right:35%](https://raw.githubusercontent.com/Minyus/kedro-starters-sklearn/master/_doc_images/kedro_viz.png)


## MLflow Config (mlflow_config.py)

```python
import pipelinex

mlflow_hooks = (
    pipelinex.MLflowBasicLoggerHook(
        uri="sqlite:///mlruns/sqlite.db",
        experiment_name="experiment_001",
        artifact_location="./mlruns/experiment_001",
        offset_hours=0,  # Specify the offset hour (e.g. 0 for UTC/GMT +00:00) to log in MLflow
    ),  # Configure and log duration time for the pipeline
    pipelinex.MLflowCatalogLoggerHook(
        auto=True,  # If True (default), for each dataset (Python func input/output) not listed in catalog, 
        # log as a metric for {float, int} types, and log as a param for {str, list, tuple, dict, set} types.
    ),  # Enable MLflowDataSet
    pipelinex.MLflowArtifactsLoggerHook(
        filepaths_before_pipeline_run=[
            "conf/base/parameters.yml"
        ],  # Optionally specify the file paths to log before the pipeline runs
        filepaths_after_pipeline_run=[],  # Optionally specify the file paths to log after the pipeline runs
    ),
    pipelinex.MLflowEnvVarsLoggerHook(
        param_env_vars=[
            "HOSTNAME"
        ],  # Environment variables to log to MLflow as parameters
        metric_env_vars=[],  # Environment variables to log to MLflow as metrics
    ),
    pipelinex.MLflowTimeLoggerHook(),  # Log duration time to run each node (task)
)
```

## Available data interfaces

- Kedro DataSet interfaces
    - 25 Official DataSets in [kedro.extras.datasets](https://kedro.readthedocs.io/en/stable/kedro.extras.datasets.html#data-sets)
        - Pickle, CSV, Parquet, Feather, SQL, text, YAML, JSON, GeoJSON, NetworkX, etc.
    - More DataSets in [pipelinex.extras.datasets](https://github.com/Minyus/pipelinex#additional-kedro-datasets-data-interface-sets)
        - [pipelinex.MLflowDataSet](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/datasets/mlflow/mlflow_dataset.py)
        - [pipelinex.ImagesLocalDataSet](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/datasets/pillow/images_dataset.py
        )
            - loads/saves multiple numpy arrays (RGB, BGR, or monochrome image) from/to a folder in local storage using `pillow` package
        - [pipelinex.IterableImagesDataSet](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/datasets/torchvision/iterable_images_dataset.py)
            - wrapper of [`torchvision.datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) 
        - [pipelinex.AsyncAPIDataSet](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/datasets/httpx/async_api_dataset.py)
            - downloads multiple contents (e.g. images) by async HTTP requests
- Include in task processing code: Low modularity, but often quicker in short-term


## Installation

```bash
pip install 'kedro>=0.17.0' mlflow pipelinex plotly
```

![bg 100% right:45%](https://raw.githubusercontent.com/Minyus/pipelinex_sklearn/master/img/mlflow_ui.png)


## Difference between Kedro and Airflow

![](_doc_images/comp_airflow_kedro.drawio.svg)

Can be used together in different level

- Airflow DAG
    - Airflow Operator1 => Kedro Pipeline1
        - Kedro Node1
        - Kedro Node2
    - Airflow Operator2 => Kedro Pipeline2

Reference:
https://github.com/Minyus/Python_Packages_for_Pipeline_Workflow


## Pros and cons of Kedro

- Pros:
    - High modularity/reusability
        - task processing
        - read/write {file/storage/DB} with/without MLflow 
        - non-task code 
            - measure execution time
    - Auto parallel run using `multiprocessing`
        - Easily disable for debugging
    - Visualization of pipeline DAG 
- Cons:
    - It may take some time to learn

![bg 90% right:35%](https://raw.githubusercontent.com/Minyus/kedro-starters-sklearn/master/_doc_images/kedro_viz.png)


## References

MLflow's official document:
https://mlflow.org/docs/latest/index.html

Kedro's official document:
https://kedro.readthedocs.io/en/stable/index.html

Kedro starters (Cookiecutter templates) using Scikit-learn and MLflow:
https://github.com/Minyus/kedro-starters-sklearn


## Summary

- MLflow resolves pain points of Experiment Tracking & Model Management
- but MLflow API would "contaminate" your processing code
- but Kedro resolves the pain points by separating MLflow (and other data access) code from your processing code
- and even support enabling/disabling parallel run
- and can be used with/without Airflow
