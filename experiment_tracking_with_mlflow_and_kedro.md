---
marp: true
headingDivider: 2
paginate: true
theme: standard
backgroundImage: url('https://images.pexels.com/photos/19670/pexels-photo.jpg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940')
---


# Experiment Tracking & Model Management with MLflow and Kedro

<!--
_backgroundImage: url('https://images.pexels.com/photos/19670/pexels-photo.jpg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940')
-->

<p align=center>
Yusuke Minami
</p>


## Agenda

1. Experiment Tracking & Model Management
2. How MLflow can resolve the pain points
3. How Kedro can resolve the pain points 


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
            - model evaluation metrics e.g. accuracy, F1 score
            - execution time


## Model Management

- Model Management by storing artifacts:
    - models (e.g. pickle, PyTorch pt/pth, TensorFlow pb)
    - visualization of model behaviors (e.g. html, png, pdf)
      - e.g. confusion matrix
    - sample predictions (e.g. csv)
    - features & labels used for training (e.g. csv)


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
- Supports various backend databases with SQLAlchemy
  - MySQL, PostgreSQL, SQLite
- Supports various backend storages
  - AWS S3, GCS (Google Cloud Storage), HDFS (Hadoop Distributed File System), etc.
- Accessible to the backend database & storage without using MLflow server
- Intuitive API & web UI
- Package-specific auto-logging option (experimental)
  - TensorFlow/Keras, PyTorch Lightning, Scikit-learn, LightGBM, XGBoost, etc.
- Active development & community


## MLflow Architecture

![height:550px width:800px](https://raw.githubusercontent.com/Minyus/Tools_for_ML_Lifecycle_Management/main/mlflow/mlflow_architecture.png)


## How MLflow works

![](https://raw.githubusercontent.com/Minyus/Tools_for_ML_Lifecycle_Management/main/mlflow/mlflow_experiment_tracking.png)


## UI for Experiment Tracking

- view/search/filter/visualize params/metrics
- download experiment table CSV file

![bg 100% right:50%](_doc_images/mlflow_ui_metrics.png)


## UI for Model Management

- view artifact files 
  - text, YAML, JSON
  - image
  - html
  - pdf
  - GeoJSON
- download artifacts
  - models
  - visualization
  - prediction samples

![bg 100% right:50%](_doc_images/mlflow_ui_gantt.png)


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

Kedro: Python pipeline package which separates data interfaces and processing.

![height:530px](_doc_images/kedro_concept.png)


## Kedro project directory tree

```
- conf
    - base
        - catalog.yml <-- define "DataSets"
        - logging.yml
        - parameters.yml <-- define "DataSet" values
- src
    - <package>
        - catalogs
            - catalog.py <-- define "DataSets"
        - mlflow
            - mlflow_config.py
        - pipelines
            - <pipeline>
                - pipeline.py <-- assign "DataSets" 
                - <nodes>.py <-- define any processing
- main.py
```

![bg 100% right:35%](https://raw.githubusercontent.com/Minyus/kedro-starters-sklearn/master/_doc_images/kedro_viz.png)

## Kedro Catalog (catalog.py)

- Centralized list of "DataSets" (data interfaces)
    - how/where to read/write data
    - {local file, remote storage, database} with/without MLflow 

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


## Kedro Pipeline (pipeline.py)

- For each input & output of Python functions, assign an unique Kedro "DataSet" (interface) name
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
# Columns used as features ("params:features" DataSet)
features: 
  - sepal_length

# Column used as the target ("params:target" DataSet)
target: species
```

![bg 100% right:35%](https://raw.githubusercontent.com/Minyus/kedro-starters-sklearn/master/_doc_images/kedro_viz.png)


## How Kedro DataSets are logged to MLflow 

for DataSet_value in (inputs&outputs_of_Python_functions):
- if DataSet_name not in catalog:
  - if DataSet_value in {float, int}: log as an MLflow metric (numeric)
  - if DataSet_value in {str, list, tuple, dict, set}: log as an MLflow param (string)
  - else (e.g. numpy arrays): skip
- if DataSet_name in catalog (e.g. `model: MLflowDataSet(dataset="pkl")`):
  - if `dataset` == "m": log as an MLflow metric (numeric)
  - if `dataset` == "p": log as an MLflow param (string)
  - if `dataset` in {"pkl", "txt", "yaml", "yml", "json", "csv", "xls", "parquet", "png", "jpeg", "jpg"}: log as an MLflow artifact
  

To upload any local files (e.g. zip, pt/pth, pb, h5, html, pdf, etc.) to MLflow, specify the paths in MLflowArtifactsLoggerHook as in the next slide. 


## MLflow Config (mlflow_config.py)

```python
import pipelinex

mlflow_hooks = (
    pipelinex.MLflowBasicLoggerHook(
        uri="sqlite:///mlruns/sqlite.db",
        experiment_name="experiment_001",
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
    ),
    pipelinex.MLflowTimeLoggerHook(),  # Log duration time to run each node (task)
)
```

## Processing code (no Kedro/MLflow)

```python
from typing import List
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def init_model():
    return LogisticRegression(
        C=1.23456,
        max_iter=987,
        random_state=42,
    )


def train_model(model, df: pd.DataFrame, cols_features: List[str], col_target: str):
    model.fit(df[cols_features], df[col_target])
    return model


def run_inference(model, df: pd.DataFrame, cols_features: List[str]):
    df["pred_proba"] = model.predict_proba(df[cols_features])[:, 1]
    return df


def evaluate_model(model, df: pd.DataFrame, cols_features: List[str], col_target: str):
    y_pred = model.predict(df[cols_features])
    score = float(f1_score(df[col_target], y_pred))
    return score
```

![bg 100% right:35%](https://raw.githubusercontent.com/Minyus/kedro-starters-sklearn/master/_doc_images/kedro_viz.png)


## Available data interfaces

- Kedro DataSet interfaces
    - 25 Official DataSets in [kedro.extras.datasets](https://kedro.readthedocs.io/en/stable/kedro.extras.datasets.html#data-sets)
        - Pickle, CSV, Parquet, Feather, SQL, text, YAML, JSON, GeoJSON, etc.
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
        - read/write {local file, remote storage, database} with/without MLflow 
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
