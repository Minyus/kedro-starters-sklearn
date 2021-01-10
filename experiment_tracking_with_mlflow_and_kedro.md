---
marp: true
headingDivider: 2
paginate: true
---

# Experiment Tracking with MLflow and Kedro

## Experiment Tracking

Store model info to enable to query and visualize ML experiments

- metadata (in database)
    - parameters (string)
        - model hyperparameters
        - Git commit hash/message
    - metrics (numeric)
        - model metrics e.g. accuracy 
        - execution time
- artifacts (in storage)
    - models
    - visualization of model behaviors
    - samples with which the model did not work well

## Why MLflow?

- All features (except access control) are free for multiple users
- Support Auto-logging
- Support various backend databases with SQLAlchemy
- Accessible to the backend database & storage even if MLflow server is down

Reference:
https://github.com/Minyus/Tools_for_ML_Lifecycle_Management

## MLflow Architecture

![](https://raw.githubusercontent.com/Minyus/Tools_for_ML_Lifecycle_Management/main/mlflow/mlflow_architecture.drawio.svg)

## How MLflow works

![](https://raw.githubusercontent.com/Minyus/Tools_for_ML_Lifecycle_Management/main/mlflow/mlflow_experiment_tracking.drawio.svg)

## MLflow Web UI

![bg right:70%](https://raw.githubusercontent.com/Minyus/pipelinex_sklearn/master/img/mlflow_ui.png)

## MLflow example Python code

```python
import time

enable_mlflow = True

if enable_mlflow:
    import os
    import mlflow

    experiment_name = "experiment_001"

    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    mlflow.start_run(experiment_id=experiment_id, run_name=None)

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

- Not modular
    - task processing code and MLflow logging code are mixed
- No API for logging execution time
    - Need to add 2 separate lines (before and after)
    - Need to specify unique names for each subtask

## Kedro can resolve the mess 

```
- Pipeline (DAG; Python code or YAML)
    - Node1
        - DataSet(s) for reading
        - Python function
        - DataSet(s) for writing
    - Node2
- Catalog (YAML or Python code)
    - DataSet1
        - filepath
        - arguments
    - DataSet2
- Hooks (Inject non-task code between nodes)
    - MLflow logging 
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

## nodes.py example: no Kedro/MLflow code included

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

## catalog.py example

```python
from kedro.extras.datasets.pandas import CSVDataSet
from kedro.extras.datasets.pickle import PickleDataSet

catalog_dict = {
    "train_df": CSVDataSet(
        filepath="data/01_raw/train.csv",
    ),
    "test_df": CSVDataSet(
        filepath="data/01_raw/test.csv",
    ),
    "model": PickleDataSet(filepath="data/06_models/model.pkl"),
    "pred_df": CSVDataSet(
        filepath="data/07_model_output/pred.csv",
        save_args={"float_format": "%.16e"},
    ),
}
```

## parameters.yml example

```yaml
# Columns used as features 
features: 
  - sepal_length

# Column used as the target
target: species
```

## pipeline.py example

```python
from kedro.pipeline import Pipeline, node

from .nodes import init_model, train_model, evaluate_model, run_inference


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(func=init_model, inputs=None, outputs="init_model"),
            node(
                func=train_model,
                inputs=[
                    "init_model",
                    "train_df",
                    "params:features",
                    "params:target",
                ],
                outputs="model",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "model",
                    "train_df",
                    "params:features",
                    "params:target",
                ],
                outputs="score",
            ),
            node(
                func=run_inference,
                inputs=["model", "test_df", "params:features"],
                outputs="pred_df",
            ),
        ]
    )
```

## mlflow_config.py example

```python
import pipelinex

mlflow_hooks = (
    pipelinex.MLflowBasicLoggerHook(
        enable_mlflow=True,  # Enable configuring and logging to MLflow
        uri="sqlite:///mlruns/sqlite.db",
        experiment_name="experiment_001",
        artifact_location="./mlruns/experiment_001",
        offset_hours=0,  # Specify the offset hour (e.g. 0 for UTC/GMT +00:00) to log in MLflow
    ),  # Configure and log duration time for the pipeline
    pipelinex.MLflowArtifactsLoggerHook(
        enable_mlflow=True,  # Enable logging to MLflow
        filepaths_before_pipeline_run=[
            "conf/base/parameters.yml"
        ],  # Optionally specify the file paths to log before pipeline is run
        filepaths_after_pipeline_run=[
            "data/06_models/model.pkl"
        ],  # Optionally specify the file paths to log after pipeline is run
    ),  # Log artifacts of specified file paths and dataset names
    pipelinex.MLflowDataSetsLoggerHook(
        enable_mlflow=True,  # Enable logging to MLflow
    ),  # Log output datasets of (list of) float, int, and str classes
    pipelinex.MLflowTimeLoggerHook(
        enable_mlflow=True,  # Enable logging to MLflow
    ),  # Log duration time to run each node (task)
```

## Data Read/Write

- Kedro DataSet interface
    - 25 DataSets in [kedro.extras.datasets](https://kedro.readthedocs.io/en/stable/kedro.extras.datasets.html#data-sets)
        - [kedro.extras.datasets.pandas.CSVDataSet](https://kedro.readthedocs.io/en/stable/kedro.extras.datasets.pandas.CSVDataSet.html#kedro.extras.datasets.pandas.CSVDataSet)
        - [kedro.extras.datasets.pickle.PickleDataSet](https://kedro.readthedocs.io/en/stable/kedro.extras.datasets.pickle.PickleDataSet.html#kedro.extras.datasets.pickle.PickleDataSet)
        - [kedro.extras.datasets.tensorflow.TensorFlowModelDataset](https://kedro.readthedocs.io/en/stable/kedro.extras.datasets.tensorflow.TensorFlowModelDataset.html#kedro.extras.datasets.tensorflow.TensorFlowModelDataset)
    - More DataSets in [pipelinex.extras.datasets](https://github.com/Minyus/pipelinex#additional-kedro-datasets-data-interface-sets)
        - [pipelinex.ImagesLocalDataSet](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/datasets/pillow/images_dataset.py
        )
            - loads/saves multiple numpy arrays (RGB, BGR, or monochrome image) from/to a folder in local storage using `pillow` package
        - [pipelinex.IterableImagesDataSet](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/datasets/torchvision/iterable_images_dataset.py)
            - wrapper of [`torchvision.datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) 
        - [pipelinex.AsyncAPIDataSet](https://github.com/Minyus/pipelinex/blob/master/src/pipelinex/extras/datasets/httpx/async_api_dataset.py)
            - downloads multiple contents (e.g. images) by async HTTP requests
- Include in task processing code: Low modularity, but quicker in short-term

## Installation

```bash
pip install 'kedro>=0.17.0' mlflow pipelinex plotly
```

![bg 100% right:45%](https://raw.githubusercontent.com/Minyus/pipelinex_sklearn/master/img/mlflow_ui.png)

## Difference between Kedro and Airflow

Airflow: Intermediate data need to be saved in storage/database
Kedro: Intermediate data can be in memory or saved in files/storage/database

Can be used together in different level

- Airflow DAG
    - Airflow Operator1 => Kedro Pipeline1
        - Kedro Node1
        - Kedro Node2
    - Airflow Operator2 => Kedro Pipeline2
        - Kedro Node1
        - Kedro Node2

Reference:
https://github.com/Minyus/Python_Packages_for_Pipeline_Workflow

## Pros and cons of Kedro

- Pros:
    - High modularity/reusability
        - data R/W 
        - task processing
        - non-task code
            - MLflow logging
            - measure execution time
    - Parallel run of processing 
        - using `multiprocessing` under the hood
    - Visualization of pipeline DAG
- Cons:
    - It takes some time to learn

![bg 90% right:45%](https://raw.githubusercontent.com/Minyus/kedro-starters-sklearn/master/_doc_images/kedro_viz.png)

## References

MLflow's official document:
https://mlflow.org/docs/latest/index.html

Kedro's official document:
https://kedro.readthedocs.io/en/stable/index.html

Kedro Starters using Scikit-learn and MLflow:
https://github.com/Minyus/kedro-starters-sklearn
