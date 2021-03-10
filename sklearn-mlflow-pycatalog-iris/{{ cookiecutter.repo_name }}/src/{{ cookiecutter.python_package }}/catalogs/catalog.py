"""
DataSet objects can be defined here in addition to YAML ('conf/base/catalog.yml') 

Regarding MLflowDataSet, please see the API document:
https://pipelinex.readthedocs.io/en/latest/source/00_api_docs/pipelinex.extras.datasets.mlflow.html#module-pipelinex.extras.datasets.mlflow.mlflow_dataset
"""

from kedro.extras.datasets.pandas import CSVDataSet
from pipelinex import MLflowDataSet


catalog_dict = {
    "train_df": CSVDataSet(
        filepath="data/01_raw/train.csv",  # Read a csv file
    ),
    "test_df": CSVDataSet(
        filepath="data/01_raw/test.csv",  # Read a csv file
    ),
    "model": MLflowDataSet(dataset="pkl"),  # Write a pickle file & upload to MLflow
    "pred_df": MLflowDataSet(dataset="csv"),  # Write a csv file & upload to MLflow
    "score": MLflowDataSet(dataset="m"),  # Write an MLflow metric
}
