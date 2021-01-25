""" DataSet objects can be defined here in addition to YAML ('conf/base/catalog.yml') """

from kedro.extras.datasets.pandas import CSVDataSet
from pipelinex import MLflowDataSet
"""
[MLflowDataSet]
if `dataset` arg is:
- {"pkl", "txt", "yaml", "yml", "json", "csv", "xls", "parquet", "png", "jpeg", "jpg"}: log as an MLflow artifact
- "m": log as an MLflow metric (numeric)
- "p": log as an MLflow param (string)
"""

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
