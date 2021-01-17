""" DataSet objects can be defined here in addition to YAML (conf/base/catalog.yml) """

from kedro.extras.datasets.pandas import CSVDataSet
from kedro.extras.datasets.pickle import PickleDataSet
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
