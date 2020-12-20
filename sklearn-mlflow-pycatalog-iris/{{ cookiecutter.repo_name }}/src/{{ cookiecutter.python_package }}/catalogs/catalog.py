""" DataSet objects can be defined here in addition to YAML (conf/base/catalog.yml) """

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
