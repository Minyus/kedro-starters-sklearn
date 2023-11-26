# kedro-starters-sklearn 

This repository provides the following starter templates for Kedro 0.18.14.

- `sklearn-iris` trains a Logistic Regression model using Scikit-learn.
- `sklearn-mlflow-iris` adds experiment tracking feature using MLflow.

<p align="center">
<img src="_doc_images/kedro_viz.png">
Pipeline visualized by Kedro-viz
</p>

## `sklearn-iris` template

### Iris dataset

[Iris dataset](https://www.kaggle.com/uciml/iris) is included and used in default.
- Modification: for each species, setosa is encoded to 0, versicolor is encoded to 1, and virginica samples were removed.
- Split: for each species, first 25 samples were included in train.csv, and last 25 samples were included in test.csv.

### How to use

1. Install dependencies.

    ```bash
    pip install 'kedro==0.18.14' pandas scikit-learn 
    ```

2. Generate your Kedro starter project from `sklearn-iris` directory.

    ```bash
    kedro new --starter https://github.com/Minyus/kedro-starters-sklearn.git --directory sklearn-iris
    ```
    As explained by [Kedro's documentaion](https://kedro.readthedocs.io/en/stable/02_get_started/04_new_project.html), enter project_name, repo_name, and python_package. 

    Note: As your Python package name, choose a unique name and avoid a generic name such as "test" or "sklearn" used by another package. You can see the list of importable packages by running `python -c "help('modules')"`.

3. Change the current directory to the generated project directory.

    ```bash
    cd /path/to/project/directory
    ```

4. Run the project.

    ```bash
    kedro run
    ```

### Option to use Kaggle Titanic dataset

1. Download [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic/data)
2. Replace `train.csv` and `test.csv` in `/path/to/project/directory/data/01_raw` directory
3. Modify `/path/to/project/directory/base/parameters.yml` to set parameters appropriate for the dataset (commented out in default)


## `sklearn-mlflow-iris` template

This template integrates MLflow to Kedro using [PipelineX](https://github.com/Minyus/pipelinex). Even without writing MLflow code. You can:
- configure MLflow Tracking
- log inputs and outputs of Python functions set up as Kedro nodes as parameters (e.g. features used to train the model) and metrics (e.g. F1 score).
- log execution time for each Kedro node and DataSet loading/saving as metrics.
- log artifacts (e.g. models, execution time Gantt Chart visualized by Plotly, `parameters.yml` file)

In this template, MLflow logging is configured in Python code at [`src/<python_package>/mlflow/mlflow_config.py`](sklearn-mlflow-iris/%7B%7B%20cookiecutter.repo_name%20%7D%7D/src/%7B%7B%20cookiecutter.python_package%20%7D%7D/hooks.py) 

See [here](https://github.com/Minyus/pipelinex#integration-with-mlflow-by-kedro-hooks-callbacks) for details.

### How to use

1. Install dependencies.

    ```bash
    pip install 'kedro==0.18.14' pandas scikit-learn mlflow 'pipelinex>=0.7.7' plotly
    ```

2. Generate your Kedro starter project from `sklearn-mlflow-iris` directory.

    ```bash
    kedro new --starter https://github.com/Minyus/kedro-starters-sklearn.git --directory sklearn-mlflow-iris
    ```
3. Follow the same steps as `sklearn-iris` template.

### Access MLflow web UI

To access the MLflow web UI, launch the MLflow server.

```bash
mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///mlruns/sqlite.db --default-artifact-root ./mlruns
```

<p align="center">
<img src="_doc_images/mlflow_ui_metrics.png">
Logged metrics shown in MLflow's UI
</p>

<p align="center">
<img src="_doc_images/mlflow_ui_gantt.png">
Gantt chart for execution time, generated using Plotly, shown in MLflow's UI
</p>
