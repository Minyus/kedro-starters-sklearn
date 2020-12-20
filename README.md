# kedro-starters-sklearn

This repository provides Kedro starter templates that demonstrate how to use Kedro (0.17.0 or later) and Scikit-learn to train a Logistic Regression model developed based on [Kedro's official starters](https://github.com/quantumblacklabs/kedro-starters). 

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
    pip install 'kedro>=0.17.0' pandas scikit-learn 
    ```

2. Generate your Kedro starter project from `sklearn-iris` directory.

    ```bash
    kedro new --starter https://github.com/Minyus/kedro-starters-sklearn.git --directory sklearn-iris
    ```

3. Change the current directory to the generated project directory.

    ```bash
    cd /path/to/project/directory
    ```

4. Run the project.

    ```bash
    kedro run
    ```

    Alternatively:

    ```bash
    python main.py
    ```

### Debugging using Visual Studio Code (VSCode)

See the [document](https://code.visualstudio.com/docs/editor/debugging#_launch-configurations) and set up `launch.json` as follows.

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "My Project Debug Config",
            "cwd": "/path/to/project/directory",
            "type": "python",
            "program": "main.py",
            "request": "launch",
            "console": "integratedTerminal"
        }
    ]
}
```

### Option to use Kaggle Titanic dataset

1. Download [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic/data)
2. Replace `train.csv` and `test.csv` in `/path/to/project/directory/data/01_raw` directory
3. Modify `/path/to/project/directory/base/parameters.yml` to set parameters appropriate for the dataset (commented out in default)


## `sklearn-pycatalog-iris` Template

This template adds flexibility to define your Kedro DataSet Catalog in Python code (`src/<python_package>/catalogs/catalog.py`) in addition to YAML (`conf/base/catalog.yml`).

You may prefer this template if:
- You want to feed dynamic arguments (e.g. timestamp) to your Kedro DataSet objects in Python code while you may still want to define Kedro DataSet objects with only constant arguments in YAML.
- You want to utilize code auto-completion feature provided by your Python IDE (e.g. Visual Studio Code, PyCharm).
- You are not familiar with YAML.

### How to use

1. Install dependencies.

    ```bash
    pip install 'kedro>=0.17.0' pandas scikit-learn 
    ```

2. Generate your Kedro starter project from `sklearn-pycatalog-iris` directory.

    ```bash
    kedro new --starter https://github.com/Minyus/kedro-starters-sklearn.git --directory sklearn-pycatalog-iris
    ```
3. Follow the same steps as `sklearn-iris` template.
