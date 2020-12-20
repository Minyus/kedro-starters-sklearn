# kedro-starters-sklearn

This repository contains Kedro starters that demonstrate how to use Kedro (0.17.0 or later) and Scikit-learn to train a Logistic Regression model developed based on [Kedro's official starters](https://github.com/quantumblacklabs/kedro-starters). 

<p align="center">
<img src="_doc_images/kedro_viz.png">
Pipeline visualized by Kedro-viz
</p>

## Iris dataset

[Iris dataset](https://www.kaggle.com/uciml/iris) is included and used in default.
- Modification: for each species, setosa is encoded to 0, versicolor is encoded to 1, and virginica samples were removed.
- Split: for each species, first 25 samples were included in train.csv, and last 25 samples were included in test.csv.

## How to use

1. Install dependencies.

    ```bash
    pip install 'kedro>=0.17.0' pandas scikit-learn 
    ```

2. Download a Kedro starter project in one of the directories (e.g. `sklearn-iris`).

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

## Debugging using Visual Studio Code (VSCode)

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

## Option to use Kaggle Titanic dataset

1. Download [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic/data)
2. Replace `train.csv` and `test.csv` in `/path/to/project/directory/data/01_raw` directory
3. Modify `/path/to/project/directory/base/parameters.yml` to set parameters appropriate for the dataset (commented out in default)
