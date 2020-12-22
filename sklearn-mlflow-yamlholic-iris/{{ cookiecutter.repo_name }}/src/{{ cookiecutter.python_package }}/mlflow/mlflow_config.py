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
    pipelinex.AddTransformersHook(
        transformers=[
            pipelinex.MLflowIOTimeLoggerTransformer(
                enable_mlflow=True
            )  # Log duration time to load and save each dataset
        ],
    ),  # Add transformers
)
