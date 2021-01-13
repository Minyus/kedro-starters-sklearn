import pipelinex

mlflow_hooks = (
    pipelinex.MLflowBasicLoggerHook(
        enable_mlflow=True,  # Enable configuring and logging to MLflow
        uri="sqlite:///mlruns/sqlite.db",
        experiment_name="experiment_001",
        artifact_location="./mlruns/experiment_001",
        offset_hours=0,  # Specify the offset hour (e.g. 0 for UTC/GMT +00:00) to log in MLflow
    ),  # Configure and log duration time for the pipeline
    pipelinex.MLflowCatalogLoggerHook(
        enable_mlflow=True,  # Enable logging to MLflow
        mlflow_catalog={
            "params:features": "p",
            "params:target": "p",
            "model": "pkl",
            "score": "m",
            "pred_df": "csv",
        },
    ),
    pipelinex.MLflowArtifactsLoggerHook(
        enable_mlflow=True,  # Enable logging to MLflow
        filepaths_before_pipeline_run=[
            "conf/base/parameters.yml"
        ],  # Optionally specify the file paths to log before the pipeline runs
        filepaths_after_pipeline_run=[],  # Optionally specify the file paths to log after the pipeline runs
    ),
    pipelinex.MLflowEnvVarsLoggerHook(
        enable_mlflow=True,  # Enable logging to MLflow
        param_env_vars=[
            "HOSTNAME"
        ],  # Environment variables to log to MLflow as parameters
        metric_env_vars=[],  # Environment variables to log to MLflow as metrics
    ),
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
