import pipelinex

mlflow_hooks = (
    pipelinex.MLflowBasicLoggerHook(
        uri="sqlite:///mlruns/sqlite.db",
        experiment_name="experiment_001",
        artifact_location="./mlruns/experiment_001",
        offset_hours=0,  # Specify the offset hour (e.g. 0 for UTC/GMT +00:00) to log in MLflow
    ),  # Configure and log duration time for the pipeline
    pipelinex.MLflowCatalogLoggerHook(
        auto=True,  # If True (default), for each dataset (Python func input/output) not listed in catalog, log as a metric for {float, int} types, and log as a param for {str, list, tuple, dict, set} types.
    ),  # Set dataset_name attribute in each MLflowDataSet instance
    pipelinex.MLflowArtifactsLoggerHook(
        filepaths_before_pipeline_run=[
            "conf/base/parameters.yml"
        ],  # Optionally specify the file paths to log before the pipeline runs
        filepaths_after_pipeline_run=[
            "logs/info.log",
            "logs/errors.log",
        ],  # Optionally specify the file paths to log after the pipeline runs
    ),
    pipelinex.MLflowEnvVarsLoggerHook(
        param_env_vars=[
            "HOSTNAME"
        ],  # Environment variables to log to MLflow as parameters
        metric_env_vars=[],  # Environment variables to log to MLflow as metrics
    ),
    pipelinex.MLflowTimeLoggerHook(),  # Log duration time to run each node (task)
    pipelinex.AddTransformersHook(
        transformers=[
            pipelinex.MLflowIOTimeLoggerTransformer()  # Log duration time to load and save each dataset
        ],
    ),  # Add transformers
)
