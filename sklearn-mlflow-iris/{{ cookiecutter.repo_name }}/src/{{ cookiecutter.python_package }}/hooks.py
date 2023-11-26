"""
API document about the Kedro hooks for MLflow is at:
https://pipelinex.readthedocs.io/en/latest/pipelinex.mlflow_on_kedro.hooks.mlflow.html
"""

import pipelinex

hooks = (
    pipelinex.MLflowBasicLoggerHook(
        uri="sqlite:///mlruns/sqlite.db",
        experiment_name="experiment_001",
        artifact_location="./mlruns/experiment_001",
        offset_hours=0,
    ),
    pipelinex.MLflowCatalogLoggerHook(
        auto=True,
    ),
    pipelinex.MLflowArtifactsLoggerHook(
        filepaths_before_pipeline_run=["conf/base/parameters.yml"],
        filepaths_after_pipeline_run=[
            "info.log",
            "errors.log",
        ],
    ),
    pipelinex.MLflowEnvVarsLoggerHook(
        param_env_vars=["HOSTNAME"],
        metric_env_vars=[],
    ),
    pipelinex.MLflowTimeLoggerHook(),
)
