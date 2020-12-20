from kedro.pipeline import Pipeline, node

from .nodes import init_model, train_model, evaluate_model, run_inference


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(func=init_model, inputs=None, outputs="init_model"),
            node(
                func=train_model,
                inputs=[
                    "init_model",
                    "train_df",
                    "params:features",
                    "params:target",
                ],
                outputs="model",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "model",
                    "train_df",
                    "params:features",
                    "params:target",
                ],
                outputs="score",
            ),
            node(
                func=run_inference,
                inputs=["model", "test_df", "params:features"],
                outputs="pred_df",
            ),
        ]
    )
