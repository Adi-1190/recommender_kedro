"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_model, split_data, train_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [node(
                func=split_data,
                inputs="processed_data",
                outputs=["trainset","testset"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs="trainset",
                outputs="svd_algo",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["svd_algo", "testset"],
                outputs=None,
                name="evaluate_model_node",
            ),

    ]
)
