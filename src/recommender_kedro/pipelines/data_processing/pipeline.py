from kedro.pipeline import Pipeline, node, pipeline
from .nodes import parse_ratings, get_top20, get_designMatrix

"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
        node(
            func=parse_ratings,
            inputs="ratings",
            outputs="data", 
            name="parse_ratings_node"
            ),
        node(
            func=get_top20,
            inputs="data",
            outputs="top_20",
            name="get_top20_node"
            ),
        node(
            func=get_designMatrix,
            inputs="data",
            outputs="processed_data",
            name="get_designMatrix_node"
            ),
        ]
    )
