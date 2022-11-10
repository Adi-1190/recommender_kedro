import pandas as pd
from surprise import Reader
from surprise import Dataset

"""
Nodes for Data preprocessing
"""

def parse_ratings(data:pd.DataFrame) -> pd.DataFrame:
    data = data.loc[:,['user_id','movie_id','rating']]
    return data

def get_top20(ratings_data:pd.DataFrame) -> pd.DataFrame:
    """Returns top 20 highest rated movies

    Args:
        data : Raw user-ratings data.
    Returns:
        Sorted list of top 20 highest rated movies 
    """

    top_20=ratings_data.groupby(by='movie_id').sum().sort_values(by='rating', ascending=False)
    top_20.reset_index(inplace=True)

    top_20=top_20['movie_id'][:20].values

    return pd.Series(top_20)

def get_designMatrix(ratings_data:pd.DataFrame) -> pd.DataFrame:
    """Returns Design Matrix to be used by matrix factorisation algorithms 

    Args:
        data : Raw user-ratings data.
    Returns:
        User*Ratings Design Matrix
    """

    reader = Reader(rating_scale=(1, 5))

    processed_data = Dataset.load_from_df(ratings_data, reader)

    return processed_data
