import logging
from typing import Dict, Tuple, List
import time

import pandas as pd
from surprise import Reader, SVD, accuracy
from surprise import Dataset
from surprise.model_selection import train_test_split
import surprise as surprise

"""
Nodes for Data Science
"""

def split_data(data:surprise.dataset.DatasetAutoFolds) -> Tuple:
    """Splits data into training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    trainset, testset = train_test_split(data, test_size=0.25)  

    return trainset, testset


def train_model(trainset:surprise.trainset.Trainset) -> surprise.prediction_algorithms.matrix_factorization.SVD:
    """Trains the SVD model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """

    svd_algo = SVD(n_epochs=10,
                lr_all=0.005,
                reg_all=0.4)

    start=time.time()
    svd_algo.fit(trainset)
    stop = time.time()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Model training took {round(stop - start,10)}s for {trainset.n_ratings} ratings")

    return svd_algo


def evaluate_model(
                    svd_algo: surprise.prediction_algorithms.matrix_factorization.SVD,
                    testset:List
                                    ) -> pd.DataFrame:

    """Returns Design Matrix to be used by matrix factorisation algorithms 

    Args:
        data : Raw user-ratings data.
    Returns:
        User*Ratings Design Matrix
    """

    predictions = svd_algo.test(testset)
    score=accuracy.rmse(predictions)
    logger = logging.getLogger(__name__)
    logger.info("Model has a RMSE of %.3f on test data.", score)