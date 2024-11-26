import pytest
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split


def test_metrics_correct_range(compute_model_metrics):
    """
    Test to insure that all test fall within expected values less that 1.
    """
    assert fbeta <= 1 and precision <= 1 and recall <= 1


def test_model_expected(load_model):
    """
    Verify that the pipeline is indeed running the CategoricalNB() model
    """
    assert isinstance(model, CategoricalNB)


def sufficient_test_data(train_test_split):
    """
    Test if train_test_split broke down the data properly,
    setting aside approximately 20% of the data for testing.
    """


    
    test_ratio = X_test.size / (X_test.size + X_train.size)
    assert 0.19 <= test_ratio <= 0.21

