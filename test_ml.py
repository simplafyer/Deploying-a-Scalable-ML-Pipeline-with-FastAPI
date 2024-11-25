import pytest
from train_model import X_train, X_test, p, r, fb
from sklearn.naive_bayes import CategoricalNB

# TODO: implement the first test. Change the function name and input as needed
def test_metrics_correct_range(r,p,fb):
    """
    Test to insure that all test fall within expected values less that 1.
    """
    assert fb <= 1 and p <= 1 and r <= 1


# TODO: implement the second test. Change the function name and input as needed
def test_model_expected(model):
    """
    Verify that the pipeline is indeed running the CategoricalNB() model
    """
    assert isinstance(model, CategoricalNB)


def sufficient_test_data(X_train, X_test):
    """
    Test if train_test_split broke down the data properly,
    setting aside approximately 20% of the data for testing.
    """
    test_ratio = X_test.size / (X_test.size + X_train.size)
    assert 0.19 <= test_ratio <= 0.21

