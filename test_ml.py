
import os
import pytest
import pandas as pd
from train_model import X_train, X_test, p, r, fb
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from ml.model import load_model

project_path = r"/mnt/c/Users/Jacob/Desktop/School/D501/Workspace/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv(data_path)

train, test = train_test_split(data, test_size=0.2, random_state=42)

model_path = os.path.join(project_path, "model", "model.pkl")
model = load_model(
    model_path
) 



# TODO: implement the first test. Change the function name and input as needed
#def test_metrics_correct_range(r,p,fb):
    """
    Test to insure that all test fall within expected values less that 1.
    """
   # assert fb <= 1 and p <= 1 and r <= 1


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

