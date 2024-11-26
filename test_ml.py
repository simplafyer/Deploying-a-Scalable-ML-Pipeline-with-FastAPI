import pytest
import pandas as pd
import os
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from ml.data import process_data
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    train_model)

@pytest.fixture(scope="session")
def data():
    project_path = r"/mnt/c/Users/Jacob/Desktop/School/D501/Workspace/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
    data_path = os.path.join(project_path, "data", "census.csv")
    data = pd.read_csv(data_path)

    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test

@pytest.fixture(scope="session")
def model():
    project_path = r"/mnt/c/Users/Jacob/Desktop/School/D501/Workspace/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
    model_path = os.path.join(project_path, "model", "model.pkl")
    model = load_model(model_path)
    return model

@pytest.fixture(scope="session")
def metrics():

    project_path = r"/mnt/c/Users/Jacob/Desktop/School/D501/Workspace/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
    data_path = os.path.join(project_path, "data", "census.csv")
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=0.2, random_state=42)

# DO NOT MODIFY
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label='salary',  
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )      


    model = train_model(X_train,y_train)
    model_path = os.path.join(project_path, "model", "model.pkl")

    # load the model
    model = load_model(
        model_path
    ) 

    preds = inference(model,X_test)

    # Calculate and print the metrics
    p, r, fb = compute_model_metrics(y_test, preds)

    return p, r, fb



def test_metrics_correct_range(metrics):
    """
    Test to insure that all test fall within expected values less that 1.
    """
    p, r, fb = metrics
    assert fb <= 1 and p <= 1 and r <= 1


def test_model_expected(model):
    """
    Verify that the pipeline is indeed running the CategoricalNB() model
    """
    assert isinstance(model, CategoricalNB)


def test_sufficient_test_data(data):
    """
    Test if train_test_split broke down the data properly,
    setting aside approximately 20% of the data for testing.
    """
    train, test = data
    test_ratio = test.shape[0] / (test.shape[0] + train.shape[0])
    assert 0.19 <= test_ratio <= 0.21

