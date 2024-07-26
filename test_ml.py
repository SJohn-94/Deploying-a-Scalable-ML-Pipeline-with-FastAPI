import pytest
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.model import train_model, inference, compute_model_metrics

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# TODO: add necessary import

def data():
    project_path = "/home/sam_john/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
    data_path = os.path.join(project_path, "data", "census.csv")
    data = pd.read_csv(data_path, sep=',')
    return data

# TODO: implement the first test. Change the function name and input as needed
def test_model():
    """
    Tests whether the model being used in the project is RandomForestClassifier
    """
    # Your code here

    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)



# TODO: implement the second test. Change the function name and input as needed
def test_data_metrics():
    """
    Testing that the metrics fall between 0 and 1 
    """
    # Your code here

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


# TODO: implement the third test. Change the function name and input as needed
def test_datatype():
    """
    Tests whether or not precision, recall, and fbeta metrics are of float datatype
    """
    # Your code here

    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

