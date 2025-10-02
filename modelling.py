import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import sys
import os
import warnings

if __name__ == "__main__": 
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    # file dataset
    base_path = os.path.dirname(os.path.abspath(__file__))
    X_train = pd.read_csv(os.path.join(base_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(base_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(base_path, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(base_path, "y_test.csv"))

    input_example = X_train[0:5]

    with mlflow.start_run():
        mlflow.autolog()
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=3,
            min_samples_split=5, 
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mlflow.log_params(model.get_params())

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model',
            input_example=input_example
        )

        # evaluasi
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average="weighted"))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average="weighted"))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="weighted"))
        