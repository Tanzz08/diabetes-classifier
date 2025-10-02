import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import sys
import os
import warnings
from sklearn.model_selection import train_test_split

if __name__ == "__main__": 
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    # file dataset
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetes-preprocessing.csv")
    data = pd.read_csv(file_path)
    
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop('diabetes_stage', axis=1),
        data['diabetes_stage'],
        random_state=42, 
        test_size=0.2,
        stratify=data['diabetes_stage']
    )

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

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model',
            input_example=input_example
        )

        # evaluas
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average="weighted"))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average="weighted"))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="weighted"))
        