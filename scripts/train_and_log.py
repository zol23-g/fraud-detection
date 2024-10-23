import os
import pickle
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification  # For example purposes

# Function to create the models directory in the main project folder
def create_models_dir():
    # Assuming this script is inside a subfolder, we move up to the main project folder
    main_project_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    if not os.path.exists(main_project_dir):
        os.makedirs(main_project_dir)

# Function to load the data and prepare train-test split
def load_and_split_data(X, y, test_size=0.3):
    return train_test_split(X, y, test_size=test_size, random_state=42)

# Function to train a model and return predictions
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model and print structured report
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Structured outcome
    print(f"\n{'='*30}")
    print(f"Model Evaluation: {model_name}")
    print(f"{'='*30}")
    print(classification_report(y_test, y_pred))
    
    return report

# Function to log to MLflow
def log_to_mlflow(model_name, model, report):
    mlflow.set_experiment("fraud-detection")
    
    with mlflow.start_run():
        # Log model parameters (if applicable)
        mlflow.log_param("model_name", model_name)
        
        # Log metrics
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("recall", report["1"]["recall"])
        mlflow.log_metric("f1_score", report["1"]["f1-score"])
        
        # Log model itself
        mlflow.sklearn.log_model(model, model_name)

# Function to save the model as a pickle file in the main project's 'models' directory
def save_model(model, model_name):
    create_models_dir()
    main_project_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(main_project_dir, f"{model_name}.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Structured outcome for saving model
    print(f"\n{'-'*30}")
    print(f"Model saved at: {model_path}")
    print(f"{'-'*30}")

# Function to train, evaluate, log, and save models with structured printing
def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test):
    print(f"\n{'#'*40}")
    print(f"Training and Evaluating: {model_name}")
    print(f"{'#'*40}")

    # Train the model
    trained_model = train_model(model, X_train, y_train)
    
    # Evaluate the model
    report = evaluate_model(trained_model, X_test, y_test, model_name)
    
    # Log to MLflow
    log_to_mlflow(model_name, trained_model, report)
    
    # Save the model
    save_model(trained_model, model_name)
    
    return trained_model, report
