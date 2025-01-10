from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from lime.lime_tabular import LimeTabularExplainer
import pickle
import pandas as pd
import dice_ml
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# Path to save the trained model
MODEL_PATH = "models/model.pkl"

# Function to train the binary classifier
def train_binary_classifier(data):
    """
    Trains a binary classifier using the preprocessed dataset.

    Args:
        data (pd.DataFrame): Preprocessed dataset.

    Returns:
        model: Trained classifier.
        X_train, X_test, y_train, y_test: Train-test splits for features and target.
        feature_names: List of feature names.
    """
    # Convert timestamp features
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Hour'] = data['Timestamp'].dt.hour
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
    data['Month'] = data['Timestamp'].dt.month

    # One-hot encode categorical features
    categorical_features = ['Gender', 'Continent']
    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    data = data.astype({col: 'float' for col in data.select_dtypes(include=['uint8']).columns})


    # Drop unnecessary columns
    data.drop(columns=['Timestamp', 'Ad Topic Line'], inplace=True)

    # Separate features (X) and target (y)
    y = data['Clicked on Ad']
    X = data.drop(columns=['Clicked on Ad'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model, X_train, X_test, y_train, y_test, X.columns.tolist()

# Function to make predictions using the trained model
def predict(data):
    """
    Predicts the target variable for the given input data using the trained model.

    Args:
        data (pd.DataFrame): Input data for prediction.

    Returns:
        predictions: Predicted classes.
    """
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        raise Exception("Model file not found. Train the model first.")
    return model.predict(data)


# Function to explain predictions using LIME
# def predict_and_explain_lime(model, X_train, X_test, user_index):
#     """
#     Predicts if a user will interact with an entity and provides a LIME explanation.

#     Args:
#         model: Trained binary classifier.
#         X_train (pd.DataFrame): Training features.
#         X_test (pd.DataFrame): Test features.
#         user_index (int): Index of the user in the test set to explain.

#     Returns:
#         prediction: Predicted class (0 or 1).
#         explanation: LIME explanation object.
#     """
#     # Predict
#     # user_data = X_test.iloc[user_index].values.reshape(1, -1)
#     user_data = X_test.iloc[[user_index]]  # Retains feature names as a DataFrame

#     prediction = model.predict(user_data)

#     # Explain prediction using LIME
#     explainer = LimeTabularExplainer(
#         training_data=X_train.values,
#         feature_names=X_train.columns,
#         class_names=['Not Clicked', 'Clicked'],
#         mode='classification'
#     )
#     exp = explainer.explain_instance(X_test.iloc[user_index].values, model.predict_proba)
#     # exp.show_in_notebook()
#     return prediction, exp

# Function to explain predictions using LIME
def predict_and_explain_lime(model, X_train, X_test, user_index):
    """
    Predicts if a user will interact with an entity and provides a LIME explanation.

    Args:
        model: Trained binary classifier.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        user_index (int): Index of the user in the test set to explain.

    Returns:
        prediction: Predicted class (int).
        explanation: LIME explanation object.
    """
    # Predict using a DataFrame row to retain feature names
    user_data = X_test.iloc[[user_index]]  # Ensures DataFrame structure
    prediction = model.predict(user_data)[0]  # Extract scalar prediction

    # Create LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=['Not Clicked', 'Clicked'],
        mode='classification',
    )

    # Explain instance
    exp = explainer.explain_instance(
        data_row=X_test.iloc[user_index],
        predict_fn=model.predict_proba,
        num_features=6
    )

    # Print probability and feature contributions
    print(f"Prediction Probability: {model.predict_proba(user_data)}")
    for feature, weight in exp.as_list():
        print(f"Feature: {feature}, Weight: {weight}")

    return prediction, exp
