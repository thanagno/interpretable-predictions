import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model(data: pd.DataFrame, model_path: str):
    """Trains a Random Forest model."""
    y = data['Clicked on Ad']
    X = data.drop(columns=['Clicked on Ad'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, model_path)
    print("Model saved successfully!")
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
