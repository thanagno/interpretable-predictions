import pandas as pd
# from app.preprocess import preprocess_data

# # Load test data
# # Load Dataset
data_path = 'data/ad_10000records.csv'
data = pd.read_csv(data_path)

# # Test preprocessing
# preprocessed_data = preprocess_data(data)
# print(preprocessed_data.head())


# from app.preprocess import preprocess_data

# data_path = "data/ad_10000records.csv"
# preprocessed_data = preprocess_data(data_path)

print('################################################')

from app.preprocess import preprocess_data
from app.model import train_binary_classifier, generate_counterfactuals, predict_and_explain_lime

# # Preprocess the dataset
preprocessed_data = preprocess_data(data)

# Train the model
model, X_train, X_test, y_train, y_test, feature_names = train_binary_classifier(preprocessed_data)

print('################################################')

from app.model import predict

# Test with some input data (ensure it matches feature structure)
# Test with some input data (ensure it matches feature structure)
input_data = X_test.iloc[:5].copy()  # Using the first 5 test instances for demonstration

predictions = predict(input_data)

# Compare predictions with true labels
print("Predictions vs True Labels:")
for i, pred in enumerate(predictions):
    print(f"Instance {i+1}: Prediction: {pred}, Truth: {y_test.iloc[i]}")

print('3 ################################################')

# # Explain a prediction for a specific user index
# user_index = 0  # For example, the first test instance
# prediction, lime_explanation = predict_and_explain_lime(model, X_train, X_test, user_index)

# print("Prediction:", prediction)


# Choose a user index for testing
# user_index = 0  # Example: First test instance

# # Run prediction and explanation
# prediction, lime_explanation = predict_and_explain_lime(model, X_train, X_test, user_index)

# # Output prediction and explanation
# print(f"Prediction for user index {user_index}: {prediction[0]} (True Label: {y_test.iloc[user_index]})")

# # Save LIME explanation to an HTML file for visualization
# lime_explanation.save_to_file("lime_explanation.html")
# print("LIME Explanation saved to 'lime_explanation.html'.")

# Predict and explain for a specific user index
user_index = 0  # Example user index
prediction, lime_explanation = predict_and_explain_lime(model, X_train, X_test, user_index)

# Display the prediction and true label
print(f"Prediction for user index {user_index}: {prediction} (True Label: {y_test.iloc[user_index]})")

# Save the LIME explanation to an HTML file
lime_explanation.save_to_file("lime_explanation.html")
print("LIME Explanation saved to 'lime_explanation.html'.")

# Combine features and target for training data
# Select user data for counterfactual generation
user_data = X_test.iloc[[user_index]]  # Ensure DataFrame structure

# Combine features and target for training data
full_data = pd.concat([X_train, y_train.rename('Clicked on Ad')], axis=1)

# Specify features to vary
features_to_vary = ['Age', 'Daily Time Spent on Site', 'Daily Internet Usage']

# # Generate counterfactuals
# try:
#     cf = generate_counterfactuals(model, full_data, user_data, desired_class=1, features_to_vary=features_to_vary)
#     if cf:
#         print("Counterfactuals generated successfully for selected features!")
# except Exception as e:
#     print(f"Error during counterfactual generation: {e}")

# Generate Counterfactuals for Negative Prediction
if prediction == 0:  # If the prediction is negative
    full_data = pd.concat([X_train, y_train.rename('Clicked on Ad')], axis=1)
    user_data = X_test.iloc[[user_index]]  # Select the user's data as a DataFrame
    generate_counterfactuals(model, full_data, user_data, desired_class=1)
