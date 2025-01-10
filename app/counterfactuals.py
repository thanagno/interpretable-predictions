import dice_ml
import pandas as pd

def generate_counterfactuals(full_data, user_data, desired_class=1):
    """
    Generates counterfactual explanations for the provided user data.

    Args:
        full_data: Full dataset (processed).
        user_data: Single-row DataFrame for user instance.
        desired_class: Desired target value.

    Returns:
        Counterfactual DataFrame.
    """
    # Convert bool to float
    bool_columns = full_data.select_dtypes(include=['bool']).columns
    full_data[bool_columns] = full_data[bool_columns].astype(float)
    user_data[bool_columns] = user_data[bool_columns].astype(float)

    # Define DiCE continuous features
    continuous_features = full_data.select_dtypes(include=['float']).columns.tolist()
    continuous_features.remove('Clicked on Ad')

    # Prepare data for DiCE
    dice_data = dice_ml.Data(
        dataframe=full_data,
        continuous_features=continuous_features,
        outcome_name="Clicked on Ad"
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    explainer = dice_ml.Dice(dice_data, dice_model)

    # Generate counterfactuals
    cf = explainer.generate_counterfactuals(
        query_instances=user_data,
        total_CFs=3,
        desired_class=desired_class
    )
    return cf.cf_examples_list[0].final_cfs_df
