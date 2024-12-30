import dice_ml
import pandas as pd

def generate_counterfactuals(model, data: pd.DataFrame, user_instance: pd.DataFrame, desired_class=1):
    """Generates counterfactuals using DiCE."""
    continuous_features = data.select_dtypes(include=['float']).columns.tolist()
    continuous_features.remove('Clicked on Ad')

    dice_data = dice_ml.Data(dataframe=data, continuous_features=continuous_features, outcome_name='Clicked on Ad')
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    explainer = dice_ml.Dice(dice_data, dice_model)
    
    cf = explainer.generate_counterfactuals(
        query_instances=user_instance, 
        total_CFs=3, 
        desired_class=desired_class
    )
    return cf.cf_examples_list[0].final_cfs_df
