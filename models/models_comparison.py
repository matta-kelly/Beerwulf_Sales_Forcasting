from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score, ParameterGrid
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import logging
import os
import shap
import csv
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

output_dir = 'C:/Users/matta/Code/.vscode/BeerwulfForcast/output/model_training/models_comparison'
os.makedirs(output_dir, exist_ok=True)

# Data Loading and Preparation
def load_and_prepare_data(filepaths):
    data = pd.concat([pd.read_csv(fp).assign(Product=prod) for prod, fp in filepaths.items()])
    data.dropna(inplace=True)
    data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m-%d')
    data['Target_Sales'] = data.groupby(['Product'])['Sales'].shift(-6)
    data.dropna(subset=['Target_Sales'], inplace=True)
    return data

def setup_data(data):
    X = data.drop(['Sales', 'Target_Sales', 'Month'], axis=1)
    y = data['Target_Sales']
    return X, y  

def build_preprocessing_pipeline():
    return ColumnTransformer(transformers=[
        ('encoder', OneHotEncoder(), ['Product']),
        ('scaler', StandardScaler(), [
            'year', 'month', 'month_sin', 'month_cos',
            'sales_lag_1', 'sales_lag_2', 'sales_lag_3',
            'rolling_mean_3', 'rolling_std_3', 'rolling_mean_6',
            'rolling_std_6', 'ewm_alpha_0.3', 'ewm_alpha_0.5',
             'seasonal','seasonal_Product'
        ])
    ], remainder='passthrough')

# Parameter Grid Setup
def get_param_grids():
    return {
        'LinearRegression': {
            'rfecv__step': [1],
            'rfecv__min_features_to_select': [5],
            'regressor__fit_intercept': [True]
        },
        'RandomForestRegressor': {
            'rfecv__step': [1],
            'rfecv__min_features_to_select': [5],
            'regressor__n_estimators': [100],  # Reduced from three options to one
            'regressor__max_features': ['sqrt'],  # Reduced complexity
            'regressor__max_depth': [10]  # Single depth choice
        },
        'GradientBoostingRegressor': {
            'rfecv__step': [1],
            'rfecv__min_features_to_select': [5],
            'regressor__n_estimators': [100],  # Fewer estimators
            'regressor__learning_rate': [0.1],  # Single learning rate
            'regressor__max_depth': [3]  # Less depth
        }
    }


# Model Building
def build_model(preprocess_pipeline, model_class, rfecv_params, regressor_params):
    logging.info(f"Building model with RFECV params: {rfecv_params} and regressor params: {regressor_params}")
    return Pipeline([
        ('preprocessor', preprocess_pipeline),
        ('rfecv', RFECV(estimator=model_class(), **rfecv_params, cv=5)),
        ('regressor', model_class(**regressor_params))
    ])

def build_models(preprocess_pipeline, param_grids):
    models = []
    for model_name, params in param_grids.items():
        rfecv_params = {key.split('__')[1]: value[0] for key, value in params.items() if key.startswith('rfecv__')}
        regressor_params = {key.split('__')[1]: value for key, value in params.items() if key.startswith('regressor__')}
        model_class = globals()[model_name]
        models.append(build_model(preprocess_pipeline, model_class, rfecv_params, regressor_params))
    return models

# Model Evaluation
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, output_dir):
    logging.info(f"Starting evaluation for {model_name}")
    model.fit(X_train, y_train)
    logging.info(f"Finished fitting {model_name}")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    logging.info(f"{model_name} - RMSE: {rmse}, MAE: {mae}")

    logging.info(f"Saving prediction plot for {model_name}")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f"{model_name} Predictions")
    plt.savefig(os.path.join(output_dir, f"{model_name}_predictions.png"))
    plt.close()

    with open(os.path.join(output_dir, 'model_comparison.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([model_name, rmse, mae])

    logging.info(f"Saving residual plot for {model_name}")
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(f"{model_name} Residuals")
    plt.savefig(os.path.join(output_dir, f"{model_name}_residuals.png"))
    plt.close()

    logging.info(f"Finished evaluation for {model_name}")
    return rmse, mae

# Model Interpretation
def interpret_model(model, X_train, model_name):
    logging.info(f"Starting model interpretation for {model_name}")

    # Extract the preprocessor and regressor from the pipeline
    preprocessor = model.named_steps['preprocessor']
    rfecv = model.named_steps['rfecv']
    regressor = model.named_steps['regressor']

    logging.info(f"Transforming training data for {model_name}")
    # Transform the training data using the pipeline's preprocessor
    X_transformed = preprocessor.transform(X_train)

    logging.info(f"Applying RFECV selector for {model_name}")
    # Apply the rfecv to get the subset of features that were selected by RFECV
    X_transformed_selected = rfecv.transform(X_transformed)

    logging.info(f"Selecting SHAP explainer for {model_name}")
    # Determine the appropriate SHAP explainer based on the model type
    if isinstance(regressor, (RandomForestRegressor, GradientBoostingRegressor)):
        explainer = shap.TreeExplainer(regressor)
    else:
        explainer = shap.KernelExplainer(regressor.predict, X_transformed_selected)

    logging.info(f"Computing SHAP values for {model_name}")
    # Compute SHAP values
    shap_values = explainer(X_transformed_selected)

    logging.info(f"Generating SHAP summary plot for {model_name}")
    # Get the feature names after RFECV selection
    feature_names = [name for i, name in enumerate(preprocessor.get_feature_names_out()) if rfecv.support_[i]]
    # Save SHAP summary plot
    '''plt.figure()
    shap.summary_plot(shap_values, X_transformed_selected, feature_names=feature_names)
    plt.title(f"{model_name} SHAP Summary")
    plt.savefig(os.path.join(output_dir, f"{model_name}_shap_summary.png"))
    plt.close()'''

# Nested Cross-Validation
def nested_cross_validation(model, X, y, param_grid):
    logging.info("Starting nested cross-validation")
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, scoring='neg_mean_squared_error', n_jobs=-1)
    logging.info("Starting cross_val_score for nested CV")
    nested_score = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='neg_mean_squared_error', n_jobs=-1)
    logging.info("Finished nested cross-validation")
    return nested_score.mean()

def evaluate_and_log_model(model_class, model_name, rfecv_param_set, regressor_params, preprocess_pipeline, X_train, X_test, y_train, y_test, output_dir):
    logging.info(f"Building model with RFECV params: {rfecv_param_set} and regressor params: {regressor_params}")
    model = build_model(preprocess_pipeline, globals()[model_name], rfecv_param_set, regressor_params)
    logging.info(f"Starting nested cross-validation for {model_name} with RFECV params: {rfecv_param_set} and regressor params: {regressor_params}")
    mean_score = nested_cross_validation(model, X_train, y_train, {'regressor__' + k: [v] for k, v in regressor_params.items()})
    logging.info(f"{model_name} - Mean Nested CV Score: {mean_score}")

    logging.info(f"Starting evaluation for {model_name}")
    rmse, mae = evaluate_model(model, X_train, X_test, y_train, y_test, model_name, output_dir)
    interpret_model(model, X_train, model_name)

    with open(os.path.join(output_dir, 'model_comparison.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([model_name, mean_score, rmse, mae])

def main():
    filepaths = {
        'ProductA': 'C:/Users/matta/Code/.vscode/BeerwulfForcast/data/full_featured_data/ProductA_sales_data_featured.csv',
        'ProductB': 'C:/Users/matta/Code/.vscode/BeerwulfForcast/data/full_featured_data/ProductB_sales_data_featured.csv',
        'ProductC': 'C:/Users/matta/Code/.vscode/BeerwulfForcast/data/full_featured_data/ProductC_sales_data_featured.csv'
    }

    # Load and prepare data
    data = load_and_prepare_data(filepaths)
    X, y = setup_data(data)
    preprocess_pipeline = build_preprocessing_pipeline()
    param_grids = get_param_grids()

    # Initialize performance log
    with open(os.path.join(output_dir, 'model_comparison.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Mean Nested CV Score', 'RMSE', 'MAE'])

    # Execute model evaluation across all data
    for model_name, params in param_grids.items():
        rfecv_params = {key.split('__')[1]: value for key, value in params.items() if key.startswith('rfecv__')}
        regressor_params_list = [{key.split('__')[1]: value for key, value in param.items()} for param in ParameterGrid({k: v for k, v in params.items() if k.startswith('regressor__')})]

        model_class = globals()[model_name]
        for rfecv_param_set in ParameterGrid(rfecv_params):
            for regressor_params in regressor_params_list:
                evaluate_and_log_model(model_class, model_name, rfecv_param_set, regressor_params, preprocess_pipeline, X, X, y, y, output_dir)

if __name__ == "__main__":
    main()