from sklearn.model_selection import  KFold, GridSearchCV, cross_val_score, ParameterGrid
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import os
import shap
import pickle
import csv

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

output_dir = 'C:/Users/matta/Code/.vscode/BeerwulfForcast/output/model_training/gb_season'
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
            'seasonal', 'seasonal_Product'
        ])
    ], remainder='drop')  # Ensuring only specified columns are included

# Parameter Grid Setup
def get_param_grid():
    return {
        'rfecv__step': [1],
        'rfecv__min_features_to_select': [5, 10],
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.01, 0.1],
        'regressor__max_depth': [3, 5]
    }

# Model Building
def build_model(preprocess_pipeline, rfecv_params, regressor_params):
    logging.info(f"Building model with RFECV params: {rfecv_params} and regressor params: {regressor_params}")
    return Pipeline([
        ('preprocessor', preprocess_pipeline),
        ('rfecv', RFECV(estimator=GradientBoostingRegressor(), **rfecv_params, cv=5)),
        ('regressor', GradientBoostingRegressor(**regressor_params))
    ])

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

    with open(os.path.join(output_dir, 'gb_season_performance.csv'), 'a', newline='') as file:
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

def evaluate_and_log_model(rfecv_param_set, regressor_params, preprocess_pipeline, X_train, X_test, y_train, y_test, output_dir):
    model_name = "GradientBoostingRegressor"
    logging.info(f"Building model with RFECV params: {rfecv_param_set} and regressor params: {regressor_params}")
    model = build_model(preprocess_pipeline, rfecv_param_set, regressor_params)
    
    # Starting nested cross-validation and obtaining mean score
    logging.info(f"Starting nested cross-validation for {model_name} with RFECV params: {rfecv_param_set} and regressor params: {regressor_params}")
    mean_score = nested_cross_validation(model, X_train, y_train, {'regressor__' + k: [v] for k, v in regressor_params.items()})
    logging.info(f"{model_name} - Mean Nested CV Score: {mean_score}")

    # Starting model evaluation
    logging.info(f"Starting evaluation for {model_name}")
    rmse, mae = evaluate_model(model, X_train, X_test, y_train, y_test, model_name, output_dir)

    # Logging results to CSV including the nested CV score
    with open(os.path.join(output_dir, 'gb_season_performance.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([model_name, mean_score, rmse, mae])

    logging.info(f"Finished evaluation for {model_name}")
    return rmse, mae

def save_model_and_features(model, output_dir):
    # Save the GradientBoostingRegressor and RFECV support mask
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model.named_steps['regressor'], f)
    with open(os.path.join(output_dir, 'features_mask.pkl'), 'wb') as f:
        pickle.dump(model.named_steps['rfecv'].support_, f)

def load_model_and_features(output_dir):
    with open(os.path.join(output_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(output_dir, 'features_mask.pkl'), 'rb') as f:
        features_mask = pickle.load(f)
    return model, features_mask

def make_future_predictions(model, features_mask, preprocess_pipeline, data, products, output_dir):
    predictions = {}
    for product in products:
        future_date = pd.Timestamp('2024-12-01')
        future_data = pd.DataFrame({
            'year': [future_date.year],
            'month': [future_date.month],
            'month_sin': [np.sin(2 * np.pi * future_date.month / 12)],
            'month_cos': [np.cos(2 * np.pi * future_date.month / 12)],
            'Product': [product],
            'seasonal': [data[data['Month'].dt.month == future_date.month]['seasonal'].mean()],
            'seasonal_Product': [data[(data['Month'].dt.month == future_date.month) & (data['Product'] == product)]['seasonal_Product'].mean()]
        })

        # Prepare data using the preprocessor and apply the feature mask
        X_future = preprocess_pipeline.transform(future_data)
        X_future_selected = X_future[:, features_mask]

        # Predict and store the result
        prediction = model.predict(X_future_selected)
        predictions[product] = prediction[0]

    # Output predictions to a CSV file
    with open(os.path.join(output_dir, 'future_predictions.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Product', 'Predicted_Sales'])
        for product, prediction in predictions.items():
            writer.writerow([product, prediction])

    return predictions

def main():
    filepaths = {
        'ProductA': 'C:/Users/matta/Code/.vscode/BeerwulfForcast/data/seasonal_featured_data/ProductA_sales_data_featured.csv',
        'ProductB': 'C:/Users/matta/Code/.vscode/BeerwulfForcast/data/seasonal_featured_data/ProductB_sales_data_featured.csv',
        'ProductC': 'C:/Users/matta/Code/.vscode/BeerwulfForcast/data/seasonal_featured_data/ProductC_sales_data_featured.csv'
    }

    # Load and prepare data
    data = load_and_prepare_data(filepaths)
    X, y = setup_data(data)
    preprocess_pipeline = build_preprocessing_pipeline()
    param_grid = get_param_grid()

    # Initialize performance log
    with open(os.path.join(output_dir, 'gb_season_performance.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Mean Nested CV Score', 'RMSE', 'MAE'])

    # Execute model evaluation across all data
    rfecv_params_list = [{key.split('__')[1]: value for key, value in param.items()} for param in ParameterGrid({k: v for k, v in param_grid.items() if k.startswith('rfecv__')})]
    regressor_params_list = [{key.split('__')[1]: value for key, value in param.items()} for param in ParameterGrid({k: v for k, v in param_grid.items() if k.startswith('regressor__')})]

    for rfecv_param_set in rfecv_params_list:
        for regressor_params in regressor_params_list:
            evaluate_and_log_model(rfecv_param_set, regressor_params, preprocess_pipeline, X, X, y, y, output_dir)
    
    # Build, fit, and save the model
    model = build_model(preprocess_pipeline, rfecv_params_list[0], regressor_params_list[0])
    model.fit(X, y)
    save_model_and_features(model, output_dir)

    # Load model and feature mask for future predictions
    loaded_model, features_mask = load_model_and_features(output_dir)
    products = filepaths.keys()
    future_predictions = make_future_predictions(loaded_model, features_mask, preprocess_pipeline, data, products, output_dir)


    # Output predictions
    for product, prediction in future_predictions.items():
        print(f"Predicted sales for {product} on December 1, 2024: {prediction}")

if __name__ == "__main__":
    main()