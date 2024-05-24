import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import seaborn as sns

# Directory for output files
output_dir = 'C:/Users/matta/Code/.vscode/BeerwulfForcast/data/seasonal_featured_data'
viz_output_dir = 'C:/Users/matta/Code/.vscode/BeerwulfForcast/output/feature_vif'

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        
        # Convert 'Month' column to datetime if not already datetime
        if 'Month' in data.columns and not np.issubdtype(data['Month'].dtype, np.datetime64):
            data['Month'] = pd.to_datetime(data['Month'])
        
        # Check if 'Month' column exists and is in datetime format
        if 'Month' not in data.columns or not np.issubdtype(data['Month'].dtype, np.datetime64):
            raise ValueError("Month column is missing or not in datetime format")
        
        return data
    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")

def add_time_features(data):
    data['year'] = data['Month'].dt.year
    data['month'] = data['Month'].dt.month
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    return data

def decompose_seasonal_components(data, model='additive', freq='MS'):
    data = data.dropna(subset=['Sales'])
    data.set_index('Month', inplace=True)

    if len(data) < 24:
        print("Not enough data points for seasonal decomposition.")
        return data

    try:
        result = seasonal_decompose(data['Sales'], model=model, period=12)
        data['seasonal'] = result.seasonal
    except Exception as e:
        print(f"Decomposition failed: {e}")
        data['seasonal'] = np.nan

    return data.reset_index()

def add_interaction_features(data):
    data['seasonal_Product'] = data['seasonal'].values * (data['Product'].factorize()[0] + 1)
    return data

def save_features(data, filename):
    data.to_csv(f"{output_dir}/{filename}", index=False)


def main(selected_features):
    filepaths = [
        'C:/Users/matta/Code/.vscode/BeerwulfForcast/data/ProductA_sales_data.csv',
        'C:/Users/matta/Code/.vscode/BeerwulfForcast/data/ProductB_sales_data.csv',
        'C:/Users/matta/Code/.vscode/BeerwulfForcast/data/ProductC_sales_data.csv'
    ]

    results = []

    for filepath in filepaths:
        product_name = filepath.split('/')[-1].replace('_sales_data.csv', '')
        data = load_data(filepath)
        
        data = add_time_features(data)
        data = decompose_seasonal_components(data)
        data = add_interaction_features(data)
        
        filename = filepath.split('/')[-1].replace('.csv', '_featured.csv')
        save_features(data, filename)

    

if __name__ == "__main__":
    selected_features = ['time', 'decompose', 'interaction']
    main(selected_features)
