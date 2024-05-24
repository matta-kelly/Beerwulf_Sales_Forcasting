import pandas as pd
import numpy as np

filepath = 'C:/Users/matta/Code/.vscode/BeerwulfForcast/data/assessment_data_set.csv'
output_dir = 'C:/Users/matta/Code/.vscode/BeerwulfForcast/data'

def preprocess_and_split_data(filepath, output_dir):
    """
    Load, preprocess, and split the data by product, saving each subset to a CSV file.
    
    Parameters:
        filepath (str): Path to the CSV file containing the data.
        output_dir (str): Directory where the output files will be saved.
        
    Returns:
        dict: A dictionary containing the combined and individual product DataFrames.
    """
    # Load the dataset
    data = pd.read_csv(filepath)
    
    # Convert 'Month' column to datetime
    data['Month'] = pd.to_datetime(data['Month'])
    
    # Split data by product
    product_frames = {}
    for product in data['Product'].unique():
        product_data = data[data['Product'] == product].copy()  # Create a copy to avoid modifying the original data
        product_data['Month'] = pd.to_datetime(product_data['Month'])  # Ensure 'Month' column is in datetime format
        
        # Find the introduction date of the product
        start_date = product_data['Month'].min()
        
        # Interpolate missing sales data only after the product was introduced
        product_data['Sales'] = product_data.apply(lambda row: row['Sales'] if row['Month'] >= start_date else np.nan, axis=1)
        product_data['Sales'] = product_data['Sales'].interpolate(method='linear')  # Interpolate missing sales
        
        product_frames[product] = product_data
    
    # Save each DataFrame
    product_frames['Combined'] = pd.concat(product_frames.values(), ignore_index=True)  # Include the combined data
    for key, frame in product_frames.items():
        frame.to_csv(f"{output_dir}/{key}_sales_data.csv", index=False)
    
    return product_frames

preprocess_and_split_data(filepath, output_dir)
