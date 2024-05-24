# Beerwulf Sales Forecasting

## Project Overview
This project is aimed at forecasting sales for Beerwulf using various machine learning models. The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, and model training and evaluation. The best-performing model, Gradient Boosting (GB), is used for both historical seasonal data and future sales predictions.

## Directory Structure
The project directory is organized as follows:

BeerwulfForcast/
│
├── data/
│   ├── assessment_data_set.csv
│   ├── ProductA_sales_data.csv
│   ├── ProductB_sales_data.csv
│   ├── ProductC_sales_data.csv
│   ├── Combined_sales_data.csv
│   └── full_featured_data/
│       ├── ProductA_sales_data_featured.csv
│       ├── ProductB_sales_data_featured.csv
│       └── ProductC_sales_data_featured.csv
│   └── seasonal_featured_data/
│       ├── ProductA_sales_data_featured.csv
│       ├── ProductB_sales_data_featured.csv
│       └── ProductC_sales_data_featured.csv
│
├── output/
│   ├── EDA_Output/
│   │   ├── EDA_Report.pdf
│   │   └── various EDA visualizations
│   ├── feature_vif/
│   ├── model_training/
│       ├── models_comparison/
│       │   ├── model_comparison.csv
│       │   └── various model comparison visualizations
│       ├── gb_full/
│       │   ├── gb_full_performance.csv
│       │   ├── model.pkl
│       │   ├── features_mask.pkl
│       │   └── predictions.csv
│       └── gb_season/
│           ├── gb_season_performance.csv
│           └── various seasonal model visualizations
│
├── preprocessing.py
├── EDA.py
├── feature_full.py
├── feature_seasonal.py
├── model_comparison.py
├── gb_full.py
└── gb_seasonal.py

# Workflow
### Data Preprocessing:

File: preprocessing.py
Functionality: Load the raw sales data, preprocess it, and split it by product, saving the cleaned data for each product as well as the combined dataset.
Output: Cleaned and split CSV files for each product in the data directory.

### Exploratory Data Analysis (EDA):

File: EDA.py
Functionality: Perform initial data review, temporal analysis, product-wise analysis, and correlation analysis. Generate visualizations and save them in the output/EDA_Output directory. Create a PDF report summarizing the findings.
Output: Various EDA visualizations and a PDF report.

### Feature Engineering:

Files: feature_full.py and feature_seasonal.py
Functionality: Generate features for both full dataset and seasonal dataset. This includes time-based features, lagged features, rolling statistics, exponential weighted means, and seasonal decomposition.
Output: Feature-engineered CSV files for each product in the data/full_featured_data and data/seasonal_featured_data directories.


### Model Training and Comparison:

File: model_comparison.py
Functionality: Compare multiple models (Linear Regression, Random Forest, Gradient Boosting) using cross-validation. Evaluate and log model performance, save plots of predictions and residuals.
Output: CSV file summarizing model performance and various comparison visualizations in the output/model_training/models_comparison directory.

### Gradient Boosting - Full Dataset:

File: gb_full.py
Functionality: Train a Gradient Boosting model on the full dataset. Evaluate model performance, interpret results using SHAP values, and save the model and feature mask for future predictions.
Output: Model performance CSV, saved model and feature mask, and predictions for the most recent data points in the output/model_training/gb_full directory.

### Gradient Boosting - Seasonal Dataset:

File: gb_seasonal.py
Functionality: Train a Gradient Boosting model on the seasonal dataset. Perform nested cross-validation, evaluate model performance, and make future predictions based on seasonal trends.
Output: Model performance CSV, and predictions for a manually specified future date in the output/model_training/gb_season directory.

# Detailed Explanation of Key Files

### preprocessing.py
This script loads the raw sales data, preprocesses it by converting date columns, interpolating missing sales data, and splitting the dataset by product. Each product's data is saved as a separate CSV file.

### #EDA.py
This script performs comprehensive exploratory data analysis (EDA) on the sales data. It includes initial data review, temporal analysis, product-wise analysis using seasonal decomposition, and correlation analysis. The results are visualized and saved as images, and a PDF report is generated.

### feature_full.py
This script performs feature engineering on the full dataset. It adds time-based features, lagged sales features, rolling statistics, exponential weighted means, and performs seasonal decomposition. The feature-engineered data is saved for each product.

### feature_seasonal.py
Similar to feature_full.py, but tailored for seasonal analysis. This script adds time-based features and performs seasonal decomposition to prepare the dataset for seasonal trend analysis.

### model_comparison.py
This script compares multiple models (Linear Regression, Random Forest, Gradient Boosting) using cross-validation. It evaluates model performance using metrics such as RMSE and MAE and saves visualizations of predictions and residuals.

### gb_full.py
This script trains a Gradient Boosting model on the full dataset. It performs nested cross-validation, evaluates the model, interprets it using SHAP values, and saves the model and feature mask. The model is used to make predictions based on the most recent data points.

### gb_seasonal.py
This script trains a Gradient Boosting model on the seasonal dataset. It performs nested cross-validation and evaluates the model. The model is then used to make future predictions based on seasonal trends.

# Running the Project

1. Set up the environment: Ensure all dependencies are installed by running pip install -r requirements.txt.
2. Preprocess the data: Run preprocessing.py to preprocess and split the raw data. 
3. Perform EDA: Run EDA.py to generate visualizations and the EDA report.
4. Feature Engineering: Run feature_full.py and feature_seasonal.py to create feature-engineered datasets.
5. Model Comparison: Run model_comparison.py to compare multiple models and identify the best one.
6. Train Gradient Boosting Models: Run gb_full.py and gb_seasonal.py to train and evaluate the Gradient Boosting models, and to make future predictions.

# Output
The output of each script is saved in the respective directories within the output folder. This includes EDA visualizations, model performance summaries, saved models, feature masks, and predictions.