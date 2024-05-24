import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from statsmodels.tsa.seasonal import seasonal_decompose


def save_plot(figure, filename):
    # Ensure directory exists
    output_dir = 'C:/Users/matta/Code/.vscode/BeerwulfForcast/output/EDA_Output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    figure.savefig(filepath)
    plt.close(figure)
    return filepath


def initial_data_review(data):
    results = {}
    data['Month'] = pd.to_datetime(data['Month'])
    expected_months = pd.date_range(start=data['Month'].min(), end=data['Month'].max(), freq='MS')
    results['missing_months'] = expected_months.difference(data['Month']).tolist()
    inconsistent_sales = data[data['Sales'] < 0]
    results['inconsistent_sales_entries'] = inconsistent_sales
    duplicates = data[data.duplicated(subset=['Month', 'Product'], keep=False)]
    results['duplicate_entries'] = duplicates
    return results, None

def temporal_analysis(data):
    results = {}
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'Product' in data.columns:
        for name, group in data.groupby('Product'):
            group.set_index('Month')['Sales'].plot(ax=ax, label=name)
        ax.legend(title='Product')
    else:
        data.set_index('Month')['Sales'].plot(ax=ax)
    ax.set_xlabel('Month')
    ax.set_ylabel('Sales')
    ax.set_title('Sales Over Time by Product')
    plt.draw()
    plt.pause(0.001)
    plot_filename = save_plot(fig, 'temporal_analysis.png')
    results['trend_analysis'] = "Overlay trend analysis visualized with missing data."
    return results, plot_filename


def product_wise_analysis(data):
    results = {}
    unique_products = data['Product'].unique()
    num_products = len(unique_products)

    # Adjust figure creation based on the number of products
    if num_products == 1:
        fig, ax = plt.subplots(nrows=4, figsize=(12, 16))  # One column, four rows for each component
    else:
        fig, axes = plt.subplots(nrows=num_products, ncols=4, figsize=(20, 5 * num_products))  # 4 plots per product

    for idx, product in enumerate(unique_products):
        product_data = data[data['Product'] == product]
        product_data.drop_duplicates(subset='Month', keep='first', inplace=True)  # Remove duplicates
        product_data.set_index('Month', inplace=True)
        product_data = product_data.asfreq('MS').ffill()  # Ensure it's monthly and fill missing data

        decomposition = seasonal_decompose(product_data['Sales'], model='additive')

        # Plot each component
        plot_base_name = f'{product}_behavioral_patterns.png'
        if num_products == 1:
            ax[0].plot(decomposition.trend)
            ax[0].set_title(f'{product} - Trend')
            ax[1].plot(decomposition.seasonal)
            ax[1].set_title(f'{product} - Seasonal')
            ax[2].plot(decomposition.resid)
            ax[2].set_title(f'{product} - Residual')
            ax[3].plot(decomposition.observed)
            ax[3].set_title(f'{product} - Observed')
        else:
            axes[idx, 0].plot(decomposition.trend)
            axes[idx, 0].set_title(f'{product} - Trend')
            axes[idx, 1].plot(decomposition.seasonal)
            axes[idx, 1].set_title(f'{product} - Seasonal')
            axes[idx, 2].plot(decomposition.resid)
            axes[idx, 2].set_title(f'{product} - Residual')
            axes[idx, 3].plot(decomposition.observed)
            axes[idx, 3].set_title(f'{product} - Observed')

        plt.tight_layout()
        plot_filename = save_plot(fig, plot_base_name)
        results[product] = "Decomposed into trend, seasonal, and residual components."
        results[f'{product}_plot'] = plot_filename  # Save the plot filename under a unique key

    return results, plot_filename  # This will only return the last plot's filename; adjust as needed for your reporting


def correlation_analysis(data):
    if 'Product' in data.columns:
        pivoted_data = data.pivot_table(index='Month', columns='Product', values='Sales', aggfunc=np.sum)
        pivoted_data.fillna(0, inplace=True)
        correlation_matrix = pivoted_data.corr()

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'shrink': .5}, ax=ax)
        plt.title('Correlation Matrix of Product Sales')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        plot_filename = save_plot(fig, 'correlation_matrix.png')
        return "Correlation matrix computed and visualized.", plot_filename
    else:
        return "No product information available to analyze correlations.", None


def visualization(data):
    # If I want to add other visualizations later
    results = {"info": "Visualization results placeholder"}
    return results, None

def run_eda(part, data):
    if part == 'initial':
        return initial_data_review(data)
    elif part == 'temporal':
        return temporal_analysis(data)
    elif part == 'product':
        return product_wise_analysis(data)
    elif part == 'correlation':
        return correlation_analysis(data)
    elif part == 'visual':
        return visualization(data)
    else:
        raise ValueError("Unknown EDA part specified")

def create_pdf_report(filepaths, output_path, parts):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    for filepath in filepaths:
        data = pd.read_csv(filepath, parse_dates=['Month'])
        filename = os.path.splitext(os.path.basename(filepath))[0]

        for part in parts:
            results, plot_filename = run_eda(part, data)
            story.append(Paragraph(f"{part.capitalize()} Analysis for {filename}", styles['Title']))
            story.append(Paragraph(str(results), styles['BodyText']))
            
            if plot_filename:
                img = Image(plot_filename)
                img._restrictSize(6 * inch, 4 * inch)
                story.append(img)
            
            story.append(Spacer(1, 12))

    doc.build(story)

def main(parts):
    filepaths = [
        'C:/Users/matta/Code/.vscode/BeerwulfForcast/data/Combined_sales_data.csv'
        #'C:/Users/matta/Code/.vscode/BeerwulfForcast/data/ProductA_sales_data.csv',
        #'C:/Users/matta/Code/.vscode/BeerwulfForcast/data/ProductB_sales_data.csv',
        #'C:/Users/matta/Code/.vscode/BeerwulfForcast/data/ProductC_sales_data.csv'
    ]
    output_path = 'C:/Users/matta/Code/.vscode/BeerwulfForcast/output/EDA_Output/EDA_Report.pdf'
    create_pdf_report(filepaths, output_path, parts)

if __name__ == "__main__":
    parts = ['initial','temporal','product','correlation']  # Specify the parts to run
    main(parts)
