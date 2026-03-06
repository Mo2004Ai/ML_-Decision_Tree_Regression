# House Rent Analysis & Prediction

## Project Description
Analyze house rent prices and predict them using a Decision Tree model based on property features such as number of bedrooms, area, number of bathrooms, city, and locality.

## Libraries Used
- `pandas` and `numpy` for data processing.
- `matplotlib` and `seaborn` for visualizations and exploratory analysis.
- `scikit-learn` for building the model, data preprocessing, and evaluation.

## Project Steps
1. **Data Loading**: Load the `House_Rent_Dataset.xlsx` file.
2. **Data Understanding**: Inspect the data, check for missing values, and drop unnecessary columns.
3. **Statistical Analysis**: Detect outliers and handle them using logarithmic transformation.
4. **Relationship Analysis**: Visualize relationships between rent prices and other variables.
5. **Data Preparation for Modeling**: Encode categorical columns using LabelEncoder and OneHotEncoder.
6. **Data Splitting**: Split data into training, validation, and test sets.
7. **Model Building**: Train a Decision Tree Regressor to predict rent prices.
8. **Model Evaluation**: Calculate MSE, MAE, and R² Score on training, validation, and test sets.

## How to Use
1. Make sure the dataset file exists in the correct path or update the path in the code.
2. Install the required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
