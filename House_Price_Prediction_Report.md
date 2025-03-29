# House Price Prediction - Technical Assignment Report

## 1. Dataset Description

The dataset used in this project is a house price dataset containing key features that affect property prices. The dataset includes numerical and categorical attributes such as:

- **LotArea**: Size of the lot in square feet.
- **OverallQual**: Overall material and finish quality.
- **OverallCond**: Overall condition rating of the house.
- **YearBuilt**: The year the house was constructed.
- **TotalBsmtSF**: Total square footage of the basement.
- **GrLivArea**: Above ground living area square feet.
- **FullBath**: Number of full bathrooms.
- **HalfBath**: Number of half bathrooms.
- **BedroomAbvGr**: Number of bedrooms above ground.
- **GarageCars**: Size of the garage in car capacity.
- **SalePrice**: The target variable representing house price.

## 2. Key Findings from Exploratory Data Analysis (EDA)

### Missing Values Handling

- Missing values in numerical columns were imputed using the **median** strategy to maintain consistency.
- Categorical features were encoded using **One-Hot Encoding** to ensure they are usable in the model.

### Outlier Detection

- Outliers were identified using **box plots** and **histograms**, particularly in `GrLivArea` and `SalePrice`.
- Log transformations were considered for highly skewed distributions.

### Correlation Analysis

- **Strongly correlated features with SalePrice**:
  - `OverallQual` (0.79 correlation)
  - `GrLivArea` (0.71 correlation)
  - `GarageCars` (0.64 correlation)
- Features with **low or no correlation** were dropped to improve model efficiency.

## 3. Model Selection & Evaluation

Three machine learning models were trained:

| Model             | R-squared Score |
| ----------------- | --------------- |
| Linear Regression | 0.74            |
| Random Forest     | 0.89            |
| XGBoost           | 0.91            |

The **XGBoost model** performed the best with an **R-squared score of 0.91**.

### Evaluation Metrics

For the best-performing XGBoost model:

- **Mean Absolute Error (MAE)**: 21,500
- **Mean Squared Error (MSE)**: 1,150,000,000
- **R-squared Score**: 0.91

## 4. Feature Importance Analysis

The top 5 important features in predicting house prices were:

1. **OverallQual**
2. **GrLivArea**
3. **GarageCars**
4. **TotalBsmtSF**
5. **YearBuilt**

These features contributed the most to predicting house prices and should be prioritized in future models.

## 5. Final Insights & Recommendations

- The **XGBoost model** is the most effective and should be used for price prediction.
- More **data preprocessing** (handling outliers and feature scaling) could improve accuracy.
- Additional features such as **neighborhood quality, school ratings, and crime rates** could further enhance the model’s performance.

## 6. Deployment (Optional Bonus Task)

A **Streamlit web application** was developed to allow users to input house attributes and get a predicted price. The model is saved using **joblib** and loaded for real-time predictions.
