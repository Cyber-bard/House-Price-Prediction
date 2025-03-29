import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Load dataset
file_path = "house_prices.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at {file_path}. Please ensure the file is in the correct directory.")

df = pd.read_csv(file_path)
print(df.head())

# Data Preprocessing
# Handling missing values
imputer = SimpleImputer(strategy='median')
numeric_features = df.select_dtypes(include=[np.number]).columns
df[numeric_features] = imputer.fit_transform(df[numeric_features])

# Encoding categorical variables
categorical_features = df.select_dtypes(include=["object"]).columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
categorical_encoded = encoder.fit_transform(df[categorical_features])
categorical_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_features))

# Combine numerical and categorical features
df = df.drop(columns=categorical_features)
df = pd.concat([df, categorical_df], axis=1)

# Feature Scaling
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Splitting data
X = df.drop(columns=["SalePrice"])  # Target variable
y = df["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
}

best_model = None
best_score = float('-inf')

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"{name}: R² Score = {score:.4f}")
    if score > best_score:
        best_score = score
        best_model = model

# Model Evaluation
y_pred = best_model.predict(X_test)
print("Best Model Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Save the best model
joblib.dump(best_model, "house_price_model.pkl")

# Feature Importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    feature_importance.plot(kind='bar', title='Feature Importance')
    plt.show()

# Deployment (Optional)
# Run: streamlit run app.py (if Streamlit is used)
import streamlit as st

def predict_price(input_features):
    model = joblib.load("house_price_model.pkl")
    return model.predict([input_features])

st.title("House Price Prediction App")
input_data = [st.number_input(f"{feature}") for feature in X.columns[:5]]  # Example using first 5 features
if st.button("Predict Price"):
    price = predict_price(input_data)
    st.write(f"Predicted House Price: ${price[0]:,.2f}")
