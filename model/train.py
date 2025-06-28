# model/train.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# 1. Load dataset
data_path = "data/housing.csv"
df = pd.read_csv(data_path)

# Basic info
print(df.info())
print(df.describe())

# Plot distributions of numeric features
numeric_features = [
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "median_house_value",
]

plt.figure(figsize=(12, 8))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f"Distribution of {feature}")
plt.tight_layout()
plt.show()

# Plot correlations as heatmap
plt.figure(figsize=(8, 6))
corr = df[numeric_features].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Boxplot of median_house_value by ocean_proximity (categorical)
plt.figure(figsize=(8, 6))
sns.boxplot(x="ocean_proximity", y="median_house_value", data=df)
plt.title("House Value by Ocean Proximity")
plt.xticks(rotation=45)
plt.show()

# 2. Define features and target
features = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity"  # categorical
]
target = "median_house_value"

X = df[features]
y = df[target]

# 3. Handle missing values (optional but likely needed)
X["total_bedrooms"] = X["total_bedrooms"].fillna(X["total_bedrooms"].median())

# 4. Preprocessing: one-hot encode 'ocean_proximity'
categorical_features = ["ocean_proximity"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="passthrough"  # leave all other (numerical) columns as-is
)

# 5. Build pipeline with preprocessor + model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Fit model
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE on test set: {rmse:.2f}")

# 9. Save model
output_dir = "model_output"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(model, os.path.join(output_dir, "house_price_model.joblib"))
print("Model saved to model_output/house_price_model.joblib")
