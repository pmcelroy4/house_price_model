# model/predict.py

import pandas as pd
import joblib
import os

# 1. Load new data for prediction
new_data_path = "data/new_housing.csv"
new_df = pd.read_csv(new_data_path)

# 2. Load the trained model
model_path = "model_output/house_price_model.joblib"
model = joblib.load(model_path)

# 3. Make predictions
predictions = model.predict(new_df)

# 4. Add predictions as a new column
new_df["predicted_median_house_value"] = predictions

# 5. Save predictions to CSV
output_path = "predictions.csv"
new_df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")

# 6. Optionally print some summary stats
print(f"Predictions summary:")
print(new_df["predicted_median_house_value"].describe())