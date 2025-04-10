 import pandas as pd
from datetime import datetime, timedelta
from calendar import monthrange
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv("data.csv")

# Convert date columns
df["Planned End Date"] = pd.to_datetime(df["Planned End Date"], errors='coerce')
df = df.dropna(subset=["Planned End Date", "Lead Time To Deploy (Days)"])

# Feature engineering
df["planned_month"] = df["Planned End Date"].dt.month
df["planned_day"] = df["Planned End Date"].dt.day

# Select features and target
X = df[["Service Line", "Organisation Level 8", "Assignment Group", "CR Category", "planned_month", "planned_day"]]
y = df["Lead Time To Deploy (Days)"]

# Define categorical columns
categorical_cols = ["Service Line", "Organisation Level 8", "Assignment Group", "CR Category"]

# Preprocessing and model pipeline
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split and fit model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Generate prediction for each day in next month
today = datetime.today()
year = today.year
next_month = today.month + 1 if today.month < 12 else 1
year = year if today.month < 12 else year + 1
days_in_next_month = monthrange(year, next_month)[1]

# Build prediction DataFrame
predict_dates = [datetime(year, next_month, day) for day in range(1, days_in_next_month + 1)]
sample_inputs = pd.DataFrame([{
    "Service Line": "Wholesale Technology",
    "Organisation Level 8": "WS Global Payment Solutions",
    "Assignment Group": "HTSA-PCD-ADV",
    "CR Category": "Software",
    "planned_month": date.month,
    "planned_day": date.day
} for date in predict_dates])

# Predict
predictions = pipeline.predict(sample_inputs)
result_df = pd.DataFrame({
    "Date": predict_dates,
    "Predicted Lead Time (Days)": predictions
})

# Show output
print(result_df)

# Optional: Save to CSV
result_df.to_csv("Predicted_Lead_Time_Next_Month.csv", index=False)

