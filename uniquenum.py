import pandas as pd
from datetime import datetime, timedelta
from calendar import monthrange
from sklearn.ensemble import RandomForestRegressor

# Step 1: Read only the required columns
df = pd.read_csv("your_input.csv", usecols=["Planned End Date", "Unique Number"])

# Step 2: Convert Planned End Date
df["Planned End Date"] = pd.to_datetime(df["Planned End Date"], format="%d-%b-%y", errors='coerce')
df = df.dropna(subset=["Planned End Date", "Unique Number"])  # Drop bad dates or nulls

# Step 3: Feature engineering
df["day"] = df["Planned End Date"].dt.day
df["weekday"] = df["Planned End Date"].dt.weekday
df["month"] = df["Planned End Date"].dt.month

# Step 4: Train model
X = df[["day", "weekday", "month"]]
y = df["Unique Number"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Step 5: Generate future dates for next month
today = datetime.today()
next_month = today.month + 1 if today.month < 12 else 1
next_year = today.year if today.month < 12 else today.year + 1
days_in_month = monthrange(next_year, next_month)[1]

future_dates = pd.date_range(start=datetime(next_year, next_month, 1), periods=days_in_month)

future_df = pd.DataFrame({
    "Planned End Date": future_dates,
    "day": future_dates.day,
    "weekday": future_dates.weekday,
    "month": future_dates.month
})

# Step 6: Predict
future_df["Predicted Unique Number"] = model.predict(future_df[["day", "weekday", "month"]])

# Step 7: Output
print(future_df[["Planned End Date", "Predicted Unique Number"]])

# Optional: Save to CSV
future_df.to_csv("predicted_unique_number_next_month.csv", index=False)
