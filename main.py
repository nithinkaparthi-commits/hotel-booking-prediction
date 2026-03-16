import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("hotel_bookings_edited.csv")

data = df[['lead_time','stays_in_weekend_nights','stays_in_week_nights',
           'adults','children','babies','adr','is_canceled']]

data = data.fillna(0)

X = data.drop('is_canceled', axis=1)
y = data['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()

model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved")