from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Running script...")

X, y = load_iris(return_X_y=True)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "models/model.pkl")

print("Model trained successfully")
