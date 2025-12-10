# train_model.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from data_loader import load_data

def train_and_save_model():
    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "model.pkl")
    print("Model saved as model.pkl")

if __name__ == "__main__":
    train_and_save_model()
