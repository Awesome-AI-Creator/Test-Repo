# test_model.py
import joblib
from sklearn.metrics import accuracy_score
from data_loader import load_data

def test_model():
    X_train, X_test, y_train, y_test = load_data()

    model = joblib.load("model.pkl")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    test_model()
