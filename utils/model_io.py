import joblib
import os

def save_model(model, filename, filepath):

    if filepath is None:
        filepath = os.getcwd()

    os.makedirs(filepath, exist_ok=True)

    # full path
    full_path = os.path.join(filepath, filename)

    # save
    joblib.dump(model, full_path)
    print(f"Model saved to {full_path}")

def load_model(filename, filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model found at {filepath}")

    full_path = os.path.join(filepath, filename)
    model = joblib.load(full_path)
    print(f"Model loaded from {full_path}")
    return model