import os
import pandas as pd


def main():
    # Check training data
    assert os.path.exists("london_house_prices.csv"), "Training data CSV missing"

    # Check model artifacts
    assert os.path.exists("artifacts/model.joblib"), "Model artifact missing"
    assert os.path.exists("artifacts/metadata.json"), "Metadata missing"

    # Check predictions
    assert os.path.exists("predictions.csv"), "Predictions CSV missing"
    df = pd.read_csv("predictions.csv")
    assert "predicted_price" in df.columns, "predicted_price column missing in predictions"
    assert len(df) > 0, "Predictions file is empty"
    print("Smoke test passed: data, model, and predictions are present and valid.")


if __name__ == "__main__":
    main()
