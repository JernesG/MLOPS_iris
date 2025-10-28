import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

# Define paths
RAW_DATA_PATH = Path("data/raw/iris.csv")
PROCESSED_TRAIN_PATH = Path("data/processed/train.csv")
PROCESSED_TEST_PATH = Path("data/processed/test.csv")
ENCODER_PATH = Path("models/label_encoder.pkl")

# Define target column
TARGET_COLUMN = "Species"

def preprocess():

    #Load raw data
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded raw data with shape: {df.shape}")

    #Remove ID column if it exists
    for col in df.columns:
        if col.lower() == "id":
            df = df.drop(columns=[col])
            print(f"Removed ID column: {col}")

    #Encode target column
    le = LabelEncoder()
    df[TARGET_COLUMN] = le.fit_transform(df[TARGET_COLUMN])
    print(f"Encoded target column '{TARGET_COLUMN}'")
    print(f"Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Save encoder for reuse
    ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(le, ENCODER_PATH)
    print(f"Saved label encoder to: {ENCODER_PATH}")

    #Split into train/test sets
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)

    # Combine and save as train/test CSVs
    train_df = pd.concat([x_train, y_train], axis=1)
    test_df = pd.concat([x_test, y_test], axis=1)

    PROCESSED_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_PATH, index=False)

    print(f"Train data saved to: {PROCESSED_TRAIN_PATH}")
    print(f"Test data saved to: {PROCESSED_TEST_PATH}")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

if __name__ == "__main__":
    preprocess()
