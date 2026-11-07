import os
import pandas as pd
from sklearn.model_selection import train_test_split


METADATA_CSV = r"D:\Paper1_FloodGeneralization\metadata_hand_labeled_s1.csv"
OUTPUT_DIR = r"D:\Paper1_FloodGeneralization\splits"

RANDOM_STATE = 42

# Cross-event test event
TEST_EVENT = "Sri-Lanka"


def save_split(split_dir, train_df, val_df, test_df):
    os.makedirs(split_dir, exist_ok=True)

    train_df.to_csv(os.path.join(split_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(split_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(split_dir, "test.csv"), index=False)


def create_random_split(df):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df["event"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=temp_df["event"]
    )

    split_dir = os.path.join(OUTPUT_DIR, "random_split")
    save_split(split_dir, train_df, val_df, test_df)

    print("\nRandom split created")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))
    print("\nRandom test event distribution:")
    print(test_df["event"].value_counts())


def create_cross_event_split(df):
    available_events = sorted(df["event"].unique())

    if TEST_EVENT not in available_events:
        raise ValueError(f"{TEST_EVENT} not found. Available events: {available_events}")

    test_df = df[df["event"] == TEST_EVENT].copy()
    remaining_df = df[df["event"] != TEST_EVENT].copy()

    train_df, val_df = train_test_split(
        remaining_df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=remaining_df["event"]
    )

    split_dir = os.path.join(OUTPUT_DIR, "cross_event_split")
    save_split(split_dir, train_df, val_df, test_df)

    print("\nCross-event split created")
    print("Held-out test event:", TEST_EVENT)
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    print("\nCross-event train event distribution:")
    print(train_df["event"].value_counts())

    print("\nCross-event validation event distribution:")
    print(val_df["event"].value_counts())

    print("\nCross-event test event distribution:")
    print(test_df["event"].value_counts())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(METADATA_CSV)

    print("Total samples:", len(df))
    print("\nEvents:")
    print(df["event"].value_counts())

    create_random_split(df)
    create_cross_event_split(df)

    print("\nAll split files saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()