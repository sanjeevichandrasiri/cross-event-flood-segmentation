import os
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# PATHS
# =========================================================
METADATA_CSV = r"D:\Paper1_FloodGeneralization\metadata_hand_labeled_s1.csv"
OUTPUT_DIR = r"D:\Paper1_FloodGeneralization\splits\multi_cross_event_splits"

# Choose 3 held-out events
HELD_OUT_EVENTS = ["Sri-Lanka", "Ghana", "Bolivia"]

VAL_RATIO = 0.20
RANDOM_STATE = 42


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(METADATA_CSV)

    print("Total samples:", len(df))
    print("\nEvent distribution:")
    print(df["event"].value_counts())

    for held_out_event in HELD_OUT_EVENTS:
        print("\n" + "=" * 60)
        print("Creating split for held-out event:", held_out_event)

        test_df = df[df["event"] == held_out_event].copy()
        remaining_df = df[df["event"] != held_out_event].copy()

        train_df, val_df = train_test_split(
            remaining_df,
            test_size=VAL_RATIO,
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=remaining_df["event"]
        )

        split_dir = os.path.join(OUTPUT_DIR, f"heldout_{held_out_event}")
        os.makedirs(split_dir, exist_ok=True)

        train_path = os.path.join(split_dir, "train.csv")
        val_path = os.path.join(split_dir, "val.csv")
        test_path = os.path.join(split_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print("Train samples:", len(train_df))
        print("Validation samples:", len(val_df))
        print("Test samples:", len(test_df))

        print("\nTrain event distribution:")
        print(train_df["event"].value_counts())

        print("\nValidation event distribution:")
        print(val_df["event"].value_counts())

        print("\nTest event distribution:")
        print(test_df["event"].value_counts())

        print("\nSaved to:")
        print(split_dir)

    print("\nAll multi-event cross-event splits created successfully.")


if __name__ == "__main__":
    main()