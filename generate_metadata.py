import os
import pandas as pd


DATA_ROOT = r"D:\Paper1_FloodGeneralization\data\sen1floods11\v1.1"
OUTPUT_CSV = r"D:\Paper1_FloodGeneralization\metadata_hand_labeled_s1.csv"


def get_event_name(sample_id):
    return sample_id.split("_")[0]


def normalize_id(file_path):
    name = os.path.basename(file_path)
    name = os.path.splitext(name)[0]

    # Source image suffixes
    name = name.replace("_S1Hand", "")
    name = name.replace("_S1Weak", "")
    name = name.replace("_S2Hand", "")
    name = name.replace("_S2Weak", "")

    # Label/mask suffixes
    name = name.replace("_LabelHand", "")
    name = name.replace("_label", "")
    name = name.replace("_Label", "")
    name = name.replace("_S1OtsuLabelWeak", "")
    name = name.replace("_S2IndexLabelWeak", "")

    return name


def find_tif_files(root):
    tif_files = []

    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            if file.lower().endswith(".tif"):
                tif_files.append(os.path.join(dirpath, file))

    return tif_files


def main():
    print("Scanning:", DATA_ROOT)

    all_tifs = find_tif_files(DATA_ROOT)
    print("Total tif files found:", len(all_tifs))

    # Use only hand-labeled Sentinel-1 images
    s1_files = [
        f for f in all_tifs
        if os.path.basename(f).endswith("_S1Hand.tif")
    ]

    # Use only hand-labeled masks
    label_files = [
        f for f in all_tifs
        if os.path.basename(f).endswith("_LabelHand.tif")
    ]

    print("S1 hand-labeled image files found:", len(s1_files))
    print("Hand-labeled mask files found:", len(label_files))

    image_dict = {normalize_id(f): f for f in s1_files}
    label_dict = {normalize_id(f): f for f in label_files}

    rows = []

    for sample_id, image_path in image_dict.items():
        if sample_id in label_dict:
            rows.append({
                "id": sample_id,
                "image_path": image_path,
                "mask_path": label_dict[sample_id],
                "event": get_event_name(sample_id)
            })

    df = pd.DataFrame(rows)

    print("\nMatched image-mask pairs:", len(df))

    if len(df) == 0:
        print("\nNo matched pairs found. Showing sample normalized names:")
        print("\nImage IDs:")
        print(list(image_dict.keys())[:10])
        print("\nLabel IDs:")
        print(list(label_dict.keys())[:10])
        return

    print("\nEvents found:")
    print(df["event"].value_counts())

    df.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved metadata file:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()