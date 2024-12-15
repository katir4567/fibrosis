import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import cv2  # For loading images during visualization

# Define the main directory containing the image dataset
main_directory = r"data/images"

def prepare_metadata(img_dir):
    """
    Prepares metadata by traversing the image directory and collecting paths and labels.

    Args:
        img_dir (str): Root directory containing subfolders of classes (e.g., 'normal', 'fibrosis', 'pneumonia').

    Returns:
        pd.DataFrame: Metadata with columns 'Path' (image paths) and 'Label' (class labels).
    """
    data = []
    classes = ['Normal', 'Fibrosis', 'Pneumonia']
    for class_label, class_name in enumerate(classes):
        class_dir = os.path.join(img_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                data.append({'Path': os.path.join(class_dir, filename), 'Label': class_label})
    metadata = pd.DataFrame(data)
    metadata.to_csv("dataset.csv", index=False)
    return metadata

def plot_class_distribution(metadata):
    """
    Plots the distribution of classes in the dataset.

    Args:
        metadata (pd.DataFrame): Metadata with a 'Label' column representing class labels.
    """
    class_counts = metadata['Label'].value_counts().sort_index()
    classes = ['Normal', 'Fibrosis', 'Pneumonia']
    class_counts.index = classes
    class_counts.plot(kind='bar', color=['#66B2FF', '#FF9999', '#99FF99'])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

def visualize_samples(metadata):
    """
    Visualizes a sample image from each class in the dataset.

    Args:
        metadata (pd.DataFrame): Metadata containing image paths and labels.
    """
    classes = ['Normal', 'Fibrosis', 'Pneumonia']
    sample_data = metadata.groupby('Label').first().reset_index()
    plt.figure(figsize=(15, 5))

    for i, row in sample_data.iterrows():
        img_path = row['Path']
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(1, len(sample_data), i + 1)
        plt.imshow(img)
        plt.title(f"Class: {classes[row['Label']]}")
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    # Prepare metadata for the given directory
    metadata = prepare_metadata(main_directory)

    # Split and save train/test data
    train_val_data, test_data = train_test_split(metadata, test_size=0.2, stratify=metadata['Label'], random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(train_val_data, train_val_data['Label'])):
        train_data = train_val_data.iloc[train_idx]
        val_data = train_val_data.iloc[val_idx]

        train_data.to_csv(f"data/CSVs/fold_{fold_num}_train.csv", index=False)
        val_data.to_csv(f"data/CSVs/fold_{fold_num}_val.csv", index=False)

    # Visualize class distribution
    plot_class_distribution(metadata)

    # Visualize sample images
    visualize_samples(metadata)
