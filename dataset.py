import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class lung_Xray_dataset(Dataset):
    def __init__(self, dataset, img_dir='data/images'):
        """
        Initializes the dataset by storing the DataFrame and image directory path.
        
        Args:
            dataset (DataFrame): DataFrame containing the metadata (image filenames and labels).
            img_dir (str): Path to the folder containing the images (default is 'data/images').
        """
        self.dataset = dataset
        self.img_dir = img_dir
    
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Fetches the image and label for a specific index."""
        
        # Get image filename from the dataset (make sure it includes the subfolder)
        img_name = self.dataset.iloc[idx]['Path']  # Assuming 'Path' column contains the image name
        img_path = os.path.join(self.img_dir, img_name)  # Join with the image directory path

        # Read the image (ensure it's RGB format, if needed)
        img = cv2.imread(img_path)  # Read the image using OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if it's in BGR
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Get the label (target)
        target = self.dataset.iloc[idx]['KL']  # Assuming 'KL' column contains the class label (Fibrosis, Normal, Pneumonia)

        # Return image and label
        return {
            'img': img,
            'target': target
        }
