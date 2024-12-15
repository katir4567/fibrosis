import os
import torch
from torch.utils.data import DataLoader
import pandas as pd

from dataset import lung_Xray_dataset
from model import MyModel
from trainer import train_model, evaluate_model
from args import get_args
from data_prep import plot_class_distribution, visualize_samples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    args = get_args()

    # Prepare metadata
    metadata = pd.read_csv("dataset.csv")
    plot_class_distribution(metadata)
    visualize_samples(metadata, "data/images")

    # 5-fold Cross-Validation
    for fold in range(5):
        print(f"Fold {fold+1}/5")

        train_set = pd.read_csv(f"data/CSVs/fold_{fold}_train.csv")
        val_set = pd.read_csv(f"data/CSVs/fold_{fold}_val.csv")

        train_dataset = lung_Xray_dataset(train_set, img_dir="data/images")
        val_dataset = lung_Xray_dataset(val_set, img_dir="data/images")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

        model = MyModel(backbone=args.backbone).to(device)

        # Train and track losses
        train_losses, val_losses = train_model(model, train_loader, val_loader, args)

        # Evaluate the model
        evaluate_model(model, val_loader)
