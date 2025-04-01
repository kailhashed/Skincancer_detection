import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

class SkinCancerDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_train=True):
        """
        Initialize the dataset.
        
        Args:
            df (pd.DataFrame): DataFrame containing image metadata
            img_dir (str): Directory containing the images
            transform: Albumentations transform pipeline
            is_train (bool): Whether this is for training or validation
        """
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        
        # Create label encoder
        self.label_encoder = {label: idx for idx, label in enumerate(df['dx'].unique())}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image path
        img_name = self.df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get label
        label = self.label_encoder[self.df.iloc[idx]['dx']]
        
        # Apply transforms
        if self.transform and self.is_train:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

def get_transforms():
    """
    Get data augmentation transforms.
    
    Returns:
        train_transform: Transform pipeline for training
        val_transform: Transform pipeline for validation
    """
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

def create_data_loaders(df, img_dir, batch_size=32, num_workers=4):
    """
    Create training and validation data loaders.
    
    Args:
        df (pd.DataFrame): DataFrame containing image metadata
        img_dir (str): Directory containing the images
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    # Split data
    train_df = df[df['fold'] != 4]
    val_df = df[df['fold'] == 4]
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = SkinCancerDataset(train_df, img_dir, transform=train_transform, is_train=True)
    val_dataset = SkinCancerDataset(val_df, img_dir, transform=val_transform, is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 