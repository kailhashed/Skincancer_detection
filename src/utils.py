import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

def create_directories():
    """Create necessary directories for the project."""
    directories = ['data', 'models', 'notebooks', 'src']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_training_history(history):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def save_model(model, optimizer, epoch, val_acc, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, path)

def load_model(model, optimizer, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_acc']

def analyze_dataset(df):
    """Analyze and print dataset statistics."""
    print("\nDataset Analysis:")
    print("-" * 50)
    print(f"Total number of images: {len(df)}")
    print("\nClass distribution:")
    print(df['dx'].value_counts())
    print("\nAge distribution:")
    print(df['age'].describe())
    print("\nGender distribution:")
    print(df['sex'].value_counts())
    print("\nLocalization distribution:")
    print(df['localization'].value_counts())

def plot_class_distribution(df):
    """Plot class distribution."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='dx')
    plt.title('Distribution of Skin Cancer Types')
    plt.xticks(rotation=45)
    plt.show()

def plot_age_distribution(df):
    """Plot age distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='age', bins=30)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

def plot_gender_distribution(df):
    """Plot gender distribution."""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='sex')
    plt.title('Gender Distribution')
    plt.show()

def plot_localization_distribution(df):
    """Plot localization distribution."""
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='localization')
    plt.title('Distribution of Lesion Localization')
    plt.xticks(rotation=45)
    plt.show() 