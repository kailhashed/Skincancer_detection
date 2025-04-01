import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold

from data_loader import create_data_loaders
from model import create_model, create_ensemble
from utils import save_model, load_model, plot_training_history, plot_confusion_matrix

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        train_loss: Average training loss
        train_acc: Training accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(train_loader), 100 * correct / total

def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        val_loss: Average validation loss
        val_acc: Validation accuracy
        all_preds: All predictions
        all_labels: All true labels
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(val_loader), 100 * correct / total, all_preds, all_labels

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir):
    """
    Train the model with early stopping and model checkpointing.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on
        save_dir: Directory to save model checkpoints
    """
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, optimizer, epoch, val_acc, os.path.join(save_dir, 'best_model.pth'))
            print(f'Model saved with validation accuracy: {val_acc:.2f}%')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix for best model
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth'))['model_state_dict'])
    _, _, all_preds, all_labels = validate(model, val_loader, criterion, device)
    plot_confusion_matrix(all_labels, all_preds, classes=val_loader.dataset.label_encoder.keys())

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Load data
    df = pd.read_csv('data/HAM10000_metadata.csv')
    
    # Create k-fold cross validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    df['fold'] = -1
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        df.loc[val_idx, 'fold'] = fold
    
    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    num_epochs = 50
    
    # Train models for each fold
    models = []
    for fold in range(5):
        print(f'\nTraining fold {fold + 1}/5')
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(df, 'data/HAM10000_images')
        
        # Create model
        model = create_model(device=device)
        
        # Initialize optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        
        # Train model
        train_model(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            num_epochs, device, 'models'
        )
        
        # Save model for ensemble
        models.append(model)
    
    # Create and save ensemble model
    ensemble = create_ensemble(models, device)
    torch.save(ensemble.state_dict(), 'models/ensemble_model.pth')
    print('\nEnsemble model saved')

if __name__ == '__main__':
    main() 