import torch
import torch.nn as nn
import timm

class SkinCancerModel(nn.Module):
    def __init__(self, num_classes=7):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of output classes
        """
        super(SkinCancerModel, self).__init__()
        
        # Load EfficientNet-B4 pretrained model
        self.model = timm.create_model('efficientnet_b4', pretrained=True)
        
        # Modify the classifier
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

class EnsembleModel(nn.Module):
    def __init__(self, models):
        """
        Initialize ensemble model.
        
        Args:
            models (list): List of trained models
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, x):
        # Get predictions from all models
        predictions = torch.stack([model(x) for model in self.models])
        # Average the predictions
        return torch.mean(predictions, dim=0)

def create_model(num_classes=7, device=None):
    """
    Create and initialize the model.
    
    Args:
        num_classes (int): Number of output classes
        device: Device to move the model to
        
    Returns:
        model: Initialized model
    """
    model = SkinCancerModel(num_classes=num_classes)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    return model

def create_ensemble(models, device=None):
    """
    Create an ensemble of models.
    
    Args:
        models (list): List of trained models
        device: Device to move the ensemble to
        
    Returns:
        ensemble: Ensemble model
    """
    ensemble = EnsembleModel(models)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ensemble = ensemble.to(device)
    return ensemble 