import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GeM(nn.Module):
    """
    Generalized Mean Pooling (Learnable Pooling)
    Parameter p: 1 = Average Pooling, Infinity = Max Pooling
    The model learns the best p for each feature map.
    """
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class CheXpertModel(nn.Module):
    def __init__(self, num_classes=14, backbone_name='densenet121', pretrained=True):
        super(CheXpertModel, self).__init__()
        
        # 1. BACKBONE
        if backbone_name == 'densenet121':
            # Load standard DenseNet
            base_model = models.densenet121(weights='DEFAULT' if pretrained else None)
            
            # Extract features (Everything except the classifier)
            self.features = base_model.features
            
            # Feature map channels (DenseNet121 outputs 1024 channels)
            self.num_ftrs = 1024
            
        elif backbone_name == 'efficientnet_b0':
            # Fallback for extreme low memory
            base_model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
            self.features = base_model.features
            self.num_ftrs = 1280
            
        # 2. NECK (Pooling)
        # Replacing standard GAP with GeM
        self.pool = GeM()
        
        # 3. HEAD (Classifier)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(self.num_ftrs, num_classes)
        
    def forward(self, x):
        # x: [Batch, 3, 224, 224]
        
        # 1. Extract Features
        # Output: [Batch, 1024, 7, 7] (for DenseNet)
        f = self.features(x)
        
        # 2. Pooling (Neck)
        # Output: [Batch, 1024, 1, 1]
        f = self.pool(f)
        
        # Flatten: [Batch, 1024]
        f = torch.flatten(f, 1)
        
        # Future Expansion Point:
        # If lateral features existed, we would concat here:
        # f = torch.cat([f_frontal, f_lateral], dim=1)
        
        # 3. Classification (Head)
        f = self.dropout(f)
        logits = self.classifier(f)
        
        return logits
