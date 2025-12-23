import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GeM(nn.Module):
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
            base_model = models.densenet121(weights='DEFAULT' if pretrained else None)
            self.features = base_model.features
            self.num_ftrs = 1024
            
        elif backbone_name == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
            self.features = base_model.features
            self.num_ftrs = 1280
            
        # 2. NECK
        self.pool = GeM()
        
        # 3. HEAD (Richer MLP)
        # Replaced single Linear layer with MLP
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        f = self.features(x)
        f = self.pool(f)
        f = torch.flatten(f, 1)
        logits = self.classifier(f)
        return logits
