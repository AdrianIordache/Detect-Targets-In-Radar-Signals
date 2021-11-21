from utils import *

class RadarSignalsModel(nn.Module):
    def __init__(self, model_name, n_targets, pretrained = True):
        super().__init__()
        self.model      = timm.create_model(model_name, pretrained = pretrained)

        if 'efficientnet' in model_name:
            in_features           = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        else:
            in_features     = self.model.head.in_features
            self.model.head = nn.Identity()
        
        self.head = nn.Linear(in_features, n_targets)

    def forward(self, x, embeddings = False, debug = False):
        x = self.model(x)
        if embeddings: return x

        x = self.head(x)
        return x