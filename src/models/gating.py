"""Gating network that consumes pooled features + probs and outputs weights for base models."""
import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    def __init__(self, feat_dim_per_model=256, num_models=3, num_classes=6):
        super().__init__()
        # expected input: concat([proj_f1, proj_f2, proj_f3, probs_concat])
        proj_total = feat_dim_per_model * num_models
        probs_dim = num_models * num_classes
        in_dim = proj_total + probs_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_models),
            nn.Softmax(dim=1)
        )

    def forward(self, proj_feats, probs_concat):
        # proj_feats: tensor (B, num_models*proj_dim)
        # probs_concat: tensor (B, num_models*num_classes)
        x = torch.cat([proj_feats, probs_concat], dim=1)
        weights = self.net(x)
        return weights
