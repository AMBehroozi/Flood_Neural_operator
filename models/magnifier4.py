import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        # Learns to generate scale and shift parameters from the conditioning features
        # We use a 1x1x1 conv to map the conditioning channels to gamma and beta
        self.gen_gamma = nn.Conv3d(width, width, kernel_size=1)
        self.gen_beta = nn.Conv3d(width, width, kernel_size=1)

    def forward(self, x, condition):
        # x is the feature map we want to modulate
        # condition is the feature map providing the context (e.g., Bed/Coarse info)
        gamma = self.gen_gamma(condition)
        beta = self.gen_beta(condition)
        return gamma * x + beta

class FiLMLightMagnifier(nn.Module):
    def __init__(self, in_channels=3, width=32, num_refinement_layers=3):
        super().__init__()
        self.in_channels = in_channels
        
        # Branch 1: Condition Extractor (Bed + Coarse)
        self.cond_extractor = nn.Sequential(
            nn.Conv3d(2, width, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Branch 2: Baseline Extractor (Bathtub)
        self.base_extractor = nn.Sequential(
            nn.Conv3d(1, width, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # FiLM Layer to mix them
        self.film = FiLMLayer(width)
        
        # Refinement Bottleneck
        refinement_list = []
        for _ in range(num_refinement_layers):
            refinement_list.append(nn.Conv3d(width, width, kernel_size=3, padding=1))
            refinement_list.append(nn.GELU())
        self.refinement = nn.Sequential(*refinement_list)
        
        self.projection = nn.Conv3d(width, 1, kernel_size=1)

    def forward(self, x):
        # x shape: [nb, 3, Pf, Pf, nt]
        # x[:, 0:2] -> Coarse + Bed (Conditions)
        # x[:, 2:3] -> Bathtub (Physical Baseline)
        
        cond_features = self.cond_extractor(x[:, 0:2])
        base_features = self.base_extractor(x[:, 2:3])
        
        # Apply FiLM: Condition modulates the Baseline features
        # 
        mixed_features = self.film(base_features, cond_features)
        
        # Deep Refinement
        feat = self.refinement(mixed_features)
        
        # Residual Connection to the original Bathtub channel
        residual = self.projection(feat)
        return x[:, 2:3] + residual