import torch
import torch.nn as nn
import torch.nn.functional as F

class LightMagnifier(nn.Module):
    def __init__(self, in_channels=3, width=32, num_refinement_layers=3):
        """
        Args:
            in_channels (int): Now 3 (Interpolated, Bed, Bathtub).
            width (int): Hidden dimension (model capacity).
            num_refinement_layers (int): How many layers in the bottleneck.
        """
        super().__init__()
        self.in_channels = in_channels
        
        # 1. Unified Feature Extraction (More efficient than separate branches)
        # We process all input channels together to learn spatial correlations immediately
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, width, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # 2. Flexible Refinement Bottleneck
        # Allows you to make the model deeper or shallower based on PSU cluster limits
        refinement_list = []
        for _ in range(num_refinement_layers):
            refinement_list.append(nn.Conv3d(width, width, kernel_size=3, padding=1))
            refinement_list.append(nn.GELU())
        self.refinement = nn.Sequential(*refinement_list)
        
        # 3. Final Projection to Residual
        self.projection = nn.Conv3d(width, 1, kernel_size=1)

    def forward(self, x):
        """
        x shape: [nb, in_channels, Pf, Pf, nt]
        """
        # --- RESIDUAL ANCHOR SELECTION ---
        # If we have 3 channels, the "Bathtub" (channel 2) is our best physical guess.
        # If we only have 2, the "Interpolated" (channel 0) is our guess.
        if self.in_channels == 3:
            physical_baseline = x[:, 2:3, :, :, :] # Channel 2: Bathtub
        else:
            physical_baseline = x[:, 0:1, :, :, :] # Channel 0: Interpolated Coarse
        
        # 1. Extract Features from all augmented inputs
        feat = self.feature_extractor(x)
        
        # 2. Deep Refinement
        feat = self.refinement(feat)
        
        # 3. Predict the Correction (Residual)
        residual = self.projection(feat)
        
        # 4. FINAL SUM: Physics + Neural Correction
        return physical_baseline + residual