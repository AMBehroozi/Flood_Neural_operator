import torch
import torch.nn as nn
import torch.nn.functional as F

class LightMagnifier(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        
        # Branch A: Water Depth (Extracts flow features)
        self.water_stream = nn.Sequential(
            nn.Conv3d(1, width//2, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Branch B: Bed Topography (Extracts terrain constraints)
        self.bed_stream = nn.Sequential(
            nn.Conv3d(1, width//2, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Lightweight Bottleneck (No complex Dense Blocks)
        # Just 3 layers to learn the interaction between terrain and water
        self.refinement = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Projection back to a single water depth channel
        self.projection = nn.Conv3d(width, 1, kernel_size=1)

    def forward(self, x):
        # x shape: [nb, 2, Pf, Pf, nt]
        # Store original interpolated water for Global Skip Connection
        base_water = x[:, 0:1, :, :, :] 
        bed = x[:, 1:2, :, :, :]
        
        # 1. Separate feature extraction
        w_feat = self.water_stream(base_water)
        b_feat = self.bed_stream(bed)
        
        # 2. Fusion
        combined = torch.cat([w_feat, b_feat], dim=1)
        
        # 3. Light Refinement
        x = self.refinement(combined)
        
        # 4. Global Skip Connection
        # The model predicts the DELTA (difference), and we add it to the base
        residual = self.projection(x)
        
        return base_water + residual