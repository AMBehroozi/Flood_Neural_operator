import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialCrossAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        # Query: Derived from Water (Coarse + Bathtub)
        self.q_conv = nn.Conv3d(width, width, kernel_size=1)
        # Key: Derived from Terrain (Bed)
        self.k_conv = nn.Conv3d(width, width, kernel_size=1)
        # Value: Derived from Terrain (Bed)
        self.v_conv = nn.Conv3d(width, width, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv3d(width, width, kernel_size=1)

    def forward(self, water_feat, bed_feat):
        # Shapes: [nb, c, h, w, t]
        b, c, h, w, t = water_feat.shape
        
        # We treat each time step as a separate spatial relationship
        # Reshape to [b*t, c, h*w] for efficient matrix multiplication
        q = self.q_conv(water_feat).permute(0, 4, 1, 2, 3).reshape(-1, c, h*w)
        k = self.k_conv(bed_feat).permute(0, 4, 1, 2, 3).reshape(-1, c, h*w)
        v = self.v_conv(bed_feat).permute(0, 4, 1, 2, 3).reshape(-1, c, h*w)

        # 1. Compute Attention Map (Relate water to terrain)
        # energy: [b*t, h*w, h*w]
        energy = torch.bmm(q.transpose(1, 2), k) 
        attention = self.softmax(energy / (c ** 0.5))

        # 2. Apply Attention to Terrain Values
        # out: [b*t, c, h*w]
        out = torch.bmm(v, attention.transpose(1, 2))
        
        # 3. Reshape back to original 5D tensor
        out = out.reshape(b, t, c, h, w).permute(0, 2, 3, 4, 1)
        return self.out_conv(out)

class CrossAttentionMagnifier(nn.Module):
    def __init__(self, width=32, num_refinement_layers=3):
        super().__init__()
        
        # Branch 1: Terrain Feature Extractor (Channel 1)
        self.bed_net = nn.Sequential(
            nn.Conv3d(1, width, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Branch 2: Water Feature Extractor (Channel 0 and 2)
        self.water_net = nn.Sequential(
            nn.Conv3d(2, width, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # The Cross-Attention Engine
        self.cross_attn = SpatialCrossAttention(width)
        
        # Post-Attention Refinement
        layers = []
        for _ in range(num_refinement_layers):
            layers.append(nn.Conv3d(width, width, kernel_size=3, padding=1))
            layers.append(nn.GELU())
        self.refinement = nn.Sequential(*layers)
        
        self.projection = nn.Conv3d(width, 1, kernel_size=1)

    def forward(self, x):
        # x: [nb, 3, Pf, Pf, nt]
        # Channels: 0=Interp_Coarse, 1=Bed, 2=Bathtub
        
        # 1. Separate Feature Extraction
        water_info = x[:, [0, 2], ...] # Coarse + Bathtub
        bed_info = x[:, 1:2, ...]      # Topography
        
        f_water = self.water_net(water_info)
        f_bed = self.bed_net(bed_info)
        
        # 2. Cross-Attention: Water queries the Terrain
        # This tells the model "Where should the water go based on the land?"
        
        mixed = self.cross_attn(f_water, f_bed)
        
        # 3. Refine the mixed features
        feat = self.refinement(mixed + f_water) # Skip connection for stability
        
        # 4. Final Residual Projection
        residual = self.projection(feat)
        
        # Anchored to the Bathtub baseline
        return x[:, 2:3, ...] + residual