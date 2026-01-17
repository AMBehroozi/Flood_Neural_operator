import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock3d(nn.Module):
    """
    A deeper block that allows features to bypass layers, 
    preserving the 'interpolated' coarse physics while adding 
    high-resolution details from the bed topography.
    """
    def __init__(self, nf=32, gc=16):
        super().__init__()
        # gc = growth channel
        self.conv1 = nn.Conv3d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv3d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv3d(nf + 2 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        return x3 * 0.2 + x  # Residual scaling for stability

class Deep3DMagnifier(nn.Module):
    def __init__(self, in_channels=2, width=48, num_blocks=4):
        super().__init__()
        
        # 1. Initial Feature Extraction
        self.lifting = nn.Conv3d(in_channels, width, kernel_size=3, padding=1)
        
        # 2. Deep Residual Body
        # Each block refines the spatial/temporal features
        self.res_blocks = nn.ModuleList([
            ResidualDenseBlock3d(nf=width, gc=width//2) 
            for _ in range(num_blocks)
        ])
        
        # 3. Global Skip Connection Fusion
        self.trunk_conv = nn.Conv3d(width, width, kernel_size=3, padding=1)
        
        # 4. Final Projection to 1 channel (Water Depth/Velocity)
        self.projection = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(width, 1, kernel_size=1)
        )

    def forward(self, x):
        # x: [nb, 2, P_fine, P_fine, nt]
        feat = self.lifting(x)
        
        # Run through the deep residual body
        res = feat
        for block in self.res_blocks:
            res = block(res)
        
        # Global skip connection: helps the model keep the base physics 
        # from the interpolation while adding the learned details
        res = self.trunk_conv(res)
        out = feat + res
        
        return self.projection(out)