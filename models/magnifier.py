import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ------------------------------------------------------------------
# SpectralConv2d (UNCHANGED - Your exact implementation)
# ------------------------------------------------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights1_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    def compl_mul2d(self, input_real, input_imag, weights_real, weights_imag):
        return torch.einsum("bixy,ioxy->boxy", input_real, weights_real) - \
               torch.einsum("bixy,ioxy->boxy", input_imag, weights_imag), \
               torch.einsum("bixy,ioxy->boxy", input_real, weights_imag) + \
               torch.einsum("bixy,ioxy->boxy", input_imag, weights_real)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        x_ft_real, x_ft_imag = x_ft.real, x_ft.imag

        out_ft_real = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, device=x.device)
        out_ft_imag = torch.zeros_like(out_ft_real)

        out_ft_real[:, :, :self.modes1, :self.modes2], out_ft_imag[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft_real[:, :, :self.modes1, :self.modes2], x_ft_imag[:, :, :self.modes1, :self.modes2],
                             self.weights1_real, self.weights1_imag)

        out_ft_real[:, :, -self.modes1:, :self.modes2], out_ft_imag[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft_real[:, :, -self.modes1:, :self.modes2], x_ft_imag[:, :, -self.modes1:, :self.modes2],
                             self.weights2_real, self.weights2_imag)

        out_ft = torch.complex(out_ft_real, out_ft_imag)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


# ------------------------------------------------------------------
# FNOBlock2d with Dropout (2D spatial FFT only)
# ------------------------------------------------------------------
class FNOBlock2d(nn.Module):
    def __init__(self, channels, modes_x, modes_y, dropout=0.0):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes_x, modes_y)
        self.local = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Input shape: [nb, c, height, width, t]
        nb, c, height, width, t = x.shape

        # Reshape to treat time as batch: [nb*t, c, height, width]
        x_reshaped = x.permute(0, 4, 1, 2, 3).reshape(nb * t, c, height, width)

        # Apply pure 2D spectral conv (FFT on height/width only)
        x_spec = self.spectral(x_reshaped)
        
        # Apply local 2D conv
        x_local = self.local(x_reshaped)

        # Combine and activate
        x_combined = self.activation(x_spec + x_local)
        x_combined = self.dropout(x_combined)

        # Reshape back to 5D [nb, c, height, width, t]
        return x_combined.reshape(nb, t, c, height, width).permute(0, 2, 3, 4, 1)


# ------------------------------------------------------------------
# Residual Block with Dropout and Channel Expansion
# ------------------------------------------------------------------
class ResidualBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.expand = in_channels != out_channels
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        
        # Projection for residual if channels change
        if self.expand:
            self.residual_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x):
        residual = self.residual_proj(x)
        x = self.act(self.conv1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        return self.act(x + residual)


# ------------------------------------------------------------------
# Spatial Attention Module
# ------------------------------------------------------------------
class SpatialAttention3d(nn.Module):
    """Lightweight spatial attention to focus on important regions"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        attention_input = torch.cat([max_pool, avg_pool], dim=1)
        attention_map = self.sigmoid(self.conv(attention_input))
        return x * attention_map


# ------------------------------------------------------------------
# Multi-Scale Pyramid Pooling Module
# ------------------------------------------------------------------
class PyramidPooling3d(nn.Module):
    """Multi-scale context aggregation"""
    def __init__(self, in_channels, pool_sizes=[2, 4, 8]):
        super().__init__()
        self.pool_sizes = pool_sizes
        out_channels_per_pool = in_channels // len(pool_sizes)
        
        self.convs = nn.ModuleList([
            nn.Conv3d(in_channels, out_channels_per_pool, kernel_size=1)
            for _ in pool_sizes
        ])
        
        # Fusion: in_channels (original) + out_channels_per_pool * len(pool_sizes) (pooled)
        total_channels = in_channels + out_channels_per_pool * len(pool_sizes)
        self.fusion = nn.Conv3d(total_channels, in_channels, kernel_size=1)

    def forward(self, x):
        _, _, h, w, t = x.shape
        pyramid_features = [x]
        
        for pool_size, conv in zip(self.pool_sizes, self.convs):
            pooled = F.adaptive_avg_pool3d(x, (h // pool_size, w // pool_size, t // pool_size))
            pooled = conv(pooled)
            upsampled = F.interpolate(pooled, size=(h, w, t), mode='trilinear', align_corners=False)
            pyramid_features.append(upsampled)
        
        return self.fusion(torch.cat(pyramid_features, dim=1))


# ------------------------------------------------------------------
# Enhanced Magnifier Model with Progressive Channel EXPANSION
# ------------------------------------------------------------------
class MagnifierModel(nn.Module):
    """
    Enhanced 2D FNO Magnifier Model for Spatial Upscaling with:
    - 2D spatial FFT only (no temporal overhead)
    - Progressive channel EXPANSION (correct for upscaling)
    - Multi-scale pyramid pooling
    - Spatial attention
    - Skip connections between stages
    - Dropout regularization
    - Gradient checkpointing support
    
    INPUT:   [nb, 2, P_fine, P_fine, nt]
    OUTPUT:  [nb, 1, P_fine, P_fine, nt]
    """
    def __init__(
        self,
        in_channels=2,
        base_channels=48,           # Starting channels
        num_fno_blocks=5,           # Increased for better mixing
        fno_modes_x=16,             # Higher spatial modes
        fno_modes_y=16,
        num_refinement_blocks=5,    # Progressive expansion stages
        num_residual_per_block=3,
        channel_multipliers=[1.0, 1.33, 1.67, 2.0, 2.0],  # Progressive expansion
        dropout=0.1,
        use_attention=True,
        use_pyramid_pooling=True,
        use_gradient_checkpointing=False
    ):
        super().__init__()
        
        assert len(channel_multipliers) == num_refinement_blocks, \
            f"Need {num_refinement_blocks} multipliers, got {len(channel_multipliers)}"
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Initial lifting
        self.lifting = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # FNO mixer blocks with dropout (2D spatial FFT)
        self.fno_mixer = nn.ModuleList([
            FNOBlock2d(base_channels, fno_modes_x, fno_modes_y, dropout=dropout)
            for _ in range(num_fno_blocks)
        ])
        
        # Multi-scale pyramid pooling after FNO
        self.pyramid_pooling = PyramidPooling3d(base_channels) if use_pyramid_pooling else nn.Identity()
        
        # Spatial attention after FNO
        self.attention = SpatialAttention3d(base_channels) if use_attention else nn.Identity()
        
        # Progressive channel expansion in refinement stages
        self.refinement_stages = nn.ModuleList()
        current_channels = base_channels
        self.channel_counts = [base_channels]
        
        for i in range(num_refinement_blocks):
            next_channels = int(base_channels * channel_multipliers[i])
            
            stage_blocks = []
            for j in range(num_residual_per_block):
                if j == 0:
                    # First block in stage handles channel expansion
                    stage_blocks.append(ResidualBlock3d(current_channels, next_channels, dropout=dropout))
                    current_channels = next_channels
                else:
                    # Subsequent blocks maintain channels
                    stage_blocks.append(ResidualBlock3d(current_channels, current_channels, dropout=dropout))
            
            self.refinement_stages.append(nn.Sequential(*stage_blocks))
            self.channel_counts.append(current_channels)
        
        # Skip connection fusion layers (from FNO output to each refinement stage)
        self.skip_fusions = nn.ModuleList([
            nn.Conv3d(base_channels + ch, ch, kernel_size=1)
            for ch in self.channel_counts[1:]  # Skip first (no fusion needed)
        ])
        
        # Final projection from expanded channels to output
        self.final_projection = nn.Conv3d(current_channels, 1, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _checkpoint_forward(self, module, x):
        """Conditional gradient checkpointing"""
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(module, x, use_reentrant=False)
        else:
            return module(x)

    def forward(self, patch_input):
        # Lifting
        x = self.lifting(patch_input)
        
        # FNO mixer with checkpointing (2D spatial FFT only)
        for fno_block in self.fno_mixer:
            x = self._checkpoint_forward(fno_block, x)
        
        # Multi-scale context
        x = self.pyramid_pooling(x)
        
        # Spatial attention
        x = self.attention(x)
        
        # Store FNO output for skip connections
        fno_output = x
        
        # Progressive refinement with channel expansion and skip connections
        for i, stage in enumerate(self.refinement_stages):
            x = self._checkpoint_forward(stage, x)
            
            # Add skip connection from FNO output
            if i < len(self.skip_fusions):
                x = self.skip_fusions[i](torch.cat([x, fno_output], dim=1))
        
        # Final projection to single channel output
        refined_u = self.final_projection(x)
        
        return refined_u


# ------------------------------------------------------------------
# Multi-Scale Loss for Training
# ------------------------------------------------------------------
class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss that supervises at different resolutions
    Helps with training stability and detail preservation
    """
    def __init__(self, scales=[1, 2, 4], weights=[1.0, 0.5, 0.25]):
        super().__init__()
        self.scales = scales
        self.weights = weights
        self.base_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale == 1:
                loss = self.base_loss(pred, target)
            else:
                # Downsample both pred and target
                pred_down = F.avg_pool3d(pred, kernel_size=scale, stride=scale)
                target_down = F.avg_pool3d(target, kernel_size=scale, stride=scale)
                loss = self.base_loss(pred_down, target_down)
            
            total_loss += weight * loss
        
        return total_loss

