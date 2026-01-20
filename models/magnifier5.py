import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer with normalization and optional gating.
    
    Args:
        width: Number of channels in the main feature map
        norm_type: Type of normalization ('instance', 'group', 'layer')
        num_groups: Number of groups for GroupNorm (only used if norm_type='group')
        use_gating: Whether to use learned gating to blend conditioned/unconditioned features
    """
    def __init__(self, width, norm_type='instance', num_groups=8, use_gating=False):
        super().__init__()
        self.width = width
        self.use_gating = use_gating
        
        # Normalization layer
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm3d(width, affine=False)
        elif norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups, width, affine=False)
        elif norm_type == 'layer':
            # LayerNorm for 3D: normalize over C, D, H, W
            self.norm = nn.GroupNorm(1, width, affine=False)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
        
        # Conditioning pathway - generates gamma and beta
        self.gen_gamma = nn.Conv3d(width, width, kernel_size=1)
        self.gen_beta = nn.Conv3d(width, width, kernel_size=1)
        
        # Optional gating mechanism
        if use_gating:
            self.gen_gate = nn.Conv3d(width, width, kernel_size=1)
            # Initialize gate to produce ~0.5 (balanced blend initially)
            nn.init.constant_(self.gen_gate.bias, 0.0)
        
        # Initialize for identity behavior (gamma ≈ 1, beta ≈ 0)
        self._init_film_params()
    
    def _init_film_params(self):
        """Initialize gamma to produce ~1.0 and beta to produce ~0.0"""
        # Gamma: weight=0, bias=1 → output ≈ 1
        nn.init.zeros_(self.gen_gamma.weight)
        nn.init.constant_(self.gen_gamma.bias, 1.0)
        
        # Beta: all zeros → output ≈ 0
        nn.init.zeros_(self.gen_beta.weight)
        nn.init.zeros_(self.gen_beta.bias)
    
    def forward(self, x, condition):
        """
        Args:
            x: Main feature map (B, C, D, H, W)
            condition: Conditioning feature map (B, C, D, H, W)
                      Should have same spatial dims as x, or will be interpolated
        
        Returns:
            Modulated feature map (B, C, D, H, W)
        """
        # Handle spatial misalignment if needed
        if condition.shape[2:] != x.shape[2:]:
            condition = F.interpolate(
                condition, 
                size=x.shape[2:], 
                mode='trilinear', 
                align_corners=False
            )
        
        # Normalize the main features
        x_norm = self.norm(x)
        
        # Generate modulation parameters from conditioning
        gamma = self.gen_gamma(condition)  # scale
        beta = self.gen_beta(condition)     # shift
        
        # Apply FiLM: γ * x_norm + β
        x_modulated = gamma * x_norm + beta
        
        # Optional: blend with unconditioned features using learned gate
        if self.use_gating:
            gate = torch.sigmoid(self.gen_gate(condition))
            x_modulated = gate * x_modulated + (1 - gate) * x
        
        return x_modulated

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