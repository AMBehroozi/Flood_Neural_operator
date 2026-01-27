import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y, modes_t):
        super(SpectralConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes_x #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes_y
        self.modes3 = modes_t

        self.scale = (1 / (in_channels * out_channels))
        # Initialize real and imaginary parts separately
        self.weights1_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights1_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        
        self.weights2_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights2_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        
        self.weights3_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights3_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        
        self.weights4_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))
        self.weights4_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3))

    def compl_mul3d(self, input_real, input_imag, weights_real, weights_imag):
        # (batch, in_channel, x,y,t), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input_real, weights_real) - \
               torch.einsum("bixyz,ioxyz->boxyz", input_imag, weights_imag), \
               torch.einsum("bixyz,ioxyz->boxyz", input_real, weights_imag) + \
               torch.einsum("bixyz,ioxyz->boxyz", input_imag, weights_real)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])
        x_ft_real, x_ft_imag = x_ft.real, x_ft.imag
        
        # Multiply relevant Fourier modes
        out_ft_real = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, device=x.device)
        out_ft_imag = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, device=x.device)
        
        # First set of modes
        out_ft_real[:, :, :self.modes1, :self.modes2, :self.modes3], \
        out_ft_imag[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft_real[:, :, :self.modes1, :self.modes2, :self.modes3],
                           x_ft_imag[:, :, :self.modes1, :self.modes2, :self.modes3],
                           self.weights1_real, self.weights1_imag)
        
        # Second set of modes
        out_ft_real[:, :, -self.modes1:, :self.modes2, :self.modes3], \
        out_ft_imag[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft_real[:, :, -self.modes1:, :self.modes2, :self.modes3],
                           x_ft_imag[:, :, -self.modes1:, :self.modes2, :self.modes3],
                           self.weights2_real, self.weights2_imag)
        
        # Third set of modes
        out_ft_real[:, :, :self.modes1, -self.modes2:, :self.modes3], \
        out_ft_imag[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft_real[:, :, :self.modes1, -self.modes2:, :self.modes3],
                           x_ft_imag[:, :, :self.modes1, -self.modes2:, :self.modes3],
                           self.weights3_real, self.weights3_imag)
        
        # Fourth set of modes
        out_ft_real[:, :, -self.modes1:, -self.modes2:, :self.modes3], \
        out_ft_imag[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft_real[:, :, -self.modes1:, -self.modes2:, :self.modes3],
                           x_ft_imag[:, :, -self.modes1:, -self.modes2:, :self.modes3],
                           self.weights4_real, self.weights4_imag)
        
        # Combine real and imaginary parts
        out_ft = torch.complex(out_ft_real, out_ft_imag)
        
        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x



class UNetEncoder3d(nn.Module):
    def __init__(self, channels, num_layers, target_mx, target_my):
        """
        Flexible 3D Encoder that downsamples spatially to specific dimensions.
        
        Args:
            channels (int): Constant number of channels (C).
            num_layers (int): Number of convolutional downsampling steps.
            target_mx (int): Exact target size for dimension x.
            target_my (int): Exact target size for dimension y.
        """
        super(UNetEncoder3d, self).__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv3d(channels, channels, kernel_size=3, 
                              stride=(2, 2, 1), padding=1),
                    nn.GELU(),
                    nn.Conv3d(channels, channels, kernel_size=3, 
                              stride=1, padding=1),
                    nn.GELU()
                )
            )
            
        self.encoder = nn.Sequential(*layers)
        
        # This layer forces the output to be exactly (target_mx, target_my, nt)
        # It handles the math if (nx / 2^layers) doesn't perfectly match mx/my.
        self.final_pool = nn.AdaptiveAvgPool3d((target_mx, target_my, None))

    def forward(self, x):
        # x shape: [batch, C, nx, ny, nt]
        x = self.encoder(x)
        # final_pool preserves the last dimension (nt) automatically if passed None
        # but to be safe with all PyTorch versions, we pass the current nt.
        nt = x.shape[-1]
        x = nn.functional.adaptive_avg_pool3d(x, (self.final_pool.output_size[0], 
                                                 self.final_pool.output_size[1], 
                                                 nt))
        return x


class ResidualBlock3d(nn.Module):
    """Residual block for better feature refinement."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)  # Skip connection


class DeepDecoderBlock(nn.Module):
    """
    Deep decoder block with learned upsampling + multiple refinement layers.
    Much stronger than single conv refinement.
    """
    def __init__(self, channels, num_residual_blocks=2):
        super().__init__()
        
        # Learned upsampling
        self.conv_tp = nn.ConvTranspose3d(
            channels, channels, 
            kernel_size=3, 
            stride=(2, 2, 1), 
            padding=1,
            output_padding=(1, 1, 0)
        )
        
        # Multiple residual blocks for refinement
        self.residual_blocks = nn.ModuleList([
            ResidualBlock3d(channels) for _ in range(num_residual_blocks)
        ])
        
        # Final refinement
        self.final_conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x, target_x, target_y):
        # Learned upsampling
        x = self.conv_tp(x)
        
        # Crop/pad to exact target size
        curr_x, curr_y = x.shape[2], x.shape[3]
        if curr_x != target_x or curr_y != target_y:
            if curr_x >= target_x and curr_y >= target_y:
                x = x[:, :, :target_x, :target_y, :]
            else:
                pad_x = max(0, target_x - curr_x)
                pad_y = max(0, target_y - curr_y)
                x = F.pad(x, (0, 0, 0, pad_y, 0, pad_x))
        
        # Deep refinement with residual blocks
        for res_block in self.residual_blocks:
            x = res_block(x)
        
        # Final refinement
        x = self.act(self.final_conv(x))
        
        return x


class DeepDynamicUNetDecoder3d(nn.Module):
    """
    Deeper decoder with residual refinement blocks.
    
    Total depth per upsampling stage:
    - 1 ConvTranspose3d (upsampling)
    - 2 ResidualBlocks = 4 Conv3d layers
    - 1 final Conv3d
    = 6 learnable layers per block
    
    With 4 blocks: 24 total layers (vs 8 in original)
    """
    def __init__(self, channels, num_layers=4, num_residual_blocks=2):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            DeepDecoderBlock(channels, num_residual_blocks) 
            for _ in range(num_layers)
        ])

    def forward(self, x, final_nx, final_ny):
        curr_x, curr_y = x.shape[2], x.shape[3]
        
        # Calculate intermediate target sizes
        targets_x = []
        targets_y = []
        
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                targets_x.append(min(curr_x * (2 ** (i + 1)), final_nx))
                targets_y.append(min(curr_y * (2 ** (i + 1)), final_ny))
            else:
                targets_x.append(final_nx)
                targets_y.append(final_ny)
        
        # Apply decoder layers with deep refinement
        for i, layer in enumerate(self.layers):
            x = layer(x, targets_x[i], targets_y[i])
        
        return x


class FNO3d(nn.Module):
    def __init__(self, T_in, T_out, modes_x, modes_y, modes_t, width=20, encoder_kernel_size_x=128, encoder_kernel_size_y=128, encoder_num_layers=4):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """
        self.T_in = T_in
        self.T_out = T_out
        self.modes1 = modes_x
        self.modes2 = modes_y
        self.modes3 = modes_t
        self.encoder_kernel_size_x = encoder_kernel_size_x
        self.encoder_kernel_size_y = encoder_kernel_size_y
        self.encoder_num_layers = encoder_num_layers
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.encoder = UNetEncoder3d(channels=self.width, 
                                            num_layers=self.encoder_num_layers, 
                                            target_mx=self.encoder_kernel_size_x, 
                                            target_my=self.encoder_kernel_size_y)

        # # Initialize here instead of None
        # self.decoder = DeepDynamicUNetDecoder3d(
        #                                     channels=self.width,           # Your width
        #                                     num_layers=4,          # 4 upsampling stages (20→40→80→160→313)
        #                                     num_residual_blocks=1  # 2 residual blocks per stage
# )
        self.fc0 = nn.Linear(self.T_in + 3, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, forcing, u0, B):
        original_nx, original_ny = u0.shape[1], u0.shape[2]
        u0 = u0.unsqueeze(-2).repeat(1, 1, 1, self.T_out + 1, 1)
        forcing = forcing.unsqueeze(-1)
        grid = self.get_grid(u0.shape, u0.device)
        
        B = B.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, self.T_out + 1, 1)
        # torch.Size([2, 64, 64, 40, 10]) torch.Size([2, 64, 64, 40, 3])
        
        x = torch.cat((forcing, grid), dim=-1)  # shape [nb, nx, ny, T_out, C]
        x = self.fc0(x)                      # shape [nb, nx, ny, T_out, width]
        x = x.permute(0, 4, 1, 2, 3)         # shape [nb, width, nx, ny, T_out]
        
        if self.padding > 0:
            x = F.pad(x, [0,self.padding])  # pad the domain if input is non-periodic shape [B, C, W, H, T+self.padding]
        
        x = self.encoder(x)                   # shape [nb, width, mx, my, T_out]
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2                        # shape [B, C, W, H, T+self.padding]

        x = x[..., :-self.padding]          # shape [B, C, W, H, T]
        # x = self.decoder(x, original_nx, original_ny)                # shape [B, C, nx, ny, T]
        x = x.permute(0, 2, 3, 4, 1)        # shape [B, W, H, T, C]  
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x).squeeze(-1)
        return x[..., 1:]


    def get_grid(self, shape, device):
        """
        Generate normalized coordinate grids for 3D data.
        
        Args:
            shape: (batchsize, size_x, size_y, size_z)
            device: torch device
        
        Returns:
            grid: [batchsize, size_x, size_y, size_z, 3] containing (x, y, t) coordinates
        """
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        
        # Create 1D coordinate arrays
        x = torch.linspace(0, 1, size_x, device=device)
        y = torch.linspace(0, 1, size_y, device=device)
        t = torch.linspace(0, 1, size_z, device=device)
        
        # Create 2D meshgrid for x and y
        x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')  # [size_x, size_y]
        
        # Expand to full shape [batchsize, size_x, size_y, size_z, 1]
        x_grid = x_grid[None, :, :, None, None].expand(batchsize, -1, -1, size_z, 1)
        y_grid = y_grid[None, :, :, None, None].expand(batchsize, -1, -1, size_z, 1)
        t_grid = t[None, None, None, :, None].expand(batchsize, size_x, size_y, -1, 1)
        
        # Concatenate along last dimension
        return torch.cat([x_grid, y_grid, t_grid], dim=-1)

'''

if __name__ == "__main__":
    device = torch.device(f"cuda")  # Set device based on rank
    
    T_in=1
    T_out=88
    model = FNO3d(T_in=T_in, T_out=T_out, 
                modes_x=8, modes_y=8, modes_t=8, 
                width=20,
                encoder_kernel_size_x=100,
                encoder_kernel_size_y=50,
                encoder_num_layers=4)
    model = model.to(device)  # Move model to the correct GPU before wrapping with DDP
    nbatch, s0, s1 = 4, 313, 165
    u_in = torch.rand(nbatch, s0, s1, T_in).to(device)  # Move input tensors to the same device
    forcing = torch.rand(nbatch, s0, s1, T_in + T_out).to(device)
    parameters = torch.rand(nbatch, s0, s1).to(device)

    u_out = model(forcing, u_in, parameters)   # u_in:  [nb, nx, ny,  T_in] parameters: [nb, nx, ny]
    print(f"{u_out.shape}")           # u_out: [nb, nx, ny,  T_out] 



'''