# models/pinn_wave2d_advanced.py
#
# Advanced PINN architecture for the 2D wave equation:
# - Scales inputs (t, x, y) from [0, 1] to [-1, 1]
# - Applies Fourier positional encoding to capture oscillatory behavior
# - Uses residual blocks with LayerNorm for stability

import math
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """
    Fixed (non-trainable) Fourier positional encoding.

    Given input coords of shape (N, D), returns:
        [coords, sin(B coords), cos(B coords)]

    where B encodes multiple frequencies along each input dimension.
    """

    def __init__(self, in_dim: int, num_frequencies: int = 6):
        super().__init__()

        self.in_dim = in_dim
        self.num_frequencies = num_frequencies

        # Use powers of 2 for frequencies: 1, 2, 4, 8, ...
        freq_bands = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        # Shape: (num_frequencies,)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (N, in_dim), assumed to be roughly in [-1, 1].
        Returns: (N, in_dim + 2 * in_dim * num_frequencies)
        """
        # coords: (N, in_dim)
        # Expand dims to apply frequencies: (N, in_dim, num_frequencies)
        #   arg = 2π * coords[..., None] * freq_bands[None, None, :]
        arg = 2.0 * math.pi * coords.unsqueeze(-1) * self.freq_bands

        sin_feats = torch.sin(arg)
        cos_feats = torch.cos(arg)

        # Flatten last two dims into one feature dimension
        sin_flat = sin_feats.view(coords.shape[0], -1)
        cos_flat = cos_feats.view(coords.shape[0], -1)

        return torch.cat([coords, sin_flat, cos_flat], dim=-1)


class ResidualBlock(nn.Module):
    """
    Simple residual MLP block: Linear -> Tanh -> Linear,
    with a residual connection and LayerNorm.
    """

    def __init__(self, width: int):
        super().__init__()

        self.lin1 = nn.Linear(width, width)
        self.lin2 = nn.Linear(width, width)
        self.act = nn.Tanh()
        self.norm = nn.LayerNorm(width)

        # Optional: small initialization on the second layer
        nn.init.zeros_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.lin1(x))
        out = self.lin2(out)
        out = self.norm(x + out)  # residual + normalization
        out = self.act(out)
        return out


class PINNWave2DAdvanced(nn.Module):
    """
    Advanced PINN for 2D wave equation:
      u_tt = c^2 (u_xx + u_yy)

    Input:  (t, x, y) each as (N, 1) tensors.
    Output: u(t, x, y) as (N, 1).
    """

    def __init__(
        self,
        num_frequencies: int = 6,
        hidden_width: int = 128,
        num_res_blocks: int = 5,
    ):
        super().__init__()

        self.in_dim = 3  # (t, x, y)
        self.ff = FourierFeatures(in_dim=self.in_dim, num_frequencies=num_frequencies)

        # Compute encoded feature dimension:
        # original coords: in_dim
        # sin & cos: 2 * in_dim * num_frequencies
        ff_dim = self.in_dim + 2 * self.in_dim * num_frequencies

        self.input_layer = nn.Linear(ff_dim, hidden_width)
        self.input_act = nn.Tanh()

        # Residual trunk
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_width) for _ in range(num_res_blocks)]
        )

        # Output head
        self.output_layer = nn.Linear(hidden_width, 1)

        self._init_weights()

    def _init_weights(self):
        # Xavier-like initialization for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.output_layer:
                    # Smaller weights on output to avoid huge initial values
                    nn.init.uniform_(m.weight, -1e-3, 1e-3)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        t, x, y: each (N, 1)
        Returns: u: (N, 1)
        """
        # Concatenate into coords in [0, 1]
        coords = torch.cat([t, x, y], dim=1)  # (N, 3)

        # Scale from [0, 1] to [-1, 1]
        coords_scaled = 2.0 * coords - 1.0

        # Fourier positional encoding
        feats = self.ff(coords_scaled)  # (N, ff_dim)

        # Pass through residual MLP trunk
        h = self.input_act(self.input_layer(feats))

        for block in self.res_blocks:
            h = block(h)

        u = self.output_layer(h)
        return u
