import torch
import torch.nn as nn
import numpy as np

def normalize(x: torch.Tensor, dim=None, eps=1e-4) -> torch.Tensor:
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32) # type: torch.Tensor
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

class MPFourier(nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32) # type: torch.Tensor
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

class MPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x: torch.Tensor, gain=1) -> torch.Tensor:
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # type: torch.Tensor # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return F.conv2d(x, w, padding=(w.shape[-1]//2,))

class AdaptiveLossWeightMLP(nn.Module):
    def __init__(
            self,
            noise_scheduler,
            logvar_channels=128,
        ):
        super().__init__()
        self.a_bar_mean = noise_scheduler.alphas_cumprod.mean()
        self.a_bar_std = noise_scheduler.alphas_cumprod.std()
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

    def forward(self, a_bar: torch.Tensor):
        c_noise = a_bar.sub_(self.a_bar_mean).div_(self.a_bar_std)
        return self.logvar_linear(self.logvar_fourier(c_noise)).squeeze()

# lambda_weights is for setting an initial timestep weighting, if unsure use torch.ones((1000,)).  
# Recommend avoiding anything that goes to zero.  Model should converge to the same place regardless
# of what this is, but it will never be able to change away from a zero and will struggle with near-zero.

# loss_scaled is not to be used for backwards passes, it is however a metric I find useful to track.
# It will always converge to 1 when loss weights are close to their final position.

# Note that your loss will almost certainly converge to a negative value.  This is completely normal!
# Do not panic!  You might want to track MSE loss separately.

# alphas_cumprod is a departure from the EDM2 paper's sigma and does not have to be the input to the model.
# The only important things are that the input value is monotonically increasing or decreasing and that it is
# normalized on expectation (which is already done for this)

lambda_weights = torch.ones((1000,))

def loss_weighting(loss, timesteps, adaptive_loss_model, noise_scheduler, device):
    timesteps = timesteps.to("cpu")
    adaptive_loss_weights = adaptive_loss_model(noise_scheduler.alphas_cumprod[timesteps].to(device))
    loss_scaled = loss * (lambda_weights[timesteps].to(device) / torch.exp(adaptive_loss_weights)) # type: torch.Tensor
    loss = loss_scaled + adaptive_loss_weights # type: torch.Tensor

    return loss, loss_scaled

# Train this using a learning rate around 0.005 (yes this actually works), using Adam.  Do not use weight decay.
# You probably do also want some sort of decay schedule as well, just make sure it doesn't decay too early.
# If you feel adventurous, Aaron Defazio's schedule free Adam seems to work fairly well on this.
