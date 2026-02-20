"""
CNN Autoencoder for Image Compression
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for image compression.
    Uses strided convolutions for downsampling and transposed convolutions for upsampling.
    """
    def __init__(self, latent_dim=64):
        super(ImageAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)  # -> N x 64 x H/2 x W/2
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # -> N x 128 x H/4 x W/4
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # -> N x 256 x H/8 x W/8
        self.enc4 = nn.Conv2d(256, latent_dim, kernel_size=3, stride=2, padding=1)  # -> N x latent_dim x H/16 x W/16
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1)  # -> N x 256 x H/8 x W/8
        self.dec2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # -> N x 128 x H/4 x W/4
        self.dec3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # -> N x 64 x H/2 x W/2
        self.dec4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)  # -> N x 3 x H x W
        
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(64)
        
    def encode(self, x):
        """Encode image to latent representation"""
        x = F.relu(self.bn1(self.enc1(x)))
        x = F.relu(self.bn2(self.enc2(x)))
        x = F.relu(self.bn3(self.enc3(x)))
        x = F.relu(self.enc4(x))
        return x
    
    def decode(self, z):
        """Decode latent representation to image"""
        x = F.relu(self.bn4(self.dec1(z)))
        x = F.relu(self.bn5(self.dec2(x)))
        x = F.relu(self.bn6(self.dec3(x)))
        x = torch.sigmoid(self.dec4(x))  # Output in [0, 1] range
        return x
    
    def forward(self, x):
        """Full forward pass: encode then decode"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon


class LightweightAutoencoder(nn.Module):
    """
    Lightweight autoencoder for faster inference.
    Suitable for real-time compression with fewer parameters.
    """
    def __init__(self, latent_dim=32):
        super(LightweightAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1)
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        
    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        return x
    
    def decode(self, z):
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon


def get_model(model_type='standard', latent_dim=64, device='cpu'):
    """
    Factory function to get the appropriate model.
    
    Args:
        model_type: 'standard' or 'lightweight'
        latent_dim: Dimension of latent space (compression level)
        device: 'cpu' or 'cuda'
    
    Returns:
        Initialized model
    """
    if model_type == 'lightweight':
        model = LightweightAutoencoder(latent_dim=latent_dim)
    else:
        model = ImageAutoencoder(latent_dim=latent_dim)
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode by default
    return model
