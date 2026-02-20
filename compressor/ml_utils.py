"""
Utilities for CNN-based image compression
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from .ml_models import get_model


def get_device():
    """Get the best available device (GPU if available, else CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pretrained_model(model_path, model_type='standard', latent_dim=64):
    """
    Load a pretrained model from disk.
    
    Args:
        model_path: Path to the saved model weights
        model_type: 'standard' or 'lightweight'
        latent_dim: Latent dimension used during training
    
    Returns:
        Loaded model in evaluation mode
    """
    device = get_device()
    model = get_model(model_type=model_type, latent_dim=latent_dim, device=device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using untrained model.")
    
    model.eval()
    return model


def compress_image_cnn(in_path, out_path, model=None, model_path=None, latent_dim=64, model_type='standard'):
    """
    Compress an image using CNN autoencoder.
    
    Args:
        in_path: Input image path
        out_path: Output image path
        model: Pre-loaded model (optional)
        model_path: Path to model weights (used if model is None)
        latent_dim: Compression level (lower = more compression)
        model_type: 'standard' or 'lightweight'
    
    Returns:
        Reconstructed image as numpy array
    """
    device = get_device()
    
    # Load or use provided model
    if model is None:
        if model_path and os.path.exists(model_path):
            model = load_pretrained_model(model_path, model_type, latent_dim)
        else:
            # Use untrained model (for demo purposes)
            model = get_model(model_type=model_type, latent_dim=latent_dim, device=device)
            print("Warning: Using untrained model. Results may be poor.")
    
    # Load and preprocess image
    img = Image.open(in_path).convert('RGB')
    original_size = img.size
    
    # Ensure dimensions are divisible by 16 (due to 4 pooling layers)
    w, h = img.size
    new_w = ((w + 15) // 16) * 16
    new_h = ((h + 15) // 16) * 16
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] and CHW format
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    
    # Compress and reconstruct
    with torch.no_grad():
        reconstructed = model(img_tensor)
    
    # Convert back to image
    reconstructed = reconstructed.squeeze(0).cpu()  # Remove batch dimension
    reconstructed = reconstructed.permute(1, 2, 0).numpy()  # CHW -> HWC
    reconstructed = (reconstructed * 255).astype(np.uint8)
    
    # Resize back to original size if needed
    out_img = Image.fromarray(reconstructed)
    if out_img.size != original_size:
        out_img = out_img.resize(original_size, Image.LANCZOS)
    
    # Save with appropriate quality settings
    ext = os.path.splitext(out_path)[1].lower()
    save_kwargs = {}
    if ext in ('.jpg', '.jpeg'):
        save_kwargs['quality'] = 90
        save_kwargs['optimize'] = True
    
    out_img.save(out_path, **save_kwargs)
    return np.array(out_img)


def train_autoencoder(train_images_dir, model_save_path, epochs=50, batch_size=8, 
                      latent_dim=64, model_type='standard', learning_rate=0.001):
    """
    Train an autoencoder on a directory of images.
    
    Args:
        train_images_dir: Directory containing training images
        model_save_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        latent_dim: Latent dimension (compression level)
        model_type: 'standard' or 'lightweight'
        learning_rate: Learning rate for optimizer
    
    Returns:
        Trained model
    """
    from torch.utils.data import Dataset, DataLoader
    
    device = get_device()
    print(f"Training on device: {device}")
    
    # Define dataset
    class ImageDataset(Dataset):
        def __init__(self, img_dir, transform=None):
            self.img_dir = img_dir
            self.transform = transform
            self.images = [f for f in os.listdir(img_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.images[idx])
            img = Image.open(img_path).convert('RGB')
            
            # Resize to fixed size for training
            img = img.resize((256, 256), Image.LANCZOS)
            
            if self.transform:
                img = self.transform(img)
            return img
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = ImageDataset(train_images_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"Found {len(dataset)} training images")
    
    # Initialize model
    model = get_model(model_type=model_type, latent_dim=latent_dim, device=device)
    model.train()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, images in enumerate(dataloader):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model


def get_compression_ratio(in_path, latent_dim=64, model_type='standard'):
    """
    Estimate compression ratio based on latent dimension.
    
    Args:
        in_path: Input image path
        latent_dim: Latent dimension
        model_type: 'standard' or 'lightweight'
    
    Returns:
        Estimated compression ratio
    """
    img = Image.open(in_path)
    w, h = img.size
    
    # Original size in pixels * 3 channels
    original_size = w * h * 3
    
    # Latent representation size (downsampled by 16x in standard, 8x in lightweight)
    if model_type == 'standard':
        compressed_w = (w + 15) // 16
        compressed_h = (h + 15) // 16
    else:
        compressed_w = (w + 7) // 8
        compressed_h = (h + 7) // 8
    
    compressed_size = compressed_w * compressed_h * latent_dim
    
    ratio = original_size / compressed_size
    return ratio
