"""
Training script for CNN Image Autoencoder
Run this script to train a CNN model for image compression.

Usage:
    python train_cnn.py --images_dir path/to/training/images --epochs 50

Options:
    --images_dir: Directory containing training images (required)
    --epochs: Number of training epochs (default: 50)
    --batch_size: Batch size for training (default: 8)
    --latent_dim: Latent dimension - lower = more compression (default: 64)
    --model_type: 'standard' or 'lightweight' (default: standard)
    --learning_rate: Learning rate for optimizer (default: 0.001)
    --output_name: Name for the saved model (optional, auto-generated if not provided)
"""

import argparse
import os
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'svd_site.settings')
import django
django.setup()

from compressor.ml_utils import train_autoencoder


def main():
    parser = argparse.ArgumentParser(description='Train CNN Autoencoder for Image Compression')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing training images')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Latent dimension (default: 64)')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'lightweight'],
                        help='Model type: standard or lightweight (default: standard)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Output model name (optional)')
    
    args = parser.parse_args()
    
    # Validate images directory
    if not os.path.isdir(args.images_dir):
        print(f"Error: Directory '{args.images_dir}' does not exist.")
        sys.exit(1)
    
    # Count images
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_count = sum(1 for f in os.listdir(args.images_dir) 
                      if f.lower().endswith(image_extensions))
    
    if image_count == 0:
        print(f"Error: No images found in '{args.images_dir}'")
        print(f"Supported formats: {', '.join(image_extensions)}")
        sys.exit(1)
    
    print(f"Found {image_count} training images")
    
    # Create output directory
    models_dir = os.path.join(os.path.dirname(__file__), 'compressor', 'trained_models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate output path
    if args.output_name:
        model_filename = args.output_name if args.output_name.endswith('.pth') else f"{args.output_name}.pth"
    else:
        model_filename = f'autoencoder_{args.model_type}_ld{args.latent_dim}.pth'
    
    model_path = os.path.join(models_dir, model_filename)
    
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Images Directory: {args.images_dir}")
    print(f"Model Type: {args.model_type}")
    print(f"Latent Dimension: {args.latent_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Output Path: {model_path}")
    print("="*60 + "\n")
    
    # Train the model
    try:
        model = train_autoencoder(
            train_images_dir=args.images_dir,
            model_save_path=model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            model_type=args.model_type,
            learning_rate=args.learning_rate
        )
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Model saved to: {model_path}")
        print(f"\nYou can now use this model in the web interface by selecting:")
        print(f"  - Method: CNN Autoencoder")
        print(f"  - Model Type: {args.model_type}")
        print(f"  - Latent Dimension: {args.latent_dim}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
