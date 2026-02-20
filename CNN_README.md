# CNN Autoencoder for Image Compression

This project now includes a **CNN-based deep learning approach** for image compression, in addition to the classical SVD method.

## 🚀 Features

- **Two Compression Methods**:
  - **SVD (Singular Value Decomposition)**: Classical linear algebra approach
  - **CNN Autoencoder**: Deep learning approach with neural networks

- **CNN Model Options**:
  - **Standard Model**: Better quality, more parameters
  - **Lightweight Model**: Faster inference, fewer parameters

- **Configurable Compression**: Adjust latent dimension to control compression ratio

## 📦 Installation

PyTorch has already been installed. All dependencies are ready to use.

## 🎯 Quick Start

### Using the Web Interface (Untrained Model)

1. Start the server:
   ```bash
   python manage.py runserver
   ```

2. Open browser to `http://127.0.0.1:8000/`

3. Select **CNN Autoencoder** from the compression method dropdown

4. Upload an image and compress

**Note**: Without training, the CNN will produce poor results. The model needs to be trained first!

### Training a CNN Model

To get good results with CNN compression, you need to train a model:

#### Step 1: Prepare Training Data

Collect a dataset of images (100+ recommended):
- Place all images in a single folder
- Supported formats: PNG, JPG, JPEG, BMP
- Images will be resized to 256x256 during training

Example:
```
my_training_images/
├── image1.jpg
├── image2.jpg
├── image3.png
└── ...
```

#### Step 2: Train the Model

Run the training script:

```bash
# Basic training (standard model, latent_dim=64)
python train_cnn.py --images_dir path/to/training/images --epochs 50

# Lightweight model with higher compression
python train_cnn.py --images_dir path/to/training/images --epochs 50 --model_type lightweight --latent_dim 32

# Advanced training with custom parameters
python train_cnn.py --images_dir path/to/training/images \
    --epochs 100 \
    --batch_size 16 \
    --latent_dim 64 \
    --model_type standard \
    --learning_rate 0.001
```

Training parameters:
- `--images_dir`: Path to folder with training images (required)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 8, increase if you have GPU)
- `--latent_dim`: Compression strength (lower = more compression, default: 64)
- `--model_type`: `standard` or `lightweight` (default: standard)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)

#### Step 3: Use the Trained Model

After training completes, the model is automatically saved to `compressor/trained_models/`.

Use it in the web interface:
1. Select **CNN Autoencoder** as compression method
2. Choose the same **model type** and **latent dimension** you used for training
3. Upload and compress images

## 🧪 Understanding Latent Dimension

The **latent dimension** controls compression strength:

- **latent_dim = 16**: Very high compression (smaller files, lower quality)
- **latent_dim = 32**: High compression (good for web images)
- **latent_dim = 64**: Moderate compression (balanced, recommended)
- **latent_dim = 128**: Low compression (better quality, larger files)
- **latent_dim = 256**: Minimal compression (best quality)

## 📊 SVD vs CNN Comparison

| Feature | SVD | CNN Autoencoder |
|---------|-----|-----------------|
| **Setup** | Ready to use | Requires training |
| **Speed** | Fast | Medium-Fast (depends on hardware) |
| **Quality** | Good | Excellent (when trained) |
| **Flexibility** | k parameter | Latent dimension |
| **Training** | Not needed | Required for good results |
| **Best for** | Quick compression | High-quality compression |

## 🔧 Advanced Usage

### Training on GPU (if available)

The code automatically detects and uses GPU if CUDA is available. To install GPU version of PyTorch:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Model Storage

Trained models are stored in:
```
compressor/trained_models/
├── autoencoder_standard_ld64.pth
├── autoencoder_lightweight_ld32.pth
└── ...
```

### Custom Model Names

Save model with custom name:
```bash
python train_cnn.py --images_dir ./images --output_name my_custom_model.pth
```

## 🐛 Troubleshooting

**Problem**: CNN compression produces poor results

**Solution**: The model needs to be trained first! Run the training script with your own image dataset.

---

**Problem**: "PyTorch not installed" error

**Solution**: Install PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

**Problem**: Training is very slow

**Solutions**:
- Use fewer epochs: `--epochs 20`
- Use lightweight model: `--model_type lightweight`
- Reduce batch size: `--batch_size 4`
- Install GPU version of PyTorch

---

**Problem**: Out of memory during training

**Solutions**:
- Reduce batch size: `--batch_size 4` or `--batch_size 2`
- Use lightweight model: `--model_type lightweight`
- Use smaller latent dimension: `--latent_dim 32`

## 📚 How It Works

### CNN Autoencoder Architecture

The autoencoder consists of two parts:

1. **Encoder**: Compresses the image to a low-dimensional latent representation
   - Uses strided convolutions to downsample
   - Extracts important features
   - Reduces to latent_dim channels

2. **Decoder**: Reconstructs the image from latent representation
   - Uses transposed convolutions to upsample
   - Recreates image details
   - Outputs RGB image

### Training Process

1. Load training images
2. Resize to 256x256
3. Feed through encoder → decoder
4. Compare output with original (MSE loss)
5. Update network weights via backpropagation
6. Repeat for all epochs

### Compression Process

1. Load input image
2. Pad dimensions to multiples of 16
3. Feed through trained encoder → decoder
4. Resize back to original dimensions
5. Save compressed image

## 🎓 Tips for Best Results

1. **Training Data**:
   - Use diverse images (100+ recommended)
   - Include images similar to what you'll compress
   - More data = better generalization

2. **Model Selection**:
   - Use **standard** model for best quality
   - Use **lightweight** for faster inference

3. **Compression Strength**:
   - Start with `latent_dim=64`
   - Decrease for more compression
   - Increase for better quality

4. **Training Duration**:
   - 50 epochs is usually sufficient
   - Monitor loss - stop if it plateaus
   - More epochs can improve quality

## 📖 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Autoencoders Explained](https://en.wikipedia.org/wiki/Autoencoder)
- [Image Compression with Neural Networks](https://arxiv.org/abs/1608.05148)

---

**Questions or issues?** Check the troubleshooting section or review the code in `compressor/ml_models.py` and `compressor/ml_utils.py`.
