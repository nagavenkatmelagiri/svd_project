import numpy as np
from PIL import Image
import math

def svd_compress_channel(channel, k):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    S = S[:k]
    U = U[:, :k]
    Vt = Vt[:k, :]
    return (U * S) @ Vt

def compress_image_svd(in_path, out_path, k):
    img = Image.open(in_path).convert('RGB')
    arr = np.array(img).astype(float)
    r = svd_compress_channel(arr[:, :, 0], k)
    g = svd_compress_channel(arr[:, :, 1], k)
    b = svd_compress_channel(arr[:, :, 2], k)
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)
    comp = np.stack([r, g, b], axis=2).astype(np.uint8)
    Image.fromarray(comp).save(out_path)
    return comp

def psnr(orig_arr, recon_arr):
    orig = orig_arr.astype(float)
    recon = recon_arr.astype(float)
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
