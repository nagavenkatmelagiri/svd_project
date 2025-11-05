import numpy as np
from PIL import Image, ImageFilter
import os
import math

def svd_compress_channel(channel, k):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    S = S[:k]
    U = U[:, :k]
    Vt = Vt[:k, :]
    return (U * S) @ Vt

def compress_image_svd(in_path, out_path, k, scale=1.0, sharpen=False):
    img = Image.open(in_path).convert('RGB')
    # optionally resize before compression
    try:
        scale = float(scale)
    except Exception:
        scale = 1.0
    if scale <= 0 or scale > 1:
        scale = 1.0
    if scale != 1.0:
        w, h = img.size
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.resize(new_size, Image.LANCZOS)
    arr = np.array(img).astype(float)
    r = svd_compress_channel(arr[:, :, 0], k)
    g = svd_compress_channel(arr[:, :, 1], k)
    b = svd_compress_channel(arr[:, :, 2], k)
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)
    comp = np.stack([r, g, b], axis=2).astype(np.uint8)
    out_img = Image.fromarray(comp)

    # optional sharpening to improve perceived clarity after aggressive compression/resize
    if sharpen:
        # UnsharpMask: radius small, percent moderate, threshold low
        out_img = out_img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

    # use higher quality for JPEGs to preserve clarity; PNGs saved normally
    ext = os.path.splitext(out_path)[1].lower()
    save_kwargs = {}
    if ext in ('.jpg', '.jpeg'):
        save_kwargs['quality'] = 85
        save_kwargs['optimize'] = True

    out_img.save(out_path, **save_kwargs)
    return comp

def psnr(orig_arr, recon_arr):
    orig = orig_arr.astype(float)
    recon = recon_arr.astype(float)
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
