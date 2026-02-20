import numpy as np
from PIL import Image, ImageFilter
import os
import tempfile
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

    # Convert to YCbCr and apply SVD on luminance with higher rank to preserve perceived clarity
    ycbcr = img.convert('YCbCr')
    arr = np.array(ycbcr).astype(float)

    # Use a larger k for luminance (Y) to preserve detail, keep chroma at k
    k_y = int(min(200, max(1, k * 2)))
    k_c = int(max(1, k))

    y = svd_compress_channel(arr[:, :, 0], k_y)
    cb = svd_compress_channel(arr[:, :, 1], k_c)
    cr = svd_compress_channel(arr[:, :, 2], k_c)

    y = np.clip(y, 0, 255)
    cb = np.clip(cb, 0, 255)
    cr = np.clip(cr, 0, 255)

    comp = np.stack([y, cb, cr], axis=2).astype(np.uint8)

    # create YCbCr image and convert back to RGB
    out_img = Image.fromarray(comp, mode='YCbCr').convert('RGB')

    # optional sharpening to improve perceived clarity after aggressive compression/resize
    if sharpen:
        out_img = out_img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

    # use higher quality for JPEGs to preserve clarity; PNGs saved normally
    ext = os.path.splitext(out_path)[1].lower()
    save_kwargs = {}
    if ext in ('.jpg', '.jpeg'):
        save_kwargs['quality'] = 90
        save_kwargs['optimize'] = True

    out_img.save(out_path, **save_kwargs)
    return np.array(out_img)

def psnr(orig_arr, recon_arr):
    orig = orig_arr.astype(float)
    recon = recon_arr.astype(float)
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def find_k_for_target_psnr(in_path, scale, target_psnr, max_k=200):
    """
    Binary-search minimal k in [1, max_k] such that PSNR(orig_resized, reconstruction_k) >= target_psnr.
    Returns the chosen k (int). May write temporary files during search.
    """
    # load original and resize if needed
    img = Image.open(in_path).convert('RGB')
    try:
        scale = float(scale)
    except Exception:
        scale = 1.0
    if scale != 1.0:
        w, h = img.size
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.resize(new_size, Image.LANCZOS)
    orig_arr = np.array(img)

    lo, hi = 1, max_k
    chosen = max_k
    # binary search for minimal k meeting target
    while lo <= hi:
        mid = (lo + hi) // 2
        fd, tmp_path = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        try:
            comp_arr = compress_image_svd(in_path, tmp_path, mid, scale)
            score = psnr(orig_arr, comp_arr)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        if score >= target_psnr:
            chosen = mid
            hi = mid - 1
        else:
            lo = mid + 1

    return chosen
