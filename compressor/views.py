import os
from django.shortcuts import render
from django.conf import settings
from .forms import UploadForm
from .utils import compress_image_svd, psnr
from PIL import Image
import numpy as np
import time
from django.http import JsonResponse

def index(request):
    form = UploadForm()
    return render(request, 'compressor/index.html', {'form': form})

def compress_view(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.cleaned_data['image']
            k = form.cleaned_data['k']

            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)

            in_path = os.path.join(upload_dir, img.name)
            with open(in_path, 'wb+') as f:
                for chunk in img.chunks():
                    f.write(chunk)

            base, ext = os.path.splitext(img.name)
            out_name = f"{base}_svd_k{k}{ext}"
            out_path = os.path.join(upload_dir, out_name)

            # read optional size (scale) from form (string like '1.0', '0.5')
            size = form.cleaned_data.get('size', '1.0')
            try:
                scale = float(size)
            except Exception:
                scale = 1.0

            out_name = f"{base}_svd_k{k}_s{int(scale*100)}{ext}"
            out_path = os.path.join(upload_dir, out_name)

            sharpen = form.cleaned_data.get('sharpen', False)
            comp_arr = compress_image_svd(in_path, out_path, k, scale, sharpen=sharpen)
            # get compressed image dimensions
            try:
                comp_img = Image.open(out_path)
                comp_w, comp_h = comp_img.size
            except Exception:
                comp_w, comp_h = comp_arr.shape[1], comp_arr.shape[0]

            # load original array for PSNR; if we resized before compression, compare to resized original
            orig_img = Image.open(in_path).convert('RGB')
            if scale != 1.0:
                # comp_arr has shape (h, w, 3)
                h, w = comp_arr.shape[0], comp_arr.shape[1]
                orig_img = orig_img.resize((w, h), Image.LANCZOS)
            orig = np.array(orig_img)
            computed_psnr = psnr(orig, comp_arr)

            orig_size = os.path.getsize(in_path)
            comp_size = os.path.getsize(out_path)
            ratio = orig_size / comp_size if comp_size > 0 else float('inf')

            context = {
                'form': form,
                'orig_url': settings.MEDIA_URL + f'uploads/{os.path.basename(in_path)}?t={int(time.time()*1000)}',
                'comp_url': settings.MEDIA_URL + f'uploads/{os.path.basename(out_path)}?t={int(time.time()*1000)}',
                'size': scale,
                'size_label': next((label for val, label in form.SIZE_CHOICES if float(val) == scale), f"{int(scale*100)}%"),
                'k': k,
                'orig_size': orig_size,
                'comp_size': comp_size,
                'comp_width': comp_w,
                'comp_height': comp_h,
                'ratio': ratio,
                'psnr': computed_psnr,
            }
            return render(request, 'compressor/result.html', context)
    else:
        form = UploadForm()
    return render(request, 'compressor/index.html', {'form': form})
def compress_preview(request):
    """
    AJAX endpoint: accepts POST with 'image' file and 'k' int.
    Returns JSON: { comp_url, psnr, orig_size, comp_size }
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)

    form = UploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({'error': 'invalid form', 'details': form.errors}, status=400)

    img = form.cleaned_data['image']
    k = form.cleaned_data['k']
    size = form.cleaned_data.get('size', '1.0')
    try:
        scale = float(size)
    except Exception:
        scale = 1.0

    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    ts = int(time.time() * 1000)
    safe_name = f"tmp_{ts}_{img.name}"
    in_path = os.path.join(upload_dir, safe_name)
    with open(in_path, 'wb+') as f:
        for chunk in img.chunks():
            f.write(chunk)

    base, ext = os.path.splitext(safe_name)
    out_name = f"{base}_svd_k{k}_s{int(scale*100)}{ext}"
    out_path = os.path.join(upload_dir, out_name)

    # compress and get numpy array for psnr
    sharpen = form.cleaned_data.get('sharpen', False)
    comp_arr = compress_image_svd(in_path, out_path, k, scale, sharpen=sharpen)
    # get compressed image dimensions
    try:
        comp_img = Image.open(out_path)
        comp_w, comp_h = comp_img.size
    except Exception:
        comp_w, comp_h = comp_arr.shape[1], comp_arr.shape[0]
    orig_img = Image.open(in_path).convert('RGB')
    if scale != 1.0:
        h, w = comp_arr.shape[0], comp_arr.shape[1]
        orig_img = orig_img.resize((w, h), Image.LANCZOS)
    orig = np.array(orig_img)
    computed_psnr = psnr(orig, comp_arr)

    orig_size = os.path.getsize(in_path)
    comp_size = os.path.getsize(out_path)

    comp_url = settings.MEDIA_URL + f'uploads/{os.path.basename(out_path)}'

    return JsonResponse({
        'comp_url': comp_url,
        'psnr': round(float(computed_psnr), 2),
        'orig_size': orig_size,
        'comp_size': comp_size,
        'width': comp_w,
        'height': comp_h,
        'size': scale,
        'size_label': next((label for val, label in form.SIZE_CHOICES if float(val) == scale), f"{int(scale*100)}%"),
    })


