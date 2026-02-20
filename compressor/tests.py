from django.test import TestCase
from PIL import Image
import tempfile
import os
import numpy as np
from unittest.mock import patch

from .utils import compress_image_svd


class UtilsTests(TestCase):
	def test_compress_image_svd_respects_scale(self):
		# create a temporary image 100x80
		with tempfile.TemporaryDirectory() as td:
			in_path = os.path.join(td, 'test_in.png')
			out_path = os.path.join(td, 'test_out.png')
			img = Image.new('RGB', (100, 80), color=(123, 222, 64))
			img.save(in_path)

			# compress with scale 0.5 and k small
			comp_arr = compress_image_svd(in_path, out_path, k=10, scale=0.5)

			# comp_arr should have height ~40 and width ~50 (50x40)
			h, w, c = comp_arr.shape
			self.assertEqual(c, 3)
			self.assertEqual(w, 50)
			self.assertEqual(h, 40)

			# out_path should exist
			self.assertTrue(os.path.exists(out_path))


class IntegrationTests(TestCase):
	def test_preview_and_compress_endpoints(self):
		# create a small test image in-memory
		from django.urls import reverse
		from io import BytesIO
		from django.core.files.uploadedfile import SimpleUploadedFile
		client = self.client

		img = Image.new('RGB', (60, 40), color=(10, 20, 30))
		buf = BytesIO()
		img.save(buf, format='PNG')
		buf.seek(0)

		# POST to preview - use SimpleUploadedFile so Django treats it as an uploaded file
		uploaded = SimpleUploadedFile('test.png', buf.getvalue(), content_type='image/png')
		resp = client.post(reverse('compressor:preview'), {'image': uploaded, 'k': 10, 'size': '0.5'})
		# preview should return JSON with comp_url, psnr, sizes
		self.assertEqual(resp.status_code, 200)
		data = resp.json()
		self.assertIn('comp_url', data)
		self.assertIn('psnr', data)
		self.assertIn('comp_size', data)

		# Now POST full compress (multipart) - recreate uploaded file
		buf2 = BytesIO()
		img.save(buf2, format='PNG')
		buf2.seek(0)
		uploaded2 = SimpleUploadedFile('test2.png', buf2.getvalue(), content_type='image/png')
		resp2 = client.post(reverse('compressor:compress'), {'image': uploaded2, 'k': 10, 'size': '0.5'})
		# Should render result page (200) and contain PSNR text
		self.assertEqual(resp2.status_code, 200)
		self.assertIn(b'Compression Result', resp2.content)

	@patch('compressor.views.ML_AVAILABLE', True)
	@patch('compressor.views.compress_image_cnn')
	def test_preview_endpoint_with_cnn_method(self, mock_compress_image_cnn):
		from django.urls import reverse
		from io import BytesIO
		from django.core.files.uploadedfile import SimpleUploadedFile
		from django.conf import settings

		def fake_cnn_compress(in_path, out_path, model=None, model_path=None, latent_dim=64, model_type='standard'):
			in_img = Image.open(in_path).convert('RGB')
			arr = np.array(in_img)
			Image.fromarray(arr).save(out_path)
			return arr

		mock_compress_image_cnn.side_effect = fake_cnn_compress

		img = Image.new('RGB', (64, 64), color=(100, 150, 200))
		buf = BytesIO()
		img.save(buf, format='PNG')
		buf.seek(0)

		uploaded = SimpleUploadedFile('cnn_test.png', buf.getvalue(), content_type='image/png')

		model_dir = os.path.join(settings.BASE_DIR, 'compressor', 'trained_models')
		os.makedirs(model_dir, exist_ok=True)
		model_path = os.path.join(model_dir, 'autoencoder_standard_ld64.pth')
		with open(model_path, 'wb') as model_file:
			model_file.write(b'test-model-placeholder')

		try:
			resp = self.client.post(
				reverse('compressor:preview'),
				{
					'image': uploaded,
					'method': 'cnn',
					'k': 50,
					'latent_dim': 64,
					'model_type': 'standard',
					'size': '1.0',
				}
			)
		finally:
			if os.path.exists(model_path):
				os.remove(model_path)

		self.assertEqual(resp.status_code, 200)
		data = resp.json()
		self.assertEqual(data.get('method'), 'cnn')
		self.assertIn('comp_url', data)
		self.assertIn('psnr', data)
		self.assertIn('comp_size', data)
		self.assertIn('width', data)
		self.assertIn('height', data)
