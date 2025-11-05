from django.test import TestCase
from PIL import Image
import tempfile
import os
import numpy as np

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
