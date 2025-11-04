from django.urls import path
from . import views

app_name = 'compressor'

urlpatterns = [
    path('', views.index, name='index'),
    path('compress/', views.compress_view, name='compress'),
    path('preview/', views.compress_preview, name='preview'),  # new
]
