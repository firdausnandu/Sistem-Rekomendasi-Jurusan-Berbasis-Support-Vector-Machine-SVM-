# prediksi/upload/urls.py
from django.urls import path
from ..views import prediksi_jurusan_view
from . import upload

app_name = 'prediksi'

urlpatterns = [
    path('prediksi/', prediksi_jurusan_view, name='prediksi_jurusan'),
    path('form/', upload.upload_form_view, name='upload_form'),
    path('proses/', upload.upload_csv_view, name='upload_csv'),
]
