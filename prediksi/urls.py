from django.urls import path, include
from . import views
from .upload_views import upload

app_name = 'prediksi'  # ✅ ini penting untuk fungsi redirect dengan namespace


urlpatterns = [
    #path('', views.index, name='index'),
    path('', views.prediksi_jurusan_view, name='prediksi_jurusan'),
    
    # path('upload/', include('prediksi.upload_views.urls', namespace='prediksi')),
    # path('upload-csv/', views.upload_csv_view, name='upload_csv'),
    # path('', views.upload_form_view, name='upload'),
    # path('', views.upload_csv_view, name='upload_csv'),
    path('form/', upload.upload_form_view, name='upload_form'),
    path('proses/', upload.upload_csv_view, name='upload_csv'),
    path('download_excel/', views.download_excel, name='download_excel')

]
