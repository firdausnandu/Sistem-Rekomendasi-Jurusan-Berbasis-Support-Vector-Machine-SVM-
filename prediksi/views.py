from django.shortcuts import render

# Create your views here.
# views.py
from django.http import JsonResponse
from .utils import preprocessing as prep

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from io import BytesIO
from django.http import HttpResponse


DATASET_PATH = 'C:/Users/User/Documents/jurusanku/rekomendasi_jurusan/prediksi/utils/dataset_mipa1.csv'
mapel_list = [
    'nilai_agama', 'nilai_pkn', 'nilai_bhs_indonesia', 'nilai_mtk_umum', 'nilai_sejarah_indo',
    'nilai_bhs_inggris', 'nilai_seni_budaya', 'nilai_penjas', 'nilai_prakarya',
    'nilai_mtk_peminatan', 'nilai_biologi', 'nilai_fisika', 'nilai_kimia'
]

semester = ['s1', 's2', 's3', 's4', 's5']
def download_excel(request):
    hasil_rekomendasi = request.session.get('hasil_rekomendasi', [])

    if not hasil_rekomendasi:
        return HttpResponse("Data tidak ditemukan", status=404)

    all_data = []
    for hasil in hasil_rekomendasi:
        siswa_ke = hasil.get('siswa_ke')
        for item in hasil.get('top_5', []):
            all_data.append({
                'Siswa Ke': siswa_ke,
                'Jurusan': item['jurusan'],
                'Probabilitas': item['probabilitas'],
            })

    df = pd.DataFrame(all_data)
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)

    response = HttpResponse(buffer, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename="hasil_rekomendasi.xlsx"'
    return response

def prediksi_jurusan_view(request):
    # 1. Load dan preprocessing dataset (dengan label)
    df = prep.preprocessing(DATASET_PATH, prep.bobot_jurusan)

    # 2. Proses fitur
    fitur_avg = [col for col in df.columns if col.startswith('avg_')]
    fitur_std = [col for col in df.columns if col.startswith('std_')]
    fitur_trend = [col for col in df.columns if col.startswith('trend_')]
    fitur_skor = [col for col in df.columns if col.startswith('skor_')]
    fitur_semua = fitur_avg + fitur_std + fitur_trend + fitur_skor

    # 3. Gabungkan kelas langka (opsional, bisa dihilangkan jika tidak relevan)
    threshold = 0
    value_counts = df['fakultas_diterima'].value_counts()
    df['fakultas_diterima'] = df['fakultas_diterima'].apply(
        lambda x: x if value_counts[x] >= threshold else 'lainnya'
    )

    # 4. Pisahkan X dan y
    X = df[fitur_semua]
    y = df['fakultas_diterima']

    # 5. Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. PCA
    pca = PCA(n_components=30)
    X_pca = pca.fit_transform(X_scaled)

    # 7. Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # 8. Train model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    # 9. Prediksi dari data X_test
    y_pred = model.predict(X_test)
    akurasi = accuracy_score(y_test, y_pred)

    # 10. Format hasil untuk ditampilkan
    hasil = [{"prediksi_jurusan": jurusan} for jurusan in y_pred]
    return JsonResponse({'data': hasil,
                         'akurasi_model': round(akurasi, 4)})

# # Ganti path sesuai penyimpanan Anda
# DATASET_PATH = 'C:/Users/User/Documents/jurusanku/rekomendasi_jurusan/prediksi/utils/dataset_mipa1.csv'

# # from django.http import HttpResponse

# # def index(request):
# #     return HttpResponse("Halaman Index Prediksi Jurusan")
# def prediksi_jurusan_view(request):
#     df = preprocessing.load_dataset(DATASET_PATH)
#     df, fitur_terpilih = preprocessing.generate_features(df)
#     skor_df = preprocessing.hitung_skor_jurusan_dengan_pembanding(df, preprocessing.bobot_jurusan)

#     hasil_prediksi = skor_df.idxmax(axis=1)
#     df_result = df.copy()
#     df_result['prediksi_jurusan'] = hasil_prediksi

#     hasil = df_result[['prediksi_jurusan']].to_dict(orient='records')
#     return JsonResponse({'data': hasil})

# def upload_csv_view(request):
#     if request.method == 'POST' and request.FILES.get('file'):
#         file = request.FILES['file']
#         df_baru = pd.read_csv(file)

#         # === 1. Training model dari dataset tetap ===
#         df_latih = preprocessing.load_dataset(DATASET_PATH)
#         df_latih, _ = preprocessing.generate_features(df_latih)
#         skor_df = preprocessing.hitung_skor_jurusan_dengan_pembanding(df_latih, preprocessing.bobot_jurusan)
#         for col in skor_df.columns:
#             df_latih[f'skor_{col}'] = skor_df[col]

#         fitur_avg = [col for col in df_latih.columns if col.startswith('avg_')]
#         fitur_std = [col for col in df_latih.columns if col.startswith('std_')]
#         fitur_trend = [col for col in df_latih.columns if col.startswith('trend_')]
#         fitur_skor = [col for col in df_latih.columns if col.startswith('skor_')]
#         fitur_semua = fitur_avg + fitur_std + fitur_trend + fitur_skor

#         X = df_latih[fitur_semua]
#         y = df_latih['fakultas_diterima']

#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)

#         pca = PCA(n_components=30)
#         X_pca = pca.fit_transform(X_scaled)

#         X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

#         model = SVC(kernel='linear', probability=True)
#         model.fit(X_train, y_train)

#         # === 2. Proses CSV Baru ===
#         for mapel in mapel_list:
#             kolom_mapel = [f"{mapel}_{s}" for s in semester]
#             df_baru[f'avg_{mapel}'] = df_baru[kolom_mapel].mean(axis=1)
#             df_baru[f'std_{mapel}'] = df_baru[kolom_mapel].std(axis=1)

#             X_sem = np.array(semester).reshape(-1, 1)
#             trends = []
#             for _, row in df_baru[kolom_mapel].iterrows():
#                 y_values = row.values.reshape(-1, 1)
#                 model_lin = LinearRegression().fit(X_sem, y_values)
#                 trends.append(model_lin.coef_[0][0])
#             df_baru[f'trend_{mapel}'] = trends

#         # Hitung skor jurusan
#         skor_baru = preprocessing.hitung_skor_jurusan_dengan_pembanding(df_baru, preprocessing.bobot_jurusan)
#         for col in skor_baru.columns:
#             df_baru[f'skor_{col}'] = skor_baru[col]

#         fitur_avg = [col for col in df_baru.columns if col.startswith('avg_')]
#         fitur_std = [col for col in df_baru.columns if col.startswith('std_')]
#         fitur_trend = [col for col in df_baru.columns if col.startswith('trend_')]
#         fitur_skor = [col for col in df_baru.columns if col.startswith('skor_')]
#         fitur_semua = fitur_avg + fitur_std + fitur_trend + fitur_skor

#         X_baru = df_baru[fitur_semua]
#         X_baru_scaled = scaler.transform(X_baru)
#         X_baru_pca = pca.transform(X_baru_scaled)

#         probabilitas = model.predict_proba(X_baru_pca)

#         # === 3. Ambil Top-3 Rekomendasi ===
#         hasil_rekomendasi = []
#         for idx, row in df_baru.iterrows():
#             proba_dict = dict(zip(model.classes_, probabilitas[idx]))
#             top3 = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:3]

#             rekomendasi = {
#                 'siswa_ke': idx + 1,
#                 'top_3': [
#                     {'jurusan': jur, 'probabilitas': round(prob, 4)}
#                     for jur, prob in top3
#                 ]
#             }
#             hasil_rekomendasi.append(rekomendasi)

#         return JsonResponse({'rekomendasi': hasil_rekomendasi})

#     return JsonResponse({'error': 'Gunakan POST dan upload file CSV.'}, status=400)