from django.shortcuts import render

# Create your views here.
# views.py
from django.http import JsonResponse
from ..utils import preprocessing
from ..utils.jurusan_info import info_jurusan

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import json


DATASET_PATH = 'C:/Users/User/Documents/jurusanku/rekomendasi_jurusan/prediksi/utils/dataset_mipa1.csv'
mapel_list = [
    'nilai_agama', 'nilai_pkn', 'nilai_bhs_indonesia', 'nilai_mtk_umum', 'nilai_sejarah_indo',
    'nilai_bhs_inggris', 'nilai_seni_budaya', 'nilai_penjas', 'nilai_prakarya',
    'nilai_mtk_peminatan', 'nilai_biologi', 'nilai_fisika', 'nilai_kimia'
]

semester = ['s1', 's2', 's3', 's4', 's5']

def upload_form_view(request):
    return render(request, 'prediksi/upload_form.html')

# def upload_csv_view(request):
#     if request.method == 'POST' and request.FILES.get('file'):
#         file = request.FILES['file']
#         df_baru = pd.read_csv(file)
def upload_csv_view(request):
    if request.method == 'POST' and request.FILES.get('file_prediksi'):
        file_prediksi = request.FILES['file_prediksi']
        df_baru = pd.read_csv(file_prediksi)
        file_latih = request.FILES.get('file_latih')  # Optional

        # Load dan proses data latih
        if file_latih:
            df_latih = pd.read_csv(file_latih)
        else:
            df_latih = preprocessing.load_dataset(DATASET_PATH)

        df_latih, _ = preprocessing.generate_features(df_latih)
        skor_df = preprocessing.hitung_skor_jurusan_dengan_pembanding(df_latih, preprocessing.bobot_jurusan)
        for col in skor_df.columns:
            df_latih[f'skor_{col}'] = skor_df[col]

        fitur_avg = [col for col in df_latih.columns if col.startswith('avg_')]
        fitur_std = [col for col in df_latih.columns if col.startswith('std_')]
        fitur_trend = [col for col in df_latih.columns if col.startswith('trend_')]
        fitur_skor = [col for col in df_latih.columns if col.startswith('skor_')]
        fitur_semua = fitur_avg + fitur_std + fitur_trend + fitur_skor

        X = df_latih[fitur_semua]
        y = df_latih['jurusan_diterima']
   
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=30)
        X_pca = pca.fit_transform(X_scaled)

        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, y_train)

        for mapel in mapel_list:
            kolom_mapel = [f"{mapel}_{s}" for s in semester]

            # Pastikan kolom mapel numeric dan ganti NaN dengan 0
            df_baru[kolom_mapel] = df_baru[kolom_mapel].apply(pd.to_numeric, errors='coerce').fillna(0)

            # Hitung rata-rata dan std deviasi
            df_baru[f'avg_{mapel}'] = df_baru[kolom_mapel].mean(axis=1)
            df_baru[f'std_{mapel}'] = df_baru[kolom_mapel].std(axis=1)

            # Hitung tren menggunakan LinearRegression
            X_sem = np.array([int(s[1]) for s in semester]).reshape(-1, 1)  # misal s1,s2... jadi 1,2,...
            trends = []
            for _, row in df_baru[kolom_mapel].iterrows():
                y_values = row.values.reshape(-1, 1)
                model_lin = LinearRegression().fit(X_sem, y_values)
                trends.append(model_lin.coef_[0][0])
            df_baru[f'trend_{mapel}'] = trends

        # Hitung skor jurusan
        skor_baru = preprocessing.hitung_skor_jurusan_dengan_pembanding(df_baru, preprocessing.bobot_jurusan)

        # Tambahkan skor ke df_baru sekaligus untuk mencegah fragmentasi
        skor_baru_renamed = skor_baru.add_prefix('skor_')
        df_baru = pd.concat([df_baru, skor_baru_renamed], axis=1)


        fitur_avg = [col for col in df_baru.columns if col.startswith('avg_')]
        fitur_std = [col for col in df_baru.columns if col.startswith('std_')]
        fitur_trend = [col for col in df_baru.columns if col.startswith('trend_')]
        fitur_skor = [col for col in df_baru.columns if col.startswith('skor_')]
        fitur_semua = fitur_avg + fitur_std + fitur_trend + fitur_skor

        X_baru = df_baru[fitur_semua]

        # convert ke numerik (penting!)
        X_baru = X_baru.apply(pd.to_numeric, errors='coerce').fillna(0)

        X_baru_scaled = scaler.transform(X_baru)
        X_baru_pca = pca.transform(X_baru_scaled)

        probabilitas = model.predict_proba(X_baru_pca)

        # # # === 3. Ambil Top-3 Rekomendasi ===
        # hasil_rekomendasi = []
        # for idx, row in df_baru.iterrows():
        #     proba_dict = dict(zip(model.classes_, probabilitas[idx]))
        #     top5 = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:5]

        #     tren_siswa = {col: round(row[col], 4) for col in df_baru.columns if col.startswith('trend_')}

        #     rekomendasi = {
        #         'siswa_ke': idx + 1,
        #         'tren_nilai': tren_siswa,
        #         'top_5': [
        #             {'jurusan': jur, 
        #              'probabilitas': round(prob, 4),
        #              'pengertian': info_jurusan.get(jur, {}).get('pengertian', 'Tidak tersedia'),
        #              'mata_kuliah': info_jurusan.get(jur, {}).get('mata_kuliah', []),
        #              'prospek': info_jurusan.get(jur, {}).get('prospek', 'Tidak tersedia')}
        #             for jur, prob in top5
        #         ]
        #     }
        #     hasil_rekomendasi.append(rekomendasi)
        
        # return render(request, 'prediksi/hasil_rekomendasi.html', {
        #     'hasil_rekomendasi': hasil_rekomendasi,
        #     'info_jurusan': info_jurusan})

        hasil_rekomendasi = []

        for idx, row in df_baru.iterrows():
            proba_dict = dict(zip(model.classes_, probabilitas[idx]))
            top5 = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:5]

            tren_siswa = {col: round(row[col], 4) for col in df_baru.columns if col.startswith('trend_')}

            # Ambil nilai tiap semester per mapel (sesuai mapel_list dan semester yang sudah ada)
            nilai_per_mapel = {}
            for mapel in mapel_list:
                kolom_semester = [f"{mapel}_{s}" for s in semester]
                nilai_per_mapel[mapel] = [round(row[kol], 2) for kol in kolom_semester if kol in df_baru.columns]

            rekomendasi = {
                'siswa_ke': idx + 1,
                'tren_nilai': tren_siswa,
                'nilai_semester': json.dumps(nilai_per_mapel),  # data nilai per semester untuk grafik
                'nilai_semester_dict': nilai_per_mapel,  
                'top_5': [
                    {
                        'jurusan': jur,
                        'probabilitas': round(prob, 4),
                        'pengertian': info_jurusan.get(jur, {}).get('pengertian', 'Tidak tersedia'),
                        'mata_kuliah': info_jurusan.get(jur, {}).get('mata_kuliah', []),
                        'prospek': info_jurusan.get(jur, {}).get('prospek', 'Tidak tersedia')
                    }
                    for jur, prob in top5
                ]
            }
            hasil_rekomendasi.append(rekomendasi)
            
            request.session['hasil_rekomendasi'] = [
                {
                    'siswa_ke': hasil['siswa_ke'],
                    'top_5': [
                        {
                            'jurusan': item['jurusan'],
                            'probabilitas': float(item['probabilitas'])
                        }
                        for item in hasil['top_5']
                    ]
                }
                for hasil in hasil_rekomendasi
            ]

        return render(request, 'prediksi/hasil_rekomendasi.html', {
            'hasil_rekomendasi': hasil_rekomendasi,
            'info_jurusan': info_jurusan
        })


    return JsonResponse({'error': 'Gunakan POST dan upload file CSV.'}, status=400)