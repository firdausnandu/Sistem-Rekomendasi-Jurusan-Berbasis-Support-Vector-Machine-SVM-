import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

mapel_list = [
    'nilai_agama', 'nilai_pkn', 'nilai_bhs_indonesia', 'nilai_mtk_umum', 'nilai_sejarah_indo',
    'nilai_bhs_inggris', 'nilai_seni_budaya', 'nilai_penjas', 'nilai_prakarya',
    'nilai_mtk_peminatan', 'nilai_biologi', 'nilai_fisika', 'nilai_kimia'
]

semester = ['s1', 's2', 's3', 's4', 's5']

def load_dataset(filename):
    base_dir = os.path.dirname(__file__)  # Folder utils/
    path = os.path.join(base_dir, filename)
    return pd.read_csv(path)

def generate_features(df):
    avg_semester_features = []
    trend_features = []
    std_features = []

    for mapel in mapel_list:
        kolom_mapel = [f"{mapel}_{s}" for s in semester]
        df[f'avg_{mapel}'] = df[kolom_mapel].mean(axis=1)
        df[f'std_{mapel}'] = df[kolom_mapel].std(axis=1)

        X_sem = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        trends = [LinearRegression().fit(X_sem, row.values.reshape(-1, 1)).coef_[0][0]
                  for _, row in df[kolom_mapel].iterrows()]
        df[f'trend_{mapel}'] = trends

        avg_semester_features.append(f'avg_{mapel}')
        std_features.append(f'std_{mapel}')
        trend_features.append(f'trend_{mapel}')

    final_features = avg_semester_features + trend_features + std_features
    return df, final_features

def hitung_skor_jurusan_dengan_pembanding(df, bobot_jurusan):
    skor_dict = {}
    for jurusan, bobot_mapel in bobot_jurusan.items():
        total_bobot = 0
        total_nilai = 0

        for mapel, bobot in bobot_mapel.items():
            if isinstance(mapel, tuple):
                nilai = df[list(mapel)].mean(axis=1)
            else:
                nilai = df[mapel]
            total_nilai += nilai * bobot
            total_bobot += bobot

        skor = total_nilai / total_bobot if total_bobot > 0 else 0
        skor_dict[jurusan] = skor

    return pd.DataFrame(skor_dict)

# Bobot jurusan didefinisikan sebagai variabel
bobot_jurusan = {
    'agribisnis': {
        'avg_nilai_mtk_umum' : 0.5
    },
    'agronomi': {
        'avg_nilai_biologi' : 0.5
    },
    'akuntansi': {
        'avg_nilai_mtk_umum' : 0.5
    },
    'arsitektur': {
         ('avg_nilai_mtk_umum', 'avg_nilai_fisika'): 0.5
    },
    'bahasa inggris': {
        'avg_nilai_bhs_inggris' : 0.5
    },
    'biologi': {
        'avg_nilai_biologi' : 0.5
    },
    'bisnis perjalanan wisata (d4)': {
        'avg_nilai_mtk_umum' : 0.5
    },
    'farmasi': {
        ('avg_nilai_biologi', 'avg_nilai_kimia'): 0.5
    },
    'fisika': {
        'avg_nilai_fisika' : 0.5
    },
    'geofisika': {
        ('avg_nilai_fisika', 'avg_nilai_mtk_peminatan') : 0.5
    },
    'geografi lingkungan': {
        ('avg_nilai_fisika', 'avg_nilai_mtk_umum') : 0.5
    },
    'gizi kesehatan': {
        ('avg_nilai_biologi', 'avg_nilai_kimia'): 0.5
    },
    'higiene gigi': {
         ('avg_nilai_biologi', 'avg_nilai_kimia'): 0.5
    },
    'hukum': {
        'avg_nilai_pkn': 0.5
    },
    'ilmu dan industri peternakan': {
        'avg_nilai_biologi' : 0.5
    },
    'ilmu informasi dan perpustakaan': {
        'avg_nilai_mtk_peminatan' : 0.5
    },
    'ilmu keperawatan': {
        'avg_nilai_biologi' : 0.5
    },
    'ilmu komunikasi': {
        'avg_nilai_bhs_indonesia': 0.5
    },
    'ilmu tanah': {
        'avg_nilai_biologi' : 0.5
    },
    'informatika': {
        'avg_nilai_mtk_peminatan' : 0.5
    },
    'kartografi': {
        'avg_nilai_mtk_umum' : 0.5
    },
    'kartografi dan penginderaan jauh': {
        'avg_nilai_mtk_umum' : 0.5
    },
    'kedokteran': {
        ('avg_nilai_biologi', 'avg_nilai_kimia'): 0.5
    },
    'kedokteran gigi': {
        ('avg_nilai_biologi', 'avg_nilai_kimia'): 0.5
    },
    'kedokteran hewan': {
        ('avg_nilai_biologi', 'avg_nilai_kimia'): 0.5
    },
    'kimia': {
        'avg_nilai_kimia': 0.5
    },
    'manajemen': {
        'avg_nilai_mtk_umum' : 0.5
    },
    'manajemen informasi kesehatan': {
        'avg_nilai_mtk_umum' : 0.5
    },
    'manajemen sumberdaya akuatik': {
        'avg_nilai_mtk_umum' : 0.5
    },
    'matematika': {
        'avg_nilai_mtk_peminatan' : 0.5
    },
    'metalurgi': {
        ('avg_nilai_fisika', 'avg_nilai_mtk_peminatan','avg_nilai_kimia') : 0.5
    },
    'mipa ipa': {
        ('avg_nilai_fisika', 'avg_nilai_mtk_peminatan','avg_nilai_kimia','avg_nilai_biologi') : 0.5
    },
    'pendidikan doker gigi': {
        ('avg_nilai_biologi', 'avg_nilai_kimia'): 0.5
    },
    'pendidikan geografi': {
        ('avg_nilai_fisika', 'avg_nilai_mtk_umum') : 0.5
    },
    'pendidikan ipa': {
         ('avg_nilai_fisika', 'avg_nilai_mtk_peminatan','avg_nilai_kimia','avg_nilai_biologi') : 0.5
    },
    'pendidikan kedokteran': {
        ('avg_nilai_biologi', 'avg_nilai_kimia'): 0.5
    },
    'pendidikan kimia': {
        'avg_nilai_kimia': 0.5
    },
    'pendidikan luar biasa': {
        ('avg_nilai_fisika', 'avg_nilai_mtk_peminatan','avg_nilai_kimia','avg_nilai_biologi') : 0.5
    },
    'pendidikan matematika': {
        'avg_nilai_mtk_umum': 0.5
    },
    'pengembangan produk agroindustri': {
        ('avg_nilai_biologi', 'avg_nilai_kimia'): 0.5
    },
    'pembangunan wilayah': {
        'avg_nilai_mtk_umum': 0.5
    },
    'perpajakan': {
        'avg_nilai_mtk_umum': 0.5
    },
    'pgsd': {
        ('avg_nilai_fisika', 'avg_nilai_mtk_peminatan','avg_nilai_kimia','avg_nilai_biologi') : 0.5
    },
    'proteksi tanaman': {
        'avg_nilai_biologi': 0.5
    },
    'psikologi': {
        'avg_nilai_mtk_umum': 0.5
    },
    'sastra inggris': {
        'avg_nilai_bhs_inggris' : 0.5
    },
    'sistem informasi': {
        'avg_nilai_mtk_peminatan': 0.5
    },
    'sistem informasi geografis': {
        'avg_nilai_mtk_peminatan': 0.5
    },
    'teknik biomedis': {
        ('avg_nilai_biologi', 'avg_nilai_mtk_peminatan'): 0.5
    },
    'teknik elektro': {
        ('avg_nilai_fisika', 'avg_nilai_mtk_peminatan') : 0.5
    },
    'teknik geodesi': {
        ('avg_nilai_fisika', 'avg_nilai_mtk_peminatan') : 0.5
    },
    'teknik geofisika': {
        ('avg_nilai_fisika', 'avg_nilai_mtk_peminatan') : 0.5
    },
    'teknik geologi': {
        ('avg_nilai_fisika', 'avg_nilai_mtk_peminatan') : 0.5
    },
    'teknik industri': {
        ('avg_nilai_fisika', 'avg_nilai_mtk_peminatan') : 0.5
    },
    'teknik kimia': {
        ('avg_nilai_kimia', 'avg_nilai_mtk_peminatan') : 0.5
    },
    'teknik mesin': {
        ('avg_nilai_fisika', 'avg_nilai_mtk_peminatan') : 0.5
    },
    'teknik nuklir': {
        ('avg_nilai_kimia', 'avg_nilai_mtk_peminatan') : 0.5
    },
    'teknik pertambangan': {
        ('avg_nilai_fisika','avg_nilai_kimia', 'avg_nilai_mtk_peminatan') : 0.5
    },
    'teknik pertanian': {
        'avg_nilai_biologi': 0.5
    },
    'teknik sipil': {
        ('avg_nilai_mtk_umum', 'avg_nilai_fisika'): 0.5
    },
    'teknik sumber daya air': {
        'avg_nilai_biologi': 0.5
    },
    'teknologi hasil perikanan': {
        'avg_nilai_biologi': 0.5
    },
    'teknologi industri pertanian': {
        'avg_nilai_biologi': 0.5
    },
    'teknologi informasi': {
        'avg_nilai_mtk_peminatan': 0.5
    },
    'teknologi pangan dan hasil pertanian': {
        ('avg_nilai_biologi', 'avg_nilai_kimia'): 0.5
    },
    'teknologi rekayasa dan pemeliharaan bangunan sipil': {
        ('avg_nilai_mtk_umum', 'avg_nilai_fisika'): 0.5
    },
    'teknologi rekayasa elektro': {
        ('avg_nilai_fisika', 'avg_nilai_mtk_peminatan') : 0.5
    },
    'teknologi rekayasa instrumentasi dan kontrol': {
         ('avg_nilai_fisika', 'avg_nilai_mtk_peminatan') : 0.5
    },
    'teknologi rekayasa internet': {
        'avg_nilai_mtk_peminatan': 0.5
    },
    'teknologi rekayasa mesin': {
        ('avg_nilai_fisika', 'avg_nilai_mtk_peminatan') : 0.5
    },
    'teknologi rekayasa pelaksanaan bangunan sipil': {
         ('avg_nilai_mtk_umum', 'avg_nilai_fisika'): 0.5
    },
    'teknologi rekayasa perangkat lunak': {
        'avg_nilai_mtk_peminatan' : 0.5
    },
    'teknologi veteriner': {
        ('avg_nilai_biologi', 'avg_nilai_kimia'): 0.5
    },
}