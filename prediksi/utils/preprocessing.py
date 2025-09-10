import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, top_k_accuracy_score
mapel_list = [
    'nilai_agama', 'nilai_pkn', 'nilai_bhs_indonesia', 'nilai_mtk_umum', 'nilai_sejarah_indo',
    'nilai_bhs_inggris', 'nilai_seni_budaya', 'nilai_penjas', 'nilai_prakarya',
    'nilai_mtk_peminatan', 'nilai_biologi', 'nilai_fisika', 'nilai_kimia'
]

semester = ['s1', 's2', 's3', 's4', 's5']

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

def load_dataset(filename):
    if not os.path.isabs(filename):
        base_dir = os.path.dirname(__file__)
        filename = os.path.join(base_dir, filename)
    return pd.read_csv(filename)


def generate_features(df):
    avg_semester_features = []
    trend_features = []
    std_features = []

    for mapel in mapel_list:
        kolom_mapel = [f"{mapel}_{s}" for s in semester]
        df[f'avg_{mapel}'] = df[kolom_mapel].mean(axis=1)
        df[f'std_{mapel}'] = df[kolom_mapel].std(axis=1)

        X_sem = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
        trends = [
            LinearRegression().fit(X_sem, row.values.reshape(-1, 1)).coef_[0][0]
            for _, row in df[kolom_mapel].iterrows()
        ]
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


# ✅ Fungsi fleksibel: menerima langsung DataFrame
def preprocess_dataframe(df, bobot_jurusan):
    df, _ = generate_features(df)
    skor_df = hitung_skor_jurusan_dengan_pembanding(df, bobot_jurusan)
    for col in skor_df.columns:
        df[f'skor_{col}'] = skor_df[col]

    # Deteksi kolom target
    if 'jurusan_diterima' not in df.columns:
        raise ValueError("Kolom 'jurusan_diterima' tidak ditemukan dalam DataFrame.")

    fitur_akhir = (
        [col for col in df.columns if col.startswith('avg_')] +
        [col for col in df.columns if col.startswith('std_')] +
        [col for col in df.columns if col.startswith('trend_')] +
        [col for col in df.columns if col.startswith('skor_')]
    )
    return df[fitur_akhir + ['jurusan_diterima']]


# Fungsi tetap: load dari file
def preprocessing(filename, bobot_jurusan):
    df = load_dataset(filename)
    return preprocess_dataframe(df, bobot_jurusan)


def proses_pca_dan_model(df):
    fitur_avg = [col for col in df.columns if col.startswith('avg_')]
    fitur_std = [col for col in df.columns if col.startswith('std_')]
    fitur_trend = [col for col in df.columns if col.startswith('trend_')]
    fitur_skor = [col for col in df.columns if col.startswith('skor_')]
    fitur_semua = fitur_avg + fitur_std + fitur_trend + fitur_skor

    threshold = 0
    value_counts = df['jurusan_diterima'].value_counts()
    df['jurusan_diterima'] = df['jurusan_diterima'].apply(
        lambda x: x if value_counts[x] >= threshold else 'lainnya'
    )

    X = df[fitur_semua]
    y = df['jurusan_diterima']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=30)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    return model