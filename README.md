# Deteksi Persamaan pada Pokoknya Nama Merek Dagang

Sistem deteksi persamaan pada pokoknya nama merek dagang menggunakan model *hybrid similarity* yang mengintegrasikan tiga dimensi kemiripan — tekstual, fonetik, dan semantik — sebagai Sistem Pendukung Keputusan (SPK) bagi pemeriksa merek.

## Ringkasan

Penelitian ini mengembangkan model yang menerima satu nama merek sebagai kueri, mencari kandidat termirip dari basis data merek terdaftar (mode *retrieval*), dan menghasilkan skor kemiripan, keputusan sementara, daftar kandidat, serta penjelasan kontribusi tiap dimensi. Ketiga dimensi kemiripan dipetakan ke parameter "persamaan pada pokoknya" dalam Penjelasan Pasal 21 UU Nomor 20 Tahun 2016 tentang Merek dan Indikasi Geografis:

| Dimensi | Metode | Parameter Hukum |
|---|---|---|
| Tekstual | Jaro-Winkler | Persamaan tulisan |
| Fonetik | Double Metaphone (+ normalisasi fonetik Indonesia) | Persamaan bunyi ucapan |
| Semantik | FastText praterlatih (cc.id.300, direduksi 100 dimensi) | Persamaan makna |

Label kemiripan (*ground truth*) bersumber dari putusan Pengadilan Niaga dan Kasasi Mahkamah Agung, bukan dari aturan buatan peneliti, untuk menghindari *circularity* evaluasi.

## Struktur Repositori

```
├── notebooks/
│   ├── scraping_data_merek.ipynb                # Scraping data merek per kelas (awal)
│   ├── scrape_by_name.ipynb                     # Injeksi merek pembanding dari putusan
│   ├── preprocessing_final.ipynb                # Prapemrosesan nama merek
│   ├── ekstraksi_fitur.ipynb                    # Ekstraksi 3 fitur kemiripan
│   ├── retrieval_evaluasi.ipynb                 # Bobot, threshold, evaluasi
│   ├── uji_kontribusi_komponen_retrieval.ipynb  # Eksperimen ablasi & retrieval terjangkau
│   ├── error_analysis.ipynb                     # Analisis kesalahan
│   └── kaggle_reduksi_dari_upload.ipynb         # Reduksi model semantik (Kaggle)
├── app/
│   ├── app_merek.py                             # Aplikasi SPK (Streamlit)
│   └── streamlit.ipynb                          # Peluncur aplikasi via Colab + ngrok
└── README.md
```

## Cara Menjalankan

Seluruh notebook dirancang berjalan di Google Colab (dan satu notebook khusus di Kaggle), saling terhubung melalui berkas CSV/NPY perantara. Jalankan berurutan sebagai berikut.

1. **`notebooks/scraping_data_merek.ipynb`** — mengumpulkan data merek dari PDKI per kelas barang/jasa (300 merek × 45 kelas).
2. **`notebooks/scrape_by_name.ipynb`** — menyuntikkan (injeksi) merek pembanding dari putusan pengadilan ke basis data agar tercakup dalam retrieval. Menghasilkan `haystack_final.csv`.
3. **`notebooks/preprocessing_final.ipynb`** — memproses nama merek menjadi `nama_base` dan `nama_phon`. Menghasilkan `haystack_preprocessed_fonID.csv`, `haystack_didaftar_fonID.csv`, `gold_cases_preprocessed_fonID.csv`.
4. **`notebooks/kaggle_reduksi_dari_upload.ipynb`** *(dijalankan di Kaggle)* — mereduksi model FastText cc.id.300 menjadi 100 dimensi agar ringan dijalankan di Colab. Menghasilkan `cc.id.100.bin`, diunggah manual ke Google Drive.
5. **`notebooks/ekstraksi_fitur.ipynb`** — menghitung tiga fitur kemiripan dan memuat model semantik dari Drive. Menghasilkan `gold_cases_categorized.csv`, `haystack_emb.npy`.
6. **`notebooks/retrieval_evaluasi.ipynb`** — mencari bobot dan threshold optimal melalui *grid search* dengan validasi silang lima lipatan, lalu mengevaluasi model.
7. **`notebooks/uji_kontribusi_komponen_retrieval.ipynb`** — menjalankan eksperimen ablasi (uji kontribusi tiap komponen) dan menghitung metrik retrieval pada kueri terjangkau.
8. **`notebooks/error_analysis.ipynb`** — membedah kesalahan klasifikasi dan retrieval untuk menemukan pola kegagalan sistematis.

### Menjalankan Aplikasi SPK

```bash
pip install streamlit rapidfuzz metaphone pandas numpy fasttext-wheel
streamlit run app/app_merek.py
```

Atau jalankan dari Google Colab menggunakan `app/streamlit.ipynb`, yang akan membuka aplikasi melalui URL publik ngrok.

## Hasil Utama

| Metrik | Nilai |
|---|---|
| F1-Score (klasifikasi) | 0,861 |
| Recall (klasifikasi) | 0,853 |
| AUC | 0,968 |
| Recall@10 (retrieval, kueri terjangkau) | 0,615 |
| Bobot optimal | Tekstual 0,5 · Fonetik 0,1 · Semantik 0,4 |
| Threshold | 0,55 |

Eksperimen ablasi menunjukkan dimensi fonetik bersifat redundan terhadap dimensi tekstual pada data yang digunakan (selisih F1 0,002), sedangkan dimensi semantik memberikan kontribusi nyata (selisih F1 0,049). Rincian lengkap tersedia pada notebook `notebooks/uji_kontribusi_komponen_retrieval.ipynb` dan laporan penelitian.

## Data

Berkas data mentah hasil *scraping* dan model semantik tidak disertakan dalam repositori ini karena keterbatasan ukuran berkas dan pertimbangan privasi data pemilik merek. Data dapat diperoleh kembali dengan menjalankan notebook `notebooks/scraping_data_merek.ipynb` dan `notebooks/scrape_by_name.ipynb`, dan model semantik dapat diunduh mengikuti prosedur pada `notebooks/kaggle_reduksi_dari_upload.ipynb`.

## Catatan

Sistem ini bersifat **pendukung keputusan**; hasil yang ditampilkan merupakan indikasi awal untuk membantu pemeriksaan, bukan penetapan hukum atas ada atau tidaknya persamaan pada pokoknya.

## Penulis

Skripsi Program Studi [isi program studi] — [isi nama universitas], [isi tahun].
