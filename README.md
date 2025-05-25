
# Laporan Proyek Machine Learning Terapan Pertama — Nabilah Wanara

## Domain Proyek  
Proyek ini berfokus pada bidang **sosial-ekonomi**, dengan topik utama klasifikasi status kemiskinan menggunakan pendekatan *machine learning*.

### Latar Belakang  
Menurut Badan Pusat Statistik (BPS), pada Maret 2024 persentase penduduk miskin di Indonesia tercatat sebesar 9,03% dari total populasi [[1](https://www.bps.go.id/id/pressrelease/2024/07/01/2370/persentase-penduduk-miskin-maret-2024-turun-menjadi-9-03-persen-.html)]. Meski angka ini menunjukkan penurunan dari tahun-tahun sebelumnya, tantangan dalam mengatasi kemiskinan masih menjadi agenda penting dalam pembangunan nasional. Salah satu tantangan utamanya terletak pada efektivitas distribusi program bantuan sosial dan ekonomi yang kerap tidak tepat sasaran. Ketidaktepatan ini banyak disebabkan oleh keterbatasan sistem pendataan yang masih konvensional serta belum optimalnya pemanfaatan teknologi dalam proses analisis dan pengambilan keputusan.

Dalam era digital saat ini, perkembangan teknologi informasi, khususnya machine learning, membuka peluang baru dalam meningkatkan akurasi identifikasi kelompok masyarakat yang membutuhkan. Pendekatan klasifikasi berbasis data sosial-ekonomi memungkinkan pemetaan status kemiskinan secara lebih sistematis, cepat, dan objektif. Dengan analisis prediktif, pemerintah dan lembaga sosial dapat memperoleh gambaran yang lebih rinci mengenai karakteristik kelompok miskin, sehingga program bantuan dapat lebih terarah dan tepat sasaran.

Sejumlah penelitian sebelumnya telah membuktikan potensi machine learning dalam konteks klasifikasi sosial-ekonomi. Salah satunya adalah studi oleh Obeid et al. (2021), yang menerapkan berbagai algoritma machine learning untuk mengklasifikasikan tingkat kemiskinan di Yordania dan berhasil memperoleh hasil prediksi yang akurat dan signifikan [[2](https://www.researchgate.net/publication/348898452_Poverty_Classification_Using_Machine_Learning_The_Case_of_Jordan)]. Temuan seperti ini menunjukkan bahwa teknologi data science memiliki peran strategis dalam mendukung perencanaan kebijakan yang berbasis bukti (evidence-based policy), termasuk di Indonesia.

Oleh karena itu, proyek ini menjadi penting untuk dilakukan sebagai upaya mengintegrasikan teknologi machine learning dalam sistem klasifikasi kemiskinan. Harapannya, model yang dihasilkan dapat membantu pemerintah maupun organisasi sosial dalam menyusun strategi intervensi yang lebih efektif, efisien, dan berkelanjutan dalam mengatasi kemiskinan.

## Business Understanding

### Permasalahan:
- Bagaimana cara memanfaatkan *machine learning* untuk memprediksi status kemiskinan berbasis data sosial-ekonomi?  
- Model algoritma apa yang paling efektif dalam klasifikasi tersebut?  
- Bagaimana mengatasi ketidakseimbangan label dalam data?  

### Tujuan:
- Mengembangkan model klasifikasi status kemiskinan menggunakan data yang tersedia.
- Membandingkan performa berbagai algoritma klasifikasi.
- Meningkatkan kinerja model terhadap data tidak seimbang.
- Menganalisis fitur penting melalui Exploratory Data Analysis (EDA).

### Strategi Solusi:
- Menerapkan model **K-Nearest Neighbour**, **Decision Tree**, dan **Random Forest** sebagai baseline.
- Menentukan model terbaik berdasarkan metrik akurasi, precision, recall, dan F1-score.
- Menggunakan teknik SMOTE untuk mengatasi ketidakseimbangan kelas.
- Melakukan analisis visual data secara univariat dan multivariat.

## Data Understanding  
Data bersumber dari [Kaggle - Klasikasi Kemiskinan](https://www.kaggle.com/datasets/ermila/klasifikasi-kemiskinan), yang mencakup informasi sosial-ekonomi dari berbagai wilayah di Indonesia.  
- **Jumlah data**: 514 baris  
- **Fitur**: 7 kolom  

![struktur_dataset](https://github.com/user-attachments/assets/3f77135c-3a77-4e29-975a-c70d4dddf15e)

### Deskripsi Variabel:
- `Provinsi` : Nama provinsi asal data
- `Kab/Kota` : Nama kabupaten/kota
- `Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)` : Persentase jumlah penduduk miskin di masing-masing kabupaten/kota
- `Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)` : Jumlah pengeluaran per orang per tahun
- `Indeks Pembangunan Manusia` : Nilai IPM daerah tersebut
- `Tingkat Pengangguran Terbuka` : Persentase pengangguran terbuka di wilayah tersebut
- `Klasifikasi Kemiskinan` : Label target (0 = Tidak Miskin, 1 = Miskin) 

### EDA Univariate  
Distribusi fitur numerik dianalisis untuk mengidentifikasi pola umum. Beberapa temuan:

- Persentase kemiskinan mayoritas berada di bawah 15%, namun ada yang mencapai lebih dari 20%.
- Pengeluaran per kapita mayoritas antara 8–12 juta rupiah per tahun.
- IPM terdistribusi normal di kisaran 65–75.
- Tingkat pengangguran banyak berada di bawah 6%.
- Kategori target tidak seimbang: label tidak miskin jauh lebih dominan.

![Gambar1a](https://github.com/user-attachments/assets/6c93dc6e-6e91-436b-8749-827c4b26760e)

Pada tahap EDA univariate, dilakukan analisis distribusi masing-masing variabel numerik dan target untuk memahami pola sebaran data:

- Distribusi Persentase Penduduk Miskin per Kabupaten/Kota (persen_kemiskinan_kota)
Distribusi data bersifat positif skewed dengan mayoritas kabupaten/kota memiliki persentase kemiskinan di rentang 5% hingga 15%, sementara nilai ekstrem di atas 20% cukup jarang ditemukan.
- Distribusi Pengeluaran per Kapita Disesuaikan (pengeluaran_kapita)
Distribusi pengeluaran per kapita juga positif skewed, dengan sebagian besar wilayah memiliki pengeluaran per kapita antara 8.000 hingga 12.000 ribu rupiah/tahun. Nilai ekstrim di atas 14.000 ribu rupiah relatif jarang.
- Distribusi Indeks Pembangunan Manusia (IPM)
Variabel IPM cenderung mengikuti distribusi normal dengan rata-rata di sekitar 70. Sebagian besar daerah memiliki IPM dalam rentang 65–75, sementara nilai di bawah 60 dan di atas 80 jarang terjadi.
- Distribusi Tingkat Pengangguran Terbuka (tingkat_pengangguran)
Distribusi tingkat pengangguran juga bersifat positif skewed, dengan mayoritas kabupaten/kota memiliki tingkat pengangguran di bawah 6%, dan beberapa daerah mencapai di atas 10%.
- Distribusi Klasifikasi Kemiskinan (klasifikasi_kemiskinan)
Distribusi target klasifikasi_kemiskinan sangat imbalance, di mana sebagian besar data diklasifikasikan sebagai tidak miskin (label 0), sedangkan label miskin (label 1) jumlahnya jauh lebih sedikit. Hal ini menunjukkan ketidakseimbangan kelas yang perlu diperhatikan saat pemodelan.

Distribusi provinsi juga divisualisasikan:
![gambar1b](https://github.com/user-attachments/assets/eec97911-baad-456d-8df1-140599f3ff00)

Distribusi data pada kolom provinsi : 
-   Data dengan sedikit kontribusi ada di provinsi DKI Jakarta
-   Kontibusi data paling banyak ada di provinsi Jawa Timur

### EDA Multivariate  
- **Distribusi antar provinsi** menunjukkan variasi signifikan antara proporsi miskin dan tidak miskin. Beberapa wilayah memiliki distribusi seimbang.
![gambar2a](https://github.com/user-attachments/assets/c7a1b2bf-8fae-40b1-8a98-ca2727d4ea63)

Variasi Pola Kemiskinan Antar Provinsi: Pola distribusi antara kategori "0" dan "1" sangat bervariasi antar provinsi. Ada provinsi di mana jumlah yang tidak miskin jauh lebih banyak dari yang miskin, ada yang perbedaannya tidak terlalu besar, dan bahkan ada beberapa provinsi (meskipun terlihat sedikit) di mana jumlah yang diklasifikasikan sebagai miskin hampir sebanding atau bahkan lebih banyak dari yang tidak miskin.
Beberapa provinsi terlihat memiliki batang kategori "0" yang jauh lebih tinggi daripada batang kategori "1", mengindikasikan proporsi "kemiskinan" yang relatif rendah. Contoh Provinsi Spesifik:
  + Aceh: Terlihat memiliki jumlah kategori 0 yang lebih tinggi dari kategori 1.
  + Papua: Menarik untuk diperhatikan bahwa di Papua, jumlah kategori 1 (miskin) terlihat lebih tinggi dibandingkan kategori 0 (tidak miskin).
  + Nusa tenggara timur & Maluku: memiliki jumlah kategori 0 (tidak miskin) dan kategori 1 (miskin) yang hampir sama


- **Pairplot** menunjukkan korelasi negatif antara pengeluaran/IPM terhadap kemiskinan dan hubungan positif antara pengeluaran dan IPM.
![gambar2b](https://github.com/user-attachments/assets/0f6e76d4-4c19-4f85-852d-a0b0ed1b6a78)



- **Heatmap korelasi** memperlihatkan bahwa pengeluaran dan IPM sangat berkorelasi satu sama lain, sementara pengangguran memiliki korelasi lemah terhadap target.
![gambar2c](https://github.com/user-attachments/assets/0ad585a9-2bc3-4a79-b4f5-34063413a785)
1. Pengeluaran per Kapita memiliki korelasi positif sangat kuat terhadap IPM sebesar 0.85, artinya daerah dengan pengeluaran per kapita yang tinggi umumnya memiliki IPM yang tinggi pula.
2. Persentase Kemiskinan Kota memiliki korelasi negatif sedang terhadap pengeluaran per kapita (-0.52) dan IPM (-0.47). Artinya, semakin tinggi pengeluaran dan IPM suatu wilayah, semakin rendah persentase kemiskinannya.
3. Tingkat Pengangguran menunjukkan korelasi positif lemah terhadap IPM (0.47) dan pengeluaran per kapita (0.43), tetapi hubungan ini cenderung tidak sekuat korelasi antar fitur sebelumnya.
4. Terhadap target klasifikasi_kemiskinan, fitur dengan korelasi paling tinggi adalah:
    + Persentase Kemiskinan Kota (0.57) → korelasi positif cukup kuat.
    + Pengeluaran per Kapita (-0.29) dan IPM (-0.27) → korelasi negatif lemah ke sedang.
    + Tingkat Pengangguran memiliki korelasi paling lemah terhadap klasifikasi kemiskinan (-0.035).


## Data Preparation  
Langkah-langkah utama:
- Menstandarkan nama kolom menjadi format pythonic.
- Mengubah tipe data ke numerik untuk kolom yang kurang tepat tipe datanya.
- Menghapus outlier dengan IQR method.
- One-hot encoding untuk provinsi.
- StandardScaler untuk menyesuaikan skala fitur.
- SMOTE untuk menyeimbangkan kelas.
- Split data: 80% train, 20% test.

**Alasan:**  
- Mengganti nama kolom dilakukan untuk memudahkan proses eksplorasi, analisis, dan pemodelan karena nama kolom yang ringkas, konsisten, dan mudah dipanggil dapat mengurangi potensi error saat coding
- Mengubah tipe data agar data numerik dapat digunakan dalam perhitungan statistik, visualisasi, serta sebagai input model machine learning yang hanya menerima data numerik untuk operasi matematis.
- Handling missing values diperiksa untuk memastikan tidak ada data kosong yang dapat mengganggu hasil model atau analisis.
- Handling duplicate values diperiksa agar tidak ada duplikasi data yang bisa membuat bobot informasi menjadi tidak proporsional.
- Handling Outlier (IQR Method) dilakukan untuk mengurangi pengaruh data ekstrem yang bisa mendistorsi parameter model, terutama untuk algoritma berbasis pohon (Decision Tree, Random Forest) yang cukup sensitif terhadap outlier. Dengan membersihkan outlier, model bisa lebih stabil dan akurat.
- One Hot Encoding untuk kolom kategorikal (provinsi) Karena algoritma machine learning tidak dapat memproses data kategorikal dalam format string, maka perlu diubah menjadi format numerik biner agar bisa diproses dengan benar.
- Data Scaling (StandardScaler) penting untuk model-model seperti K-Nearest Neighbour dan algoritma berbasis jarak lainnya, di mana perbedaan skala antar fitur bisa menyebabkan fitur dengan rentang nilai lebih besar mendominasi hasil model.
- Oversampling dengan SMOTE pada data latih dilakukan untuk menangani class imbalance yang bisa membuat model cenderung bias ke kelas mayoritas. Dengan menyeimbangkan jumlah data minoritas, model bisa lebih sensitif dalam mendeteksi kategori miskin. SMOTE hanya diterapkan ke data training, agar evaluasi di data testing tetap mencerminkan kondisi nyata dari distribusi data asli.

## Model Development  
Tiga algoritma utama yang digunakan:
1. **K-Nearest Neighbour (KNN)**  
   Sederhana namun sensitif terhadap skala dan noise.

2. **Decision Tree**  
   Mudah dipahami dan interpretatif, dengan pengaturan `max_depth=3`.

3. **Random Forest**  
   Akurat dan stabil, tetapi kompleks dan lebih berat secara komputasi.

## Evaluation

Dalam tahap evaluasi, metrik yang digunakan adalah

- **Accuracy**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Precision**
  
$$\text{Precision} = \frac{TP}{TP + FP}$$

- **Recall**

$$\text{Recall} = \frac{TP}{TP + FN}$$

- **F1-Score**

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Penjelasan
- TP (True Positive): Jumlah data positif yang diprediksi dengan benar sebagai positif.
- TN (True Negative): Jumlah data negatif yang diprediksi dengan benar sebagai negatif.
- FP (False Positive): Jumlah data negatif yang diprediksi secara tidak benar sebagai positif (Kesalahan Tipe I).
- FN (False Negative): Jumlah data positif yang diprediksi secara tidak benar sebagai negatif (Kesalahan Tipe II).

**Hasil akhir:**

| Metric        | KNN (%) | Decision Tree (%) | Random Forest (%) |
|---------------|---------|-------------------|--------------------|
| Accuracy      | 92.22   | 97.78             | 96.67              |
| Precision     | 45.45   | 75.00             | 66.67              |
| Recall        | 83.33   | 100.00            | 100.00             |
| F1 Score      | 93.25   | 97.92             | 96.97              |

## Kesimpulan:

Berdasarkan analisis dan pemodelan yang telah dilakukan, terdapat beberapa poin penting yang dapat disimpulkan terkait klasifikasi status kemiskinan di Indonesia:

1. Pemanfaatan Machine Learning untuk Prediksi Kemiskinan
   Algoritma machine learning terbukti dapat dimanfaatkan secara optimal untuk memprediksi status kemiskinan masyarakat berdasarkan variabel sosial-ekonomi seperti persentase kemiskinan daerah, pengeluaran per kapita, IPM, dan tingkat pengangguran. Model yang dibangun mampu mengidentifikasi pola yang membedakan individu atau wilayah miskin dan tidak miskin secara akurat.


2. Algoritma Machine Learning yang Paling Efektif
   Dari tiga model yang diuji, K-Nearest Neighbour, Decision Tree, dan Random Forest, Decision Tree tampil sebagai model dengan performa paling konsisten dan unggul di berbagai metrik evaluasi seperti akurasi, precision, recall, dan F1-score. Keunggulan ini semakin menonjol setelah diterapkannya teknik oversampling SMOTE untuk menangani ketidakseimbangan data.

3. Performa Model setelah Oversampling SMOTE
  Penerapan metode SMOTE secara signifikan membantu meningkatkan kinerja model, khususnya dalam mengenali kelas minoritas (miskin). Hal ini membuktikan bahwa penyeimbangan jumlah kelas dalam data latih merupakan langkah penting untuk meningkatkan sensitivitas model terhadap kelompok yang lebih sedikit jumlahnya.

4. Fitur-Fitur yang Paling Berpengaruh
  Berdasarkan hasil eksplorasi data dan analisis korelasi, ditemukan bahwa fitur **persentase kemiskinan kota** adalah yang paling berkaitan erat dengan status kemiskinan. Selain itu, pengeluaran per kapita dan IPM juga berkontribusi besar dalam klasifikasi. Sementara itu, tingkat pengangguran menunjukkan pengaruh yang lebih lemah terhadap hasil prediksi.


## Referensi
- [BPS - Persentase Penduduk Miskin 2024](https://www.bps.go.id/id/pressrelease/2024/07/01/2370/persentase-penduduk-miskin-maret-2024-turun-menjadi-9-03-persen-.html)  
- [DeepAI - Random Forest](https://deepai.org/machine-learning-glossary-and-terms/random-forest)
- [IBM - Decision Trees](https://www.ibm.com/think/topics/decision-trees)  
- [Obeid et al. - Poverty Classification in Jordan](https://www.researchgate.net/publication/348898452_Poverty_Classification_Using_Machine_Learning_The_Case_of_Jordan)  
- [Towards Data Science - KNN Intro](https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e)  
