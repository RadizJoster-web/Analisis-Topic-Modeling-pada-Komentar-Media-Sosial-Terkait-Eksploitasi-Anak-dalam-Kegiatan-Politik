# Analisis NLP Komentar Media Sosial Berbahasa Indonesia

Pipeline analisis teks berbahasa Indonesia yang mencakup **preprocessing**, **topic modeling (LDA)**, **text clustering (K-Means & FTC)**, dan **Named Entity Recognition (NER) berbasis aturan (rule-based)**. Studi kasus utama menggunakan dataset komentar media sosial terkait program "Exploitasi anak di bawah umur untuk kepentingan politik", dengan pengujian tambahan modul NER pada dataset ulasan aplikasi Gojek.

---

## Daftar Isi

1. [Ringkasan Project](#ringkasan-project)
2. [Arsitektur Pipeline](#arsitektur-pipeline)
3. [Dataset](#dataset)
4. [Instalasi](#instalasi)
5. [Cara Menjalankan](#cara-menjalankan)
6. [Detail Setiap Tahap](#detail-setiap-tahap)
7. [Ringkasan Hasil](#ringkasan-hasil)
8. [Keterbatasan & Catatan Penting](#keterbatasan--catatan-penting)
9. [Saran Pengembangan](#saran-pengembangan)

---

## Ringkasan Project

Notebook ini melakukan analisis teks tidak terstruktur (komentar media sosial) untuk menjawab tiga pertanyaan riset:

| Teknik | Tujuan |
|---|---|
| **LDA (Latent Dirichlet Allocation)** dengan bobot TF-IDF | Menemukan topik-topik laten yang muncul dalam komentar |
| **Clustering** (K-Means & metode custom **FTC**) | Mengelompokkan komentar berdasarkan kemiripan konten |
| **NER rule-based** | Mengekstrak entitas bernama (orang, lokasi, organisasi) dari teks bebas |

Seluruh pipeline dibangun untuk teks **bahasa Indonesia informal** (komentar media sosial), sehingga mencakup penanganan slang, stopword tambahan, dan stemming menggunakan Sastrawi.

---

## Arsitektur Pipeline

```
Data Mentah (CSV)
      │
      ▼
[1] Text Cleaning ─── lowercase, hapus mention/simbol, normalisasi huruf berulang
      │
      ▼
[2] Tokenisasi + Stopword Removal (NLTK + daftar custom)
      │
      ▼
[3] Stemming (Sastrawi)
      │
      ├──▶ [4] Topic Modeling: LDA + TF-IDF ──▶ Coherence Score (c_v)
      │
      ├──▶ [5] Clustering: TF-IDF + K-Means ──▶ Silhouette Score
      │
      ├──▶ [6] Clustering custom: FTC (Tau-Entropy Splitting)
      │
      └──▶ [7] NER Rule-Based (gazetteer + morfologi + aturan konteks)
```

---

## Dataset

Notebook ini menggunakan **dua dataset terpisah**:

| Dataset | Kolom Penting | Digunakan untuk | Path (di notebook) |
|---|---|---|---|
| Komentar sosial media (Exploitasi anak) | `comments` | Preprocessing, LDA, K-Means, FTC, NER | `/content/drive/MyDrive/Machine Learning/UAS/dataset.csv` (829 baris) |
| Ulasan aplikasi Gojek | `content` | Uji coba tambahan modul NER | `/content/gojek review.csv` (194.043 baris) |

> ⚠️ Notebook ini awalnya dibuat di **Google Colab** (ada `drive.mount(...)`). Jika ingin dijalankan secara lokal, path di atas **harus disesuaikan** dan baris `drive.mount()` dihapus/dilewati.

---

## Instalasi

```bash
pip install pandas numpy regex nltk gensim matplotlib seaborn scikit-learn Sastrawi
```

Lalu unduh resource NLTK yang dibutuhkan (sekali saja):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
```

**Dependencies utama:**
- `pandas`, `numpy` — manipulasi data
- `nltk` — tokenisasi & stopword bahasa Indonesia
- `Sastrawi` — stemming bahasa Indonesia
- `gensim` — LDA, dictionary/corpus, coherence score
- `scikit-learn` — TF-IDF, K-Means, Silhouette Score
- `matplotlib`, `seaborn` — visualisasi

---

## Cara Menjalankan

1. Siapkan file `dataset.csv` (kolom wajib: `comments`) di direktori kerja.
2. Sesuaikan path pembacaan CSV pada sel *"Tahap 1: Memuat Data & Preprocessing"* — ganti path Google Drive dengan path lokal.
3. Jalankan notebook **secara berurutan dari atas ke bawah** (lihat bagian [Keterbatasan](#keterbatasan--catatan-penting) — beberapa sel memiliki dependensi eksekusi yang implisit).
4. Untuk bagian NER, siapkan file `gojek review.csv` (kolom wajib: `content`) jika ingin menjalankan pengujian tambahan tersebut, atau lewati sel tersebut jika hanya ingin menjalankan NER pada dataset utama.

---

## Detail Setiap Tahap

### 1. Preprocessing (`clean_indonesian_text`)
Membersihkan teks dengan urutan:
- Case folding (huruf kecil semua)
- Menghapus mention (`@username`)
- Menghapus karakter non-alfabet
- Normalisasi huruf berulang (`"insaneeee"` → `"insanee"`)
- Tokenisasi (`word_tokenize`) + stopword removal (stopword bawaan NLTK Indonesia + tambahan custom: `yg`, `dg`, `klo`, `aja`, dst.)
- Filter token dengan panjang ≤ 2 karakter
- Stemming menggunakan **Sastrawi**

### 2. Topic Modeling — LDA + TF-IDF
- Dictionary & corpus dibangun dari `stemmed_tokens` menggunakan `gensim.corpora.Dictionary`
- Corpus BoW ditransformasi ke bobot **TF-IDF** sebelum dilatih ke `LdaModel` (3 topik, `passes=15`)
- Evaluasi menggunakan **Coherence Score (c_v)**

**Hasil topik yang ditemukan (top kata):**

| Topik | Kata Kunci Dominan |
|---|---|
| 0 | guru, suruh, malu, mbg |
| 1 | spanduk, bayar, kpai, anak, kreatif |
| 2 | miris, anak, banget, mbg |

### 3. Text Clustering — K-Means
- Vektorisasi dengan `TfidfVectorizer` dari `sklearn`
- `KMeans` dengan `n_clusters=3`
- Evaluasi menggunakan **Silhouette Score**

### 4. Text Clustering — FTC (Fuzzy/Tau-Entropy Clustering Custom)
Implementasi custom yang membagi (split) cluster secara **rekursif** berdasarkan **entropy tertinggi** (dihitung dari distribusi cosine similarity antar dokumen dan centroid-nya), menggunakan K-Means (`k=2`) sebagai mekanisme pemecah cluster di setiap iterasi.

### 5. Named Entity Recognition (NER) — Rule-Based
Modul NER dibangun dari nol (tanpa library seperti spaCy) dengan pendekatan **gazetteer + aturan morfologi + konteks**, terdiri dari:

- **`Entity`** dan **`Token`** — struktur data (dataclass) untuk menyimpan token & entitas hasil ekstraksi
- **Kamus fitur** (`CONTEXTUAL_FEATURES`, `POS_DICTIONARY`) — daftar kata pemicu untuk kategori seperti gelar (`PPRE`: "prof.", "dr.", "pak"), organisasi (`OPRE`: "pt.", "universitas", "kementerian"), preposisi lokasi (`LOPP`: "di", "ke", "dari"), dsb.
- **`assign_morphological`** — mendeteksi bentuk kata (UpperCase / TitleCase / LowerCase)
- **`assign_contextual`** — mencocokkan token dengan kamus fitur
- **`apply_rules`** — menerapkan aturan if-then, contoh:
  - Kata setelah gelar (prof./pak/dr.) dan berbentuk TitleCase/UpperCase → `PERSON`
  - Kata setelah preposisi lokasi dan berbentuk TitleCase → `LOC`
  - Kata dari daftar `OPRE` → `ORG`
  - Token yang diawali `@` → `PERSON` (mention di media sosial)
- **`extract_entities`** — menggabungkan token-token berurutan dengan label sama menjadi satu entitas utuh

Label entitas yang didukung: **PERSON, LOC, ORG** (5 label disebutkan dalam catatan riset, namun implementasi saat ini secara eksplisit hanya menghasilkan 3 label tersebut pada aturan `apply_rules`).

---

## Ringkasan Hasil

| Metrik | Nilai | Interpretasi |
|---|---|---|
| Coherence Score LDA (TF-IDF, `texts=stemmed_tokens`) | **0.5128** | Cukup baik secara semantik untuk data komentar informal |
| Coherence Score LDA (`texts=tokens`, tanpa stemming) | **nan** | Terjadi *runtime warning* (divide by zero) — lihat [Keterbatasan](#keterbatasan--catatan-penting) |
| Silhouette Score K-Means (k=3) | **0.0707** | Cluster mulai terbentuk namun overlap-nya masih cukup tinggi |
| Distribusi Cluster FTC | Cluster 0: 3.76% • Cluster 1: 89.45% • Cluster 2: 6.79% | Sangat tidak seimbang — mayoritas dokumen jatuh ke satu cluster besar |

**Contoh isi cluster FTC:**
- Cluster 0 → seputar keluhan "miris" terhadap guru
- Cluster 1 → cluster besar/umum (topik campuran)
- Cluster 2 → seputar "spanduk"/klaim AI-generated

---

## Keterbatasan & Catatan Penting

Sebagai catatan objektif dari hasil peninjauan kode (bukan untuk mengurangi nilai kerja yang sudah dilakukan, tapi agar pengguna lain tidak bingung saat menjalankan ulang):

1. **Sel dengan dependensi tersembunyi.** Sel yang memproses eliminasi klaster (bagian *"PROSES ELIMINASI ITERATIF KANDIDAT KLASTER"*) memanggil variabel `cluster_candidate` dan fungsi `calculate_entropy_overlap` yang **tidak didefinisikan di sel manapun dalam notebook ini**. Kemungkinan sel yang mendefinisikannya sudah terhapus/tidak tersimpan. Notebook baru tidak akan bisa menjalankan sel ini dari awal (fresh run) tanpa menambahkan definisi tersebut kembali.
2. **Coherence Score `nan`.** Pada evaluasi kedua (`texts=df['tokens']`), model dilatih dari `stemmed_tokens` tetapi dievaluasi dengan `tokens` (belum di-stem) — ketidakcocokan vocabulary ini menyebabkan pembagian oleh nol saat menghitung coherence. Untuk hasil yang valid, gunakan `texts` yang sama dengan data yang dipakai melatih model (`stemmed_tokens`).
3. **Ketergantungan pada Google Colab.** Sel pertama (`drive.mount`) dan path dataset menggunakan struktur folder Google Drive, sehingga notebook tidak langsung bisa dijalankan di lingkungan lokal/Jupyter biasa tanpa penyesuaian path.
4. **Silhouette Score rendah (0.0707)** mengindikasikan cluster K-Means saat ini tumpang tindih signifikan — wajar untuk data teks pendek dan informal, namun perlu dicatat sebagai batas kualitas model, bukan bug.
5. **5 label entitas disebutkan, namun hanya 3 yang diimplementasikan** (`PERSON`, `LOC`, `ORG`) di dalam `apply_rules`. Jika riset menyebutkan 5 label, dua label tambahan (kemungkinan terkait `PTIT`/jabatan atau `OPOS`) perlu ditambahkan aturannya secara eksplisit di `apply_rules` agar benar-benar menghasilkan label tersebut sebagai `ne_label`.

---

## Saran Pengembangan

- Tambahkan kembali definisi `cluster_candidate` dan `calculate_entropy_overlap` (atau dokumentasikan asalnya) agar seluruh notebook bisa dijalankan ulang dari awal secara mandiri (reproducible).
- Simpan hasil antara (`processed_dataset.csv`, hasil cluster, hasil NER) ke file terpisah agar setiap tahap bisa dijalankan independen tanpa mengulang preprocessing dari awal.
- Pertimbangkan evaluasi NER secara kuantitatif (precision/recall) terhadap data yang sudah dilabeli manual (gold standard), karena saat ini evaluasi hanya bersifat kualitatif (inspeksi log manual).
- Refactor path Google Colab menjadi argumen/config agar portable ke lingkungan lain (lokal, server, Kaggle, dll).
- Tambahkan docstring & type hints pada fungsi-fungsi utama (`ftc_clustering_real_data`, `indo_rule_based_ner`, dll.) untuk memudahkan kontributor lain membaca kode.

---

## Struktur Output

| File | Deskripsi |
|---|---|
| `processed_dataset.csv` | Dataset hasil preprocessing (cleaning + tokenisasi + stemming) |
