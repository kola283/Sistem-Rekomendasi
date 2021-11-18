# Laporan Proyek Machine Learning - M. Kelvin Difa

## Project Overview
Buku merupakan jendela dunia. Banyak ilmu yang bisa kita dapatkan dari membaca sebuah buku. Dan tentu saja, ada ribuan bahkan jutaan buku yang telah diterbitkan hingga saat ini. Untuk mendapatkan buku yang sesuai dengan apa yang kita cari mungkin agak sulit mengingat banyaknya buku-buku yang ada.  Maka dari itu proyek yang saya kerjakan di sini adalah sebuah sistem rekomendasi buku. Dimana proyek ini penting untuk dikerjakan karena akan membantu orang-orang untuk menemukan lebih banyak referensi buku sesuai dengan yang kita cari.

Jurnal terkait tentang sistem rekomendasi buku menggunakan metode content-based filter [jurnal](https://ejournal.undip.ac.id/index.php/jmasif/article/view/31482/17636)

## Business Understanding
### Problem Statements
1. Sistem rekomendasi seperti apa yang akan diterapkan pada kasus ini?
2. Bagaimana cara membuat model sistem rekomendasi

### Goals
1. Dengan pendekatan content-based filter maka akan dibuat sebuah sistem rekomendasi yang akan merekomendasikan beberapa buku dari pencarian sebuah buku berdasarkan kesamaan kategori yang dimiliki antara buku-buku tersebut.
2. Membuat model sistem rekomendasi dengan cara menghitung derajat kesamaan (similarity degree) antar buku dengan teknik cosine similarity.

### Solution Approach
1. Memahami dataset yang akan kita gunakan (disini menggunakan dataset 7k Books dari kaggle)
2. Melakukan data preparation pada data seperti drop kolom yang tidak digunakan dan menghilangkan missing value.
3. Algoritma sistem rekomendasi yang digunakan adalah:
**Content Based Filtering**
Content-based filtering mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi. Kelebihan dari algoritma ini adalah memiliki kemampuan untuk merekomendasikan item yang sifatnya baru bagi user karena prinsip kerjanya dengan melihat item yang memiliki kemiripan tertentu antara satu dengan yang lainnya. Sedangkan kelemahan dari algoritma ini adalah sulit untuk menghasilkan rekomendasi yang tidak terduga karena semua dipilih dan direkomendasikan berdasarkan item yang ada saja.

## Data Understanding
Dataset yang saya gunakan adalah dataset 7k Books yang berasal dari data buku pada situs GoodReads. Dataset ini berisi data buku sebanyak 6810 baris data dan 12 kolom. Linknya adalah sebagai berikut [Kaggle](https://www.kaggle.com/dylanjcastillo/7k-books-with-metadata)

Variabel-variabel pada 7k Books dataset adalah sebagai berikut.
1. isbn13           : Nomor identitas buku 13 angka
2. isbn10           : Nomor identitas buku 10 angka
3. title            : Judul buku
4. subtitle         : Jenis buku
5. authors          : Nama pengarang buku
6. categories       : Kategori atau genre buku
7. thumbnail        : Gambar cover buku
8. description      : Deskripsi cerita buku
9. published_year   : Tahun terbit buku
10. average_rating  : Rating rata-rata buku dari situs GoodReads
11. num_pages       : Banyak halaman buku
12. ratings_count   : Banyak rating yang diberikan pada buku dari situs GoodReads

Melakukan buku.info() untuk mengetahui jenis tipe data apa saja yang ada pada dataset. Dan hasilnya dataset memiliki 4 kolom berjenis float64, 1 kolom berjenis int64 dan 7 kolom berjenis object.
![image](https://github.com/kola283/gambar/blob/main/bahan/infobuku.JPG?raw=True)

## Data Preparation
Pada bagian ini tahap-tahap yang saya lakukan adalah sebagai berikut:
1. Drop kolom yang tidak digunakan. Hal ini dilakukan agar sistem hanya fokus pada fitur yang penting untuk digunakan pada pemodelan dan juga beberapa kolom memiliki missing value hampir sampai setengah dari data. Kolom yang saya gunakan hanya 3 yaitu kolom title, authors dan categories.
2. Dataset yang disiapkan akan dicek Missing Value nya. Dapat dilihat pada gambar di bawah bahwa terdapat beberapa missing value pada bagian authors dan categories.
![image](https://github.com/kola283/gambar/blob/main/bahan/missingvalue.JPG?raw=true)
Di bersihkan dengan fungsi dropna() karena missing value bisa berakibat fatal jika dibiarkan di dalam dataset.
3. Menghapus data duplikat menggunakan fungsi drop_duplicates(). Hal ini saya lakukan karena ada beberapa buku yang memiliki judul yang sama.
4. Selanjutnya,  dilakukan konversi data series menjadi list. Dalam hal ini, menggunakan fungsi tolist() dari library numpy. Hal ini dilakukan agar pengolahan data menjadi lebih mudah dan tidak terjadi error.
5. Tahap berikutnya, membuat dictionary untuk menentukan pasangan key-value pada data title, categories, dan authors yang telah kita siapkan sebelumnya. Hal ini dilakukan agar pemanggilan data menjadi lebih mudah.


## Modeling
Pada tahap ini adalah tahap pembuatan sistem rekomendasi sederhana berdasarkan kategori buku. Dengan menggunakan fitur CountVectorizer. fitur ini akan menemukan informasi sebanyak mungkin untuk menghitung derajat kemiripan. Sistem rekomendasi yang digunakan adalah content-based filtering. Model akan menyamakan tiap-tiap kategori buku yang sama untuk dijadikan rekomendasi buku. Saya hanya menggunakan fitur categories untuk perhitungan degree similarity dikarenakan referensi orang untuk membaca suatu buku adalah dengan berdasarkan kategori/genre yang diminati.

Teknik yang digunakan pada sistem rekomendasi ini adalah dengan menghitung derajat kesamaan (similarity degree) antar judul buku dengan teknik cosine similarity.
![image](https://github.com/kola283/gambar/blob/main/bahan/cvec.JPG?raw=True)
Pada tahapan ini, kita menghitung cosine similarity dataframe cvec_matrix yang kita peroleh pada tahapan sebelumnya. Dengan satu baris kode untuk memanggil fungsi cosine similarity dari library sklearn, kita telah berhasil menghitung kesamaan (similarity) antar judul buku. Kode di atas menghasilkan keluaran berupa matriks kesamaan dalam bentuk array. 

Selanjutnya adalah matriks kesamaan setiap judul buku dengan menampilkan judul buku dalam 5 sampel kolom (axis = 1) dan 10 sampel baris (axis=0).
![image](https://github.com/kola283/gambar/blob/main/bahan/metrik2.JPG?raw=True)
Dapat dilihat hasilnya, pada nilai 1.0 itu menandakan bahwa judul buku tersebut memiliki kesamaan. Sedangkan pada nilai 0.0 itu menandakan bahwa judul buku tersebut tidak memiliki kesamaan. Begitulah cara metrik ini bekerja pada model.


## Evaluation
Metrik yang akan kita gunakan pada prediksi ini adalah Precision. Precision didefinisikan dalam persamaan berikut
![image](https://github.com/kola283/gambar/blob/main/bahan/precision.png?raw=True)

Ini adalah contoh rekomendasi buku dari pencarian buku 'Murder in mesopotamia'.
![gambar](https://github.com/kola283/gambar/blob/main/bahan/rekomendasi.JPG?raw=True)
Buku 'Murder in Mesopotamia' memiliki kategori Detective and mystery stories. Lalu sistem rekomendasi juga akan merekomendasikan 5 judul buku yang memiliki kategori yang sama yaitu Detective and Mystery Stories.

Jadi perhitungan metriknya adalah sebagai berikut:
Precision = #of recommendation that are relevant/#of item we recommend.
Pada contoh rekomendasi buku di atas:
Precission = 5/5.
Jadi presisinya = 100%

