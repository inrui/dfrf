'''
Data Analysis
'''
# dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

# Mengimpor library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Mengimpor dataset
trainset = pd.read_csv('houseprice_train.csv')

'''
Melakukan pengecekan terhadap:
    1. Data yang hilang (NA)
    2. Kolom-kolom angka (numerik)
    3. Distribusi dari setiap kolom numerik
    4. Outlier
    5. Kolom-kolom kategori dan jumlah kategorinya
    6. Hubungan antara variabel independen dan dependen
'''

# Mendeteksi kolom yang memiliki data NA
col_na = [col for col in trainset.columns if trainset[col].isnull().sum() > 0]
'''
Membedah satu demi satu
trainset.columns <- melihat kolom apa saja yang ada di variabel dataku
trainset['LotFrontage'].isnull() <- akan memberikan nilai 1 jika ada baris yang berisi NA untuk kolom tersebut
trainset['LotFrontage'].isnull().sum() <- akan menjumlahkan nilai True (1) akibat NA. 
Otomatis jika >0 maka ada NA di kolom tersebut
'''

# Menghitung persentase dari kolom-kolom yang berisi NA
trainset[col_na].isnull().mean()*100


# Membuat visualisasi untuk setiap kolom berisi NA (kolom_na)
def analisis_data_na(data, col2):
    data = trainset.copy()
    # Mengecek setiap variabel jika ada NA maka 1, jika tidak maka 0
    data[col2] = np.where(data[col2].isnull(), 1, 0)
    # Sekarang kita memiliki nilai binary (0=ada data, 1=NA)
    # Sekarang bandingkan nilai median (tidak sensitif terhadap outlier) dari 'SalePrice' terhadap 2 nilai binary kolom ini
    # Pengelompokan terhadap nama col, tapi perhitungan agregasi terhadap SalePrice
    data.groupby(col2)['SalePrice'].median().plot.bar() # silakan mencoba menggunakan mean
    plt.title(col2)
    plt.tight_layout()
    plt.show()


# Membuat for loop untuk plotting kolom_na
limit = len(col_na)
i = 1
for col in col_na:
    i+=1
    analisis_data_na(trainset, col)
    if i <= limit: plt.figure() 
# Ternyata harga SalePrice saat nilainya kosong (NA) berbeda dengan SalePrice saat nilainya tidak kosong
# Ini akan jadi pertimbangan untuk feature engineering

# Menganalisis kolom-kolom numerik
col_num = [col for col in trainset.columns if trainset[col].dtypes != 'O'] #'O' adalah Pandas Object = string

# Visualisasi data kolom_numerik
num = trainset[col_num]
# Kita lihat bahwa kolom 'Id' tidak begitu berguna bagi kita

'''
Variabel yang berhubungan dengan waktu:
    YearBuilt = kapan rumah dibangun
    YearRemodAdd = kapan rumah direnov
    GarageYrBlt = kapan garasinya dibangun
    YrSold = kapan rumahnya dijual
'''

# Mmebuat kolom_tahun yang terdiri dari variabel Tahun (Year atau Yr)
col_yr = [col for col in col_num if 'Yr' in col or 'Year' in col]

# Kita visualisasikan perubahan harga dari mulai dibangun sampai terjual
trainset.groupby('YrSold')['SalePrice'].median().plot()
plt.ylabel('Nilai Median Harga Jual')
plt.title('Perubahan Harga')
plt.tight_layout()
# Ternyata harganya turun (tidak wajar), perlu investigasi lebih lanjut

# Analisis antara variabel 'Year' dan harga rumah
def analisis_data_yr(data, col2):
    data = trainset.copy()
    # Melihat perbedaan antara kolom tahun yang dimaksud dengan tahun penjualan rumah
    data[col2] = data['YrSold'] - data[col2]
    plt.scatter(data[col2], data['SalePrice'])
    plt.ylabel('SalePrice')
    plt.xlabel(col2)
    plt.tight_layout()
    plt.show()

# Membuat for loop untuk plotting kolom_tahun
limit = len(col_yr)
i = 1
for col in col_yr:
    if col !='YrSold':
        i+=1
        analisis_data_yr(trainset, col)
        if i < limit: plt.figure() 
# Ternyata semakin tua fitur 'YearBulit', 'YearRemodAdd', dan 'GarageYrBlt', semakin turun harganya
# Artinya semakin besar jarak antara ketiga waktu ini dengan waktu penjualan, maka semakin turun harganya
# mungkin karena tampilannya jadul atau butuh banyak biaya renovasi, sehingga harga jualnya rendah.

# Analisis variabel discrete/diskrit (skala hitung)
col_discrete = [col for col in col_num if len(trainset[col].unique()) <= 15 and col not in col_yr+['Id']]
descrete = trainset[col_discrete]

# Analisis data diskrit dnegan harga rumah (SalePrice)
def analisis_data_discrete(data, col2):
    data = trainset.copy()
    data.groupby(col)['SalePrice'].median().plot.bar()
    plt.title(col)
    plt.ylabel('Nilai Median Harga Jual')
    plt.tight_layout()
    plt.show()

# Membuat for loop untuk plotting kolom_diskrit
limit = len(col_discrete)
i = 1
for col in col_discrete:
    i+=1
    analisis_data_discrete(trainset, col)
    if i <= limit: plt.figure() 
# Ternyata ada beberapa kolom yang memiliki hubungan kuat, misal kolom 'OverallQual', semakin tinggi kualitas maka semakin tinggi pula harga jualnya
# Ini kita jadikan catatan untuk proses feature engineering

# Variabel kontinu
col_continu = [col for col in col_num if col not in col_discrete + col_yr + ['Id']]
kontinu = trainset[col_continu]
# Mengecek len dari var kontinu
for i in kontinu:
    print(len(trainset[i].unique()))

# Analisis data kontinu dengan harga rumah (SalePrice)
def analisis_data_kontinu(data, col2):
    data = trainset.copy()
    data[col2].hist(bins=30) # menggabungkan len(var kontinu) ke dalam 3 bins agar mudah divisualisasikan
    plt.ylabel('Jumlah rumah')
    plt.xlabel(col2)
    plt.title(col2)
    plt.tight_layout()
    plt.show()

# Membuat for loop untuk plotting kolom_kontinu
limit = len(col_continu)
i = 1
for col in col_continu:
    i+=1
    analisis_data_kontinu(trainset, col)
    if i <= limit: plt.figure() 
# Terlihat bahwa hampir semua datanya tidak berdistribusi normal (skewed right)
# Nanti kita akan lakukan logtransform agar datanya lebih mendekati normal

# Melakukan proses logtransform
def analisis_logtransform(data, col2):
    data = trainset.copy()
    # CATATAN: logaritmik tidak memperhitungkan data 0 dan negatif, jadi harus di skip kolom yg memiliki 0 dan -
    if any(data[col2] <= 0):
        pass
    else:
        data[col2] = np.log(data[col2]) # Proses logtransformation
        data[col2].hist(bins=30)
        plt.ylabel('Jumlah rumah')
        plt.xlabel(col2)
        plt.title(col2)
        plt.tight_layout()
        plt.show()

# Menentukan batas sekaligus variabel di kolom_kontinu yang tidak memiliki 0 dan negatif       
limit = 0
col_continu_log = []
for col in col_continu:
    if any(trainset[col] <= 0):
        pass
    else:
        limit+=1
        col_continu_log.append(col)
continu_log = trainset[col_continu_log]

# Membuat for loop untuk plotting kolom_kontinu (dengan logtransform)
i = 1
for col in col_continu_log:
    i+=1
    analisis_logtransform(trainset, col)
    if i<= limit: plt.figure() 
# Sekarang sudah lebih tampak normal

# Sekarang kita analisis hubungan antara SalePrice dengan variabel yang sudha ditransformasi
def analisis_logtransform_scatter(data, col2):
    data = trainset.copy()
    if any(data[col2] <= 0):
        pass
    else:
        data[col2] = np.log(data[col2])
        # Logtransform SalePrice
        data['SalePrice'] = np.log(data['SalePrice'])
        # plot
        plt.scatter(data[col2], data['SalePrice'])
        plt.ylabel('Harga Rumah')
        plt.xlabel(col2)
        plt.tight_layout()
        plt.show()

i = 1
for col in col_continu:
    if col != 'SalePrice':
        i+=1
        analisis_logtransform_scatter(trainset, col)
        if i< limit: plt.figure() 

# Analisis outlier
def analisis_outlier(data, col2):
    data = trainset.copy()
    if any(data[col2] <= 0):
        pass
    else:
        data[col2] = np.log(data[col2])
        data.boxplot(column=col2)
        plt.title(col2)
        plt.ylabel(col2)
        plt.tight_layout()
        plt.show()

# Membuat for loop untuk plotting kolom_kontinu (dengan logtransform untuk outliers)
i = 1
for col in col_continu_log:
    i+=1
    analisis_outlier(trainset, col)
    if i<= limit: plt.figure() 
# Ternyata ada banyak outliers di var kontinu kita yg sudha di logtransform. 
# Perlu dipertimbangkan apakah membuang outlier bisa meningkatkan performa modle atau tidak.

# Variabel kategori (nominal)
col_cat = [col for col in trainset.columns if trainset[col].dtypes == 'O']
cat = trainset[col_cat]

# Mengecek berapa banyak kategori yang ada di setiap variabel
cat.nunique()

# Analisis variabel yang jarang
def analisis_var_jarang(data, col2, persentase):
    data = trainset.copy()
    # Menentukan persetnase setiap kategori
    isi = data.groupby(col)['SalePrice'].count() / len(data)
    # Mengembalikan variabel yang di bawah persentase kelangkaan yang kita tentukan
    return isi[isi < persentase]

# For loop untuk variabel jarang
for col in col_cat:
    print(analisis_var_jarang(trainset, col, 0.01),'\n')

# For loop untuk plotting
i = 1
limit = len(col_cat)
for col in col_cat:
    i+=1
    analisis_data_discrete(trainset, col)
    if i<= limit: plt.figure() 
# Ternyata setiap kategori memiliki hubungan terhadap SalePrice