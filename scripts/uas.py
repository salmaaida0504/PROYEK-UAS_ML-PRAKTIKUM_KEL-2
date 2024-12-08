import pandas as pd
import numpy as np
from typing import Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# 1. Membaca dataset asli
df = pd.read_csv('data/Video_Games.csv')

# Set kolom 'Name' sebagai indeks untuk memudahkan pencarian
df.set_index('Name', inplace=True)
print(f"Dimensi awal dataset: {df.shape}")

# 2. Membersihkan dataset
# Menghapus missing values
print(f"Jumlah missing value sebelum dihapus:\n{df.isnull().sum()}")
df.dropna(inplace=True)
print(f"Jumlah missing value setelah dihapus:\n{df.isnull().sum()}")
print(f"Dimensi dataset setelah penghapusan missing value: {df.shape}")

# Menghapus duplikasi data
duplicate_count = df.duplicated().sum()
print(f"Jumlah duplikasi data: {duplicate_count}")

# lihat unique value pada kolom Genre
print("Value Kolom Genre : ")
print(df['Genre'].unique())

# lihat unique value pada kolom Platform
print("Value Kolom Platform : ")
print(df['Platform'].unique())

# lihat unique value pada kolom Rating
print("Value Kolom Rating : ")
print(df['Rating'].unique())

# Menghapus kolom yang tidak diperlukan
columns_to_drop = ['Year_of_Release', 'Publisher', 'Global_Sales', 'NA_Sales', 
                   'EU_Sales', 'JP_Sales', 'Other_Sales', 'Critic_Count', 'User_Count', 'Developer']
df.drop(columns=columns_to_drop, axis=1, inplace=True)

# Menghapus baris dengan rating 'RP'
df = df[df['Rating'] != 'RP']
print(f"Dimensi dataset setelah penghapusan kolom dan baris tertentu: {df.shape}")

# Menyimpan dataset yang sudah dibersihkan
df.to_csv('data/cleaned_data.csv', index=True)

# 3. Normalisasi dataset
# Membaca kembali dataset cleaned data
df_cleaned = pd.read_csv('data/cleaned_data.csv', index_col=0)

# Simpan salinan data asli sebelum preprocessing
df_original = df_cleaned.copy()

# Mengonversi kolom kategorikal menjadi kode numerik
df_cleaned['Platform'] = pd.Categorical(df_cleaned['Platform'])
df_cleaned['Genre'] = pd.Categorical(df_cleaned['Genre'])
df_cleaned['Rating'] = pd.Categorical(df_cleaned['Rating'])
df_cleaned['Platform'] = df_cleaned['Platform'].cat.codes
df_cleaned['Genre'] = df_cleaned['Genre'].cat.codes
df_cleaned['Rating'] = df_cleaned['Rating'].cat.codes

# Normalisasi menggunakan MinMaxScaler
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_cleaned), 
                             columns=df_cleaned.columns, 
                             index=df_cleaned.index)
print("Contoh data setelah normalisasi:")
print(df_normalized.head())

# Menyimpan data yang sudah dinormalisasi
df_normalized.to_csv('data/normalized_data.csv', index=True)

# 4. Membuat matriks cosine similarity
cosine_sim = cosine_similarity(df_normalized)
cosine_sim_df = pd.DataFrame(cosine_sim, index=df_normalized.index, columns=df_normalized.index)

print("Contoh data head:")
print(cosine_sim_df.head())  # Menampilkan 5 baris pertama
print("Contoh data kolom:")
print(cosine_sim_df.columns)  # Menampilkan kolom atau indeks
print("Contoh data index:")
print(cosine_sim_df.index)  # Menampilkan indeks DataFrame

def CosineGameRecommended(gamename, recommended_games=5):
    # Validasi keberadaan gamename di index
    if gamename not in cosine_sim_df.index:
        return pd.DataFrame(columns=["Game", "Cosine Similarity"])  # Kembalikan DataFrame kosong
    
    # Ambil data similarity untuk gamename
    data = cosine_sim_df.loc[gamename]
    if isinstance(data, pd.DataFrame):
        data = data.iloc[0]

    if not isinstance(data, pd.Series):
        return pd.DataFrame(columns=["Game", "Cosine Similarity"])  # Kembalikan DataFrame kosong jika tipe data salah

    # Urutkan similarity scores
    sim_scores = data.sort_values(ascending=False)[1:recommended_games + 1]
    result_df = pd.DataFrame({
        "Game": sim_scores.index,
        "Cosine Similarity": sim_scores.values
    })

    return result_df


# # Uji fungsi rekomendasi berbasis cosine similarity dg parameter nama game
# recommendations = CosineGameRecommended('Mario Kart Wii')
# print(recommendations)

# 6. Fungsi rekomendasi berdasarkan kategori (platform, genre, rating)
def GameRecommended(platform: str, genre: str, rating: str, recommended_games: int = 5):
    # Validasi kategori dari data asli
    platform_categories = pd.Categorical(df_original['Platform']).categories
    genre_categories = pd.Categorical(df_original['Genre']).categories 
    rating_categories = pd.Categorical(df_original['Rating']).categories

    if platform not in platform_categories:
        print(f"Platform '{platform}' tidak ditemukan dalam dataset.")
        return pd.DataFrame(columns=["Game", "Cosine Similarity"])

    if genre not in genre_categories:
        print(f"Genre '{genre}' tidak ditemukan dalam dataset.")
        return pd.DataFrame(columns=["Game", "Cosine Similarity"])

    if rating not in rating_categories:
        print(f"Rating '{rating}' tidak ditemukan dalam dataset.")
        return pd.DataFrame(columns=["Game", "Cosine Similarity"])

    # Konversi input ke kode numerik
    platform_code = platform_categories.get_loc(platform)
    genre_code = genre_categories.get_loc(genre)
    rating_code = rating_categories.get_loc(rating)

    # Filter game sesuai input
    filtered_games = df_cleaned[(df_cleaned['Platform'] == platform_code) & 
                                (df_cleaned['Genre'] == genre_code) & 
                                (df_cleaned['Rating'] == rating_code)]

    if filtered_games.empty:
        print(f"Tidak ada game yang ditemukan dengan kombinasi Platform '{platform}', Genre '{genre}', dan Rating '{rating}'.")
        return pd.DataFrame(columns=["Game", "Cosine Similarity"])

    # Hitung similarity untuk game yang difilter
    filtered_indices = filtered_games.index
    cosine_sim_filtered = cosine_sim_df.loc[filtered_indices, filtered_indices]
    recommended_games_list = cosine_sim_filtered.mean(axis=0).sort_values(ascending=False)[1:recommended_games + 1]

    # Membuat dataframe rekomendasi
    recommended = pd.DataFrame({"Game": recommended_games_list.index, "Cosine Similarity": recommended_games_list.values})
    return recommended.reset_index(drop=True)

# # Uji fungsi rekomendasi berbasis kategori (platform, genre, rating)
# recommendations_game = GameRecommended('PS3', 'Action', 'M', 5)
# print(recommendations_game)
