from flask import Flask, request, render_template
# from scripts.uas import CosineGameRecommended, GameRecommended
from uas_pengolahan import CosineGameRecommended, GameRecommended
from typing import Optional
import os
import random

app = Flask(__name__, static_folder='static', template_folder='templates')

# List untuk menyimpan gambar yang sudah dipilih
used_images = []

# Fungsi untuk mendapatkan gambar acak tanpa duplikasi
def get_random_image(folder):
    global used_images
    folder_path = os.path.join('static', folder)
    
    # Dapatkan semua gambar dalam folder
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Filter gambar yang belum dipilih
    available_images = [img for img in images if img not in used_images]
    
    if available_images:
        # Pilih gambar secara acak dari yang tersedia
        selected_image = random.choice(available_images)
        used_images.append(selected_image)  # Simpan gambar yang sudah dipilih
        return f"{folder}/{selected_image}"
    else:
        # Jika semua gambar sudah digunakan, reset list dan mulai ulang
        used_images = []
        return get_random_image(folder)

# Rute halaman utama
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Rekomendasi berdasarkan nama game
@app.route('/recommend', methods=['POST'])
def recommend():
    game_name = request.form.get('game_name', '').strip()
    recommendations = None
    error = None
    images = []

    if game_name:
        num_recommendations = 6
        result = CosineGameRecommended(game_name, num_recommendations)
        if result.empty:
            error = f"Game '{game_name}' not found in dataset."
        else:
            recommendations = result.set_index("Game").to_dict()["Cosine Similarity"]
            recommendations = dict(list(recommendations.items())[:6])
            images = [get_random_image('img/review') for _ in range(len(recommendations))]
    else:
        error = "Please enter a valid game name."

    return render_template(
        'review.html',
        recommendations=recommendations,
        error=error,
        game_name=game_name,
        images=images
    )

# Rekomendasi berdasarkan kategori
@app.route('/recommend_by_category', methods=['POST'])
def recommend_by_category():
    platform = request.form.get('platform', '').strip()
    genre = request.form.get('genre', '').strip()
    rating = request.form.get('rating', '').strip()
    error = None
    recommendations = None
    images = []

    if platform and genre and rating:
        num_recommendations = 6
        result = GameRecommended(platform, genre, rating, num_recommendations)
        if result.empty:
            error = f"No games found for Platform '{platform}', Genre '{genre}', and Rating '{rating}'."
        else:
            recommendations = result.set_index("Game").to_dict()["Cosine Similarity"]
            recommendations = dict(list(recommendations.items())[:6])
            images = [get_random_image('img/review') for _ in range(len(recommendations))]
    else:
        error = "Please complete all filters (Platform, Genre, Rating)."

    return render_template(
        'review.html',
        recommendations=recommendations,
        error=error,
        platform=platform,
        genre=genre,
        rating=rating,
        images=images
    )


if __name__ == '__main__':
    app.run(debug=True)