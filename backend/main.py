from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import joblib
import insightface
from dotenv import load_dotenv
import os
import requests

# Load environment variables from the .env file
load_dotenv()

# Access the TMDb API key
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

app = FastAPI()

# Allow CORS from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the SVM model and label encoder
clf = joblib.load("../model/svm_model.pkl")
le = joblib.load("../model/label_encoder.pkl")

# Initialize InsightFace model
face_model = insightface.app.FaceAnalysis(name='buffalo_l')
face_model.prepare(ctx_id=0, det_size=(640, 640))

# Function to fetch actor's movies and shows from TMDb API
def get_actor_movies_shows(actor_name: str):
    try:
        search_url = f"https://api.themoviedb.org/3/search/person?api_key={TMDB_API_KEY}&query={actor_name}&language=en-US"
        search_response = requests.get(search_url)
        search_response.raise_for_status()
        search_data = search_response.json()

        if not search_data.get('results'):
            return []

        actor_id = search_data['results'][0]['id']
        movies_url = f"https://api.themoviedb.org/3/person/{actor_id}/movie_credits?api_key={TMDB_API_KEY}&language=en-US"
        shows_url = f"https://api.themoviedb.org/3/person/{actor_id}/tv_credits?api_key={TMDB_API_KEY}&language=en-US"

        movies_response = requests.get(movies_url)
        shows_response = requests.get(shows_url)
        movies_response.raise_for_status()
        shows_response.raise_for_status()

        movies_data = movies_response.json()
        shows_data = shows_response.json()

        # Base URL for images
        IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

        # Process movies
        movies = [
            {
                "title": movie.get("title"),
                "poster_path": f"{IMAGE_BASE_URL}{movie.get('poster_path')}" if movie.get("poster_path") else None,
                "overview": movie.get("overview"),
                "media_type": "movie"
            }
            for movie in movies_data.get('cast', [])
            if movie.get("title") and movie.get("poster_path")
        ]

        # Process TV shows
        shows = [
            {
                "title": show.get("name"),
                "poster_path": f"{IMAGE_BASE_URL}{show.get('poster_path')}" if show.get("poster_path") else None,
                "overview": show.get("overview"),
                "media_type": "tv"
            }
            for show in shows_data.get('cast', [])
            if show.get("name") and show.get("poster_path")
        ]

        # Combine
        combined = movies + shows
        return combined

    except requests.exceptions.RequestException as e:
        print(f"TMDb API Error: {e}")
        return []

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and decode image
        img_bytes = await file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # Detect faces and get embeddings
        faces = face_model.get(img)
        if not faces:
            return {"actor": "Unknown", "movies": [], "confidence": 0.0}

        face = faces[0]
        embedding = face.embedding

        # Compute confidence
        confidence = 0.0
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba([embedding])[0]
            confidence = float(np.max(proba)) * 100

        # Apply threshold BEFORE decoding the label
        if confidence < 65:
            return {"actor": "Unknown", "movies": [], "confidence": round(confidence, 2)}

        # Decode label and fetch actor's movies and shows
        pred = clf.predict([embedding])[0]
        label = le.inverse_transform([pred])[0]
        movies_and_shows = get_actor_movies_shows(label)

        # Format the response exactly as specified
        response = {
            "actor": label,
            "movies": movies_and_shows,
            "confidence": round(confidence, 2)
        }

        return response

    except Exception as e:
        return {"actor": "Unknown", "movies": [], "confidence": 0.0}