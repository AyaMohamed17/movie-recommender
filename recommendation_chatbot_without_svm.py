from flask import Flask, request, jsonify, session
import pandas as pd
import numpy as np
import os
import re
import spacy
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from flask_session import Session

# Initialize Flask app
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load NLP models
nlp_ner = spacy.load("en_core_web_sm")
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load movie dataset
file_path = os.path.join(os.getcwd(), 'movie.csv')
if not os.path.exists(file_path):
    raise FileNotFoundError(f"⚠️ File not found: {file_path}")

merged_data = pd.read_csv(file_path)

# Ensure required columns exist
if 'genres' in merged_data.columns:
    merged_data['genre'] = merged_data['genres'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else 'Unknown')
else:
    raise KeyError("⚠️ 'genres' column not found in dataset!")

merged_data['combined_features'] = merged_data[['genre', 'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']].fillna('').agg(' '.join, axis=1)

# Convert text features into embeddings
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(merged_data['combined_features'].fillna(''))

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(merged_data.index, index=merged_data['movie_title']).drop_duplicates()

genres = merged_data['genre'].unique().tolist()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def contains_arabic(text):
    return bool(re.search(r'[؀-ۿ]', text))

def is_gibberish(text):
    words = text.split()
    if len(words) == 0:
        return True
    non_alpha_count = sum(1 for word in words if not re.search(r'[aeiouy]', word))
    return non_alpha_count / len(words) > 0.7

def extract_entities(user_input):
    doc = nlp_ner(user_input)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

def interpret_user_request(user_input):
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    inquiries = [
        "how does this work", "what can you do", "tell me about your recommendations", 
        "help", "assist me", "how can you help", "what services do you provide", 
        "explain your features", "i need help", "can you help me", "what do you offer", 
        "support", "guidance", "explain how this works"
    ]
    farewell = ["exit", "goodbye", "i don’t need recommendations anymore"]
    
    cleaned_text = clean_text(user_input)
    if cleaned_text in greetings:
        return "greeting"
    elif any(phrase in cleaned_text for phrase in inquiries):
        return "inquiry"
    elif cleaned_text in farewell:
        return "farewell"
    
    result = classifier(cleaned_text, genres)
    detected_genre = result['labels'][0]
    return detected_genre

def get_similar_movies(movie_title, num_recommendations=5):
    if movie_title not in indices:
        return {"error": "Movie title not found."}
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
    movie_indices = [i[0] for i in sim_scores]
    top_movies = merged_data.iloc[movie_indices].sort_values(by=['title_year', 'imdb_score'], ascending=[False, False]).head(num_recommendations)
    return top_movies[['movie_title', 'genre', 'director_name', 'imdb_score']].to_dict(orient='records')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    if not data or 'user_input' not in data:
        return jsonify({"error": "Please provide a user input."}), 400
    
    user_input = data['user_input'].strip()
    if contains_arabic(user_input):
        return jsonify({"bot": "Please speak in English only."})
    
    if is_gibberish(user_input):
        return jsonify({"bot": "Please enter meaningful and clear text so I can assist you."})
    
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    session['last_interaction'] = time.time()
    
    detected_intent = interpret_user_request(user_input)
    response = {}
    
    if detected_intent == "greeting":
        response["bot"] = "Hello! How can I assist you today?"
    elif detected_intent == "inquiry":
        response["bot"] = "I can recommend movies based on genre, director, actor, or ratings. Just ask!"
    elif detected_intent == "farewell":
        session.pop('chat_history', None)
        response["bot"] = "Thanks for using the Movie Recommendation System! Have a great day!"
    else:
        response["bot"] = f"I recommend movies in the {detected_intent} genre!"
        recommendations = merged_data[merged_data['genre'].str.contains(detected_intent, case=False, na=False)].sort_values(by=['title_year', 'imdb_score'], ascending=[False, False]).head(5)
        response["recommendations"] = recommendations[['movie_title', 'genre', 'imdb_score']].to_dict(orient='records')
    
    session['chat_history'].append({"user": user_input, "bot": response})
    return jsonify(response)

@app.before_request
def check_inactivity():
    if 'last_interaction' in session:
        if time.time() - session['last_interaction'] > 60:
            session['last_interaction'] = time.time()
            return jsonify({"bot": "Are you still there? Do you need any further assistance?"})

if __name__ == '__main__':
    app.run(debug=True)
