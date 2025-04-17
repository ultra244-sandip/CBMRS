import os
import numpy as np
import time
import pandas as pd
import cohere
import subprocess
import re
from rapidfuzz import process, fuzz
import unicodedata



from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from flask_session import Session
from dotenv import load_dotenv

# Import authentication functions from Credentials.py
from Credentials import register_user, login_user

# Load environment variables (.env must contain COHERE_API_KEY, etc.)
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.secret_key = "your_secret_key"
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Initialize Cohere client
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY environment variable not set")
co = cohere.Client(cohere_api_key)

# ------------------------------------------------
# Helper Normalization Functions
# ------------------------------------------------
def normalize_text(text):
    """
    Normalize Unicode characters, remove punctuation, and convert to lower case.
    This ensures that Bengali (and other non‑Latin languages) are consistently processed.
    """
    # Normalize Unicode characters (NFKC form)
    text = unicodedata.normalize('NFKC', text)
    # Remove punctuation; \w is Unicode‐aware in Python 3
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

def normalize_album(album):
    """
    Remove text within parentheses, insert a space between letters and numbers,
    then strip and lower-case the text.
    """
    album = re.sub(r'\(.*?\)', '', album)
    # Insert a space between alphabetical characters and digits, if not already present.
    album = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', album)
    return normalize_text(album)

# ------------------------------------------------
# Load Music CSV Data and Precompute Reference Lists
# ------------------------------------------------
# CSV columns: track_id, song_name, artist_name, album_movie_name, mood_label, language
music_df = pd.read_csv('music.csv')

# Clean and normalize key columns
music_df['album_movie_name'] = music_df['album_movie_name'].astype(str).apply(normalize_album)
music_df['song_name'] = music_df['song_name'].astype(str).apply(normalize_text)
music_df['artist_name'] = music_df['artist_name'].astype(str).apply(normalize_text)
music_df['mood_label'] = music_df['mood_label'].astype(str).apply(normalize_text)
music_df['language'] = music_df['language'].astype(str).apply(normalize_text)

print("DataFrame shape:", music_df.shape)
print("Unique album names:", music_df['album_movie_name'].unique())

all_moods      = music_df['mood_label'].unique().tolist()
all_artists    = music_df['artist_name'].unique().tolist()
all_languages  = music_df['language'].unique().tolist()
all_albums     = music_df['album_movie_name'].unique().tolist()
all_song_names = music_df['song_name'].unique().tolist()

# ------------------------------------------------
# Helper Functions
# ------------------------------------------------
def fuzzy_match_song(user_input, choices, score_cutoff=70):
    """
    Try a variety of fuzzy matching scorers and return the best match.
    """
    query = user_input.lower()
    # Try different scorers.
    best_candidates = []
    for scorer in [fuzz.ratio, fuzz.partial_ratio, fuzz.token_set_ratio]:
        candidate = process.extractOne(query, choices, scorer=scorer, score_cutoff=score_cutoff)
        if candidate:
            best_candidates.append(candidate)
    if not best_candidates:
        return None
    # Return candidate with the highest score.
    return max(best_candidates, key=lambda x: x[1])

def extract_entities(user_input):
    # Normalize the text and insert spaces between letters and digits.
    query = normalize_text(user_input)
    query = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', query)
    tokens = query.split()

    # Extract mood and language based on tokens.
    found_mood = next((m for m in all_moods if m in tokens), None)
    found_language = next((l for l in all_languages if l in tokens), None)

    # ----- Album Extraction -----
    album_requested = any(keyword in user_input.lower() for keyword in ['album', 'movie'])
    found_album = None
    if album_requested:
        # First: try to extract candidate tokens by filtering out common album-related stopwords.
        album_stopwords = {"play", "movie", "album", "songs", "song", "music", "some", "from"}
        candidate_tokens = [tok for tok in tokens if tok not in album_stopwords]
        candidate_string = " ".join(candidate_tokens).strip()
        # For example, "play bolidan movie songs" becomes "bolidan"
        if candidate_string:
            # Lower cutoff to 60 to allow for slight misspellings.
            best_album = process.extractOne(candidate_string, all_albums, scorer=fuzz.ratio, score_cutoff=60)
            if best_album:
                found_album = best_album[0]
        # Fallback: if no candidate found from tokens, try fuzzy matching on the full query.
        if not found_album:
            best_album = process.extractOne(query, all_albums, scorer=fuzz.token_set_ratio, score_cutoff=60)
            if best_album:
                found_album = best_album[0]
    else:
        # If not an album request, do a more strict matching.
        best_album = process.extractOne(query, all_albums, scorer=fuzz.ratio, score_cutoff=95)
        if best_album:
            found_album = best_album[0]
    print(f"Extracted album: {found_album}")

    # ----- Artist Extraction -----
    found_artist = None
    match_artist = re.search(r'(?:songs\s+(?:by|of)\s+)(.+)', user_input.lower())
    if match_artist:
        artist_candidate = match_artist.group(1).strip()
        # First, try to find any artist whose name contains the candidate as a substring.
        candidate_matches = [artist for artist in all_artists if artist_candidate in artist.lower()]
        if candidate_matches:
            # Use partial ratio fuzzy matching with a lowered cutoff since the candidate might be partial.
            best_artist = process.extractOne(artist_candidate, candidate_matches, scorer=fuzz.partial_ratio, score_cutoff=70)
            if best_artist:
                found_artist = best_artist[0]
        else:
            # Fallback to strict fuzzy matching if no substring match is found.
            best_artist = process.extractOne(artist_candidate, all_artists, scorer=fuzz.ratio, score_cutoff=80)
            if best_artist:
                found_artist = best_artist[0]
    # Additional fallback: use regex word-boundary search in case the above didn't find a match.
    if not found_artist:
        for artist in all_artists:
            pattern = r'\b' + re.escape(artist) + r'\b'
            if re.search(pattern, query):
                found_artist = artist
                break
    print(f"Extracted artist: {found_artist}")

    # ----- Song Extraction -----
    found_song = None
    if not found_artist:
        command_stopwords = {"play", "song", "songs", "music", "some", "by", "of"}
        filtered_tokens = [tok for tok in tokens if tok not in command_stopwords]
        # If both mood and language are detected and the remaining tokens are exactly these two,
        # then treat it as a generic query.
        if found_mood and found_language and len(filtered_tokens) == 2 and set(filtered_tokens) == {found_mood, found_language}:
            found_song = None
        elif filtered_tokens and not (
            (found_language and filtered_tokens == [found_language]) or
            (found_mood and filtered_tokens == [found_mood])
        ):
            # Direct substring check.
            for s in all_song_names:
                if s in query:
                    found_song = s
                    break
            # If no match, attempt fuzzy matching.
            if not found_song:
                cutoff = 75
                m_song = fuzzy_match_song(query, all_song_names, score_cutoff=cutoff)
                if m_song:
                    found_song = m_song[0]
    print(f"Extracted song: {found_song}")

    # IMPORTANT: If the query is album-related, ignore any song match.
    if album_requested:
        found_song = None

    return found_artist, found_mood, found_language, found_album, found_song, album_requested

def construct_prompt(user_input, artist, mood, language, matched_songs):
    header = f"You are a friendly music chatbot. The user said: '{user_input}'.\n"
    details = ""
    if language:
        details += f"The user requested {language.upper()} songs.\n"
    if artist:
        details += f"The user seems interested in songs by {artist.title()}.\n"
    if mood:
        details += f"The mood of the query appears to be '{mood}'.\n"
    if matched_songs:
        details += "Based on our dataset, here are some suggested songs:\n"
        # Include only the top 5 suggestions to reduce token count.
        for song in matched_songs[:2]:
            details += f"- '{song['song_name']}' by {song['artist_name']}\n"
    details += "Respond in a friendly tone, mentioning only the song names and artists from the list above, and ask if they would like to listen to any of these songs."
    return header + details

def get_stream_url(song_name, artist_name=None):
    query = song_name
    if artist_name:
        query += " " + artist_name
    try:
        search_command = [
            "yt-dlp",
            f"ytsearch1:{query}",
            "--get-url",
            "-f", "bestaudio",
            "--no-playlist"
        ]
        result = subprocess.run(search_command, capture_output=True, text=True, check=True)
        url = result.stdout.strip()
        return url if url else None
    except subprocess.CalledProcessError as e:
        app.logger.error(f"Error fetching stream URL for query '{query}': {e}")
        return None

import requests
from flask import request, Response, abort

@app.route('/proxy_audio')
def proxy_audio():
    url = request.args.get('url')
    if not url:
        print("DEBUG: No URL provided")
        abort(400, "No URL provided.")
    
    print("DEBUG: Proxying request for URL:", url)
    
    # Prepare headers to mimic browser behavior:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/110.0.0.0 Safari/537.36",
        "Referer": "https://www.youtube.com/",
        "Accept": "audio/webm, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    
    # Get Range header from the client request if available
    range_header = request.headers.get('Range')
    if range_header:
        headers['Range'] = range_header
        print("DEBUG: Range header detected:", range_header)
    
    try:
        remote_response = requests.get(url, headers=headers, stream=True)
        print("DEBUG: Received response from origin:", remote_response.status_code)
    except Exception as e:
        print("DEBUG: Error fetching URL:", str(e))
        abort(500, f"Error fetching audio URL: {e}")
    
    if remote_response.status_code == 403:
        print("DEBUG: Access forbidden for URL:", url)
        abort(403, "Access forbidden. Google may be rejecting non-browser-like requests.")
    
    content_type = remote_response.headers.get('Content-Type', 'audio/webm')
    status = remote_response.status_code

    def generate():
        try:
            for chunk in remote_response.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        except Exception as e:
            print("DEBUG: Streaming error:", str(e))
    
    response_headers = {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': content_type,
    }
    if 'Content-Range' in remote_response.headers:
        response_headers['Content-Range'] = remote_response.headers['Content-Range']
    if 'Accept-Ranges' in remote_response.headers:
        response_headers['Accept-Ranges'] = remote_response.headers['Accept-Ranges']
    
    return Response(generate(), status=status, headers=response_headers, mimetype=content_type)


def classify_intent(user_input):
    input_lower = user_input.lower().strip()
    greetings = ["hi", "hello", "hello friend", "hey", "namaste"]
    tokens = input_lower.split()
    if all(token in greetings for token in tokens):
        return "greeting"
    song_keywords = ["play", "listen", "song", "music"]
    if any(keyword in input_lower for keyword in song_keywords):
        return "song_request"
    for title in music_df['song_name']:
        if input_lower in title:
            return "song_request"
    return "conversation"

def is_affirmative(text):
    affirmative_words = ["yes", "yeah", "sure", "yep", "ok", "okay", "of course"]
    return text.strip().lower() in affirmative_words

# ------------------------------------------------
# Route Definitions
# ------------------------------------------------
@app.route('/')
def index():
    username = session.get('username')
    return render_template('index.html', username=username)

@app.route('/login')
def login_page():
    return render_template('loging.html')  # Ensure the template file is named correctly

@app.route('/chatting')
def chatting():
    username = session.get('username')
    return render_template('chatPage.html', username=username)

@app.route('/get_user')
def get_user():
    return jsonify({'username': session.get('username')})

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('recommended_songs', None)
    session.pop('current_index', None)
    session.pop('follow_up_artist', None)
    return redirect(url_for('index'))


@app.route('/chat', methods=['POST'])
def chat():
    # Get user input and affirmation flag from the request
    data = request.get_json()
    user_input = data.get("user_input", "").strip()
    is_affirmation = data.get("is_affirmation", False)

    # Validate input
    if not user_input and not is_affirmation:
        return jsonify({"error": "No user input provided."}), 400

    # --- Handle Affirmation (e.g., "yes" to more recommendations) ---
    if is_affirmation:
        print("DEBUG: Affirmation detected. User wants more recommendations.")
        print("DEBUG: Current session state:")
        print(f"  was_specific_song: {session.get('was_specific_song')}")
        print(f"  follow_up_artist: {session.get('follow_up_artist')}")
        print(f"  follow_up_mood: {session.get('follow_up_mood')}")
        print(f"  follow_up_language: {session.get('follow_up_language')}")

        was_specific_song = session.get("was_specific_song", False)
        if was_specific_song:
            print("DEBUG: This is a follow‑up for a specific song query.")
            mood = session.get("follow_up_mood")
            language = session.get("follow_up_language")
            if mood:
                print(f"DEBUG: Using mood '{mood}' for recommendations.")
                additional_songs = music_df[music_df['mood_label'] == mood]
                if language:
                    additional_songs = additional_songs[additional_songs['language'] == language]
                response_msg = "Here are more songs you may like."
            else:
                print("DEBUG: No mood found in session. Falling back to random songs.")
                additional_songs = music_df.sample(n=2)
                response_msg = "Here are some random song recommendations."
        else:
            artist = session.get("follow_up_artist")
            language = session.get("follow_up_language")
            if artist:
                print(f"DEBUG: This is a follow‑up for an artist query. Artist: {artist}")
                additional_songs = music_df[music_df['artist_name'].str.lower() == artist.lower()]
                if language:
                    additional_songs = additional_songs[additional_songs['language'] == language]
                response_msg = f"Here are more songs by {artist.title()}."
            else:
                print("DEBUG: No artist or mood in session. Falling back to random songs.")
                additional_songs = music_df.sample(n=2)
                response_msg = "Here are some random song recommendations."

        # Filter out previously recommended songs
        prev_recs = session.get("recommended_songs", [])
        prev_song_names = [song["song_name"] for song in prev_recs]

        print(f"DEBUG: Number of songs before filtering: {len(additional_songs)}")
        additional_songs = additional_songs[~additional_songs['song_name'].isin(prev_song_names)]
        print(f"DEBUG: Number of unique songs after filtering: {len(additional_songs)}")

        # Fallback if no unique songs are available
        if additional_songs.empty:
            print("DEBUG: No unique songs available. Using fallback to random songs.")
            additional_songs = music_df.sample(n=5)
            response_msg = "No new songs available, here are some alternatives."

        # Limit the recommendations if too many results are present.
        if len(additional_songs) > 2:
            additional_songs = additional_songs.sample(n=2)

        # Prepare suggestions and update session
        suggestions = additional_songs.replace({np.nan: None}).to_dict(orient='records')
        print(f"DEBUG: Final suggestions: {suggestions}")
        if suggestions:
            song_to_play = suggestions[0]
            stream_url = get_stream_url(song_to_play['song_name'], song_to_play['artist_name'])
            song_to_play['audio_url'] = stream_url if stream_url else "https://example.com/default.mp3"
        else:
            song_to_play = {}

        session['recommended_songs'] = suggestions
        session['current_index'] = 0
        print("DEBUG: Session at request start:", session)
        print("DEBUG: Sending response:", {"response": response_msg, "suggestions": suggestions})
        return jsonify({
            "response": response_msg,
            "suggestions": suggestions,
            "song": song_to_play
        })

    # --- Process New Query ---
    print("DEBUG: Processing new user query.")
    intent = classify_intent(user_input)
    if intent == "greeting":
        return jsonify({"response": "Hello there! How can I help you with your music today?"})
    
    # Extract entities.
    artist, mood, language, album, query_song, album_requested = extract_entities(user_input)
    print(f"DEBUG: Extracted entities - Artist: {artist}, Mood: {mood}, Language: {language}, Album: {album}, Song: {query_song}")
    # If a language is detected, save it in the session for follow-up recommendations.
    if language:
        session['follow_up_language'] = language

        # --- NEW: Album Query Branch ---
    if album_requested and album:
        print("DEBUG: Album query detected.")
        filtered = music_df[music_df['album_movie_name'].str.lower() == album.lower()]
        if not filtered.empty:
            session["follow_up_album"] = album
            session["was_specific_song"] = False
            follow_up_msg = f"Would you like to listen to more songs from the album {album.title()}?"
        else:
            follow_up_msg = ""
        matched_songs = filtered.replace({np.nan: None}).to_dict(orient='records')[:2]

    # --- Specific Song Query Branch ---
    elif query_song:
        print("DEBUG: Specific song query detected.")
        filtered = music_df[music_df['song_name'].str.lower() == query_song.lower()]
        if not filtered.empty:
            specific_song = filtered.iloc[0]
            session["follow_up_mood"] = specific_song['mood_label']
            session["follow_up_language"] = specific_song['language']
            session["was_specific_song"] = True
            session.pop("follow_up_artist", None)  # Clear artist context

            # Include the requested song plus mood-based recommendations,
            # but filter by the detected language.
            mood_based = music_df[music_df['mood_label'] == specific_song['mood_label']]
            if specific_song['language']:
                mood_based = mood_based[mood_based['language'] == specific_song['language']]
            mood_based = mood_based[mood_based['song_name'].str.lower() != query_song.lower()]
            additional = mood_based.sample(n=min(4, len(mood_based))) if not mood_based.empty else pd.DataFrame()
            filtered = pd.concat([filtered, additional]).drop_duplicates(subset=['song_name', 'artist_name'])
            matched_songs = filtered.replace({np.nan: None}).to_dict(orient='records')
            follow_up_msg = "Would you like to listen to more songs like this?"
        else:
            matched_songs = []
            follow_up_msg = ""
    
    # --- Artist Query Branch ---
    elif artist:
        print("DEBUG: Artist-based query detected.")
        filtered = music_df[music_df['artist_name'].str.lower() == artist.lower()]
        if language:
            filtered = filtered[filtered['language'] == language]
        if not filtered.empty:
            session["follow_up_artist"] = artist
            session["was_specific_song"] = False
            session.pop("follow_up_mood", None)
            follow_up_msg = f"Would you like to listen to more songs by {artist.title()}?"
        else:
            follow_up_msg = ""
        matched_songs = filtered.replace({np.nan: None}).to_dict(orient='records')[:2]
    
    # --- Generic Query Branch ---
    else:
        print("DEBUG: No specific query, providing recommendations.")
        if language:
            filtered = music_df[music_df['language'] == language]
            if filtered.empty:
                filtered = music_df.sample(n=2)
        else:
            filtered = music_df.sample(n=2)
        matched_songs = filtered.replace({np.nan: None}).to_dict(orient='records')
        follow_up_msg = ""
        session["was_specific_song"] = False
        session.pop("follow_up_artist", None)
        session.pop("follow_up_mood", None)
        session.pop("follow_up_album", None)

    # Prepare song for playback.
    song_to_play = {}
    if matched_songs:
        song_to_play = matched_songs[0]
        stream_url = get_stream_url(song_to_play['song_name'], song_to_play['artist_name'])
        song_to_play['audio_url'] = stream_url if stream_url else "https://example.com/default.mp3"

    session['recommended_songs'] = matched_songs
    session['current_index'] = 0

    # Generate prompt.
    prompt = construct_prompt(user_input, artist, mood, language, matched_songs)
    if follow_up_msg:
        prompt += " " + follow_up_msg

    try:
        response = co.generate(
            model="command",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        response_text = response.generations[0].text.strip()
    except Exception as e:
        return jsonify({"error": "Error generating response from the LLM", "details": str(e)}), 500

    return jsonify({
        "response": response_text,
        "suggestions": matched_songs,
        "song": song_to_play
    })


@app.route('/next_song', methods=['GET'])
def next_song():
    recommended_songs = session.get('recommended_songs', [])
    current_index = session.get('current_index', 0)

    if not recommended_songs:
        return jsonify({'response': "No songs in the queue. Please request some songs first!"}), 400

    current_index += 1
    if current_index >= len(recommended_songs):
        return jsonify({'response': "You've reached the end of the song list. Want more recommendations?"}), 200

    next_song_data = recommended_songs[current_index]
    if not next_song_data.get('audio_url'):
        stream_url = get_stream_url(next_song_data['song_name'], next_song_data['artist_name'])
        next_song_data['audio_url'] = stream_url if stream_url else "https://example.com/default.mp3"

    next_song_for_user = {
        'song_name': next_song_data['song_name'],
        'artist_name': next_song_data['artist_name'],
        'audio_url': next_song_data['audio_url']
    }

    session['current_index'] = current_index
    response_text = f"Next song: {next_song_data['song_name']} by {next_song_data['artist_name']}"
    return jsonify({'response': response_text, 'song': next_song_for_user})

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email    = data.get('email')
    password = data.get('password')
    if not (username and email and password):
        return jsonify({"error": "All fields required"}), 400
    message = register_user(username, email, password)
    return jsonify({"message": message})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not (username and password):
        return jsonify({"error": "Credentials required"}), 400
    message = login_user(username, password)
    if message == "Login successful":
        session["username"] = username
        return jsonify({"message": message})
    return jsonify({"message": message}), 401

if __name__ == '__main__':
    app.run(debug=True)