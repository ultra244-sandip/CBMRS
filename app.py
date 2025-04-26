import os
import numpy as np
import time
import pandas as pd
import cohere
import subprocess
import re
import unicodedata
        
import requests
from flask import request, Response, abort


import os, time, re, unicodedata, subprocess, logging, numpy as np, pandas as pd, requests
from datetime import timedelta
from flask import Flask, render_template, jsonify, request, session, redirect, url_for, Response, abort, stream_with_context



from rapidfuzz import process, fuzz
from flask_session import Session
from dotenv import load_dotenv

# Import authentication functions from Credentials.py
from Credentials import register_user, login_user, get_db, close_db

# Import functions to verify email
from auth import send_otp_via_email, verify_otp
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.secret_key = "your_secret_key"
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['SESSION_FILE_DIR'] = './.flask_session/'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_SAMESITE'] = "Lax"
Session(app)

import logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY environment variable not set")
co = cohere.Client(cohere_api_key)

def normalize_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

def normalize_album(album):
    album = re.sub(r'\(.*?\)', '', album)
    album = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', album)
    return normalize_text(album)

music_df = pd.read_csv('music.csv')
music_df['album_movie_name'] = music_df['album_movie_name'].astype(str).apply(normalize_album)
music_df['song_name'] = music_df['song_name'].astype(str).apply(normalize_text)
music_df['artist_name'] = music_df['artist_name'].astype(str).apply(normalize_text)
music_df['mood_label'] = music_df['mood_label'].astype(str).apply(normalize_text)
music_df['language'] = music_df['language'].astype(str).apply(normalize_text)

all_moods = music_df['mood_label'].unique().tolist()
all_artists = music_df['artist_name'].unique().tolist()
all_languages = music_df['language'].unique().tolist()
all_albums = music_df['album_movie_name'].unique().tolist()
all_song_names = music_df['song_name'].unique().tolist()
from rapidfuzz import process, fuzz
import re

# You already have a function that uses rapidfuzz; assume it's defined:
# Helper: Fuzzy song match lookup
def find_full_song_match(candidate_song, all_song_names, score_cutoff=80):
    if not candidate_song.strip():
        return None
    matches = process.extract(candidate_song, all_song_names, scorer=fuzz.token_set_ratio)
    candidate_token_count = len(candidate_song.split())
    filtered_matches = [
        (song, score) for song, score, _ in matches 
        if len(song.split()) >= candidate_token_count * 0.75
    ]
    if not filtered_matches:
        best_match = process.extractOne(candidate_song, all_song_names, scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff)
        return best_match[0] if best_match else None
    else:
        best_song = max(filtered_matches, key=lambda x: x[1])
        return best_song[0]
    
# ---------------------------------------------------
# Revised Entity Extraction
def extract_entities(user_input):
    # Normalize and remove command words.
    query = normalize_text(user_input)
    command_words = ['play', 'recommend', 'suggest', 'find', 'please']
    for word in command_words:
        query = query.replace(word, '')
    query = query.strip()
    
    # Initialize extraction variables.
    album_requested = False
    found_album, found_artist, found_mood, found_language, found_song = (None,)*5
    
    # --- Album/Movie Branch ---
    if re.search(r'(?:from)\s+\b(movie|album)\b', query):
        album_requested = True
        album_stopwords = {"album", "movie", "songs"}
        album_tokens = [tok for tok in query.split() if tok not in album_stopwords]
        album_candidate = " ".join(album_tokens).strip()
        if album_candidate:
            best_album = process.extractOne(album_candidate, all_albums, scorer=fuzz.ratio, score_cutoff=60)
            if best_album:
                found_album = best_album[0]
    
    # --- Artist Extraction ---
    # First try explicit pattern (e.g., "songs by ..." or "songs of ...")
    artist_pattern = re.compile(r'(?:songs?\s+(?:by|of)\s+)(.+)')
    match = artist_pattern.search(query)
    if match:
        artist_candidate = match.group(1).strip()
        best_artist = process.extractOne(artist_candidate, all_artists, scorer=fuzz.partial_ratio, score_cutoff=70)
        if best_artist:
            found_artist = best_artist[0]
    # Fallback: Use regex word-boundary check
    if not found_artist:
        for artist in all_artists:
            if re.search(r'\b' + re.escape(artist) + r'\b', query):
                found_artist = artist
                break

    # --- Mood and Language Extraction ---
    tokens = query.split()
    for token in tokens:
        if token in all_moods and not found_mood:
            found_mood = token
        if token in all_languages and not found_language:
            found_language = token

    # --- Candidate Song Extraction ---
    filter_tokens = set(["songs", "song", "of", "by"])
    if found_mood:
        filter_tokens.add(found_mood)
    if found_language:
        filter_tokens.add(found_language)
    if found_artist:
        filter_tokens.update(found_artist.split())
        
    # If album_requested, we skip song text extraction.
    candidate_song_tokens = [] if album_requested else [tok for tok in tokens if tok not in filter_tokens]
    
    print("DEBUG: Tokens after filtering for song extraction:", candidate_song_tokens)
    candidate_song = " ".join(candidate_song_tokens).strip() if candidate_song_tokens else ""
    print("DEBUG: Candidate song string:", candidate_song)
    if candidate_song:
        found_song = find_full_song_match(candidate_song, all_song_names, score_cutoff=80)
        print("DEBUG: Found song:", found_song)
    else:
        print("DEBUG: No specific song tokens after filtering; query likely targets artist, mood, language, or album.")
    
    return found_artist, found_mood, found_language, found_album, found_song, album_requested

# ---------------------------------------------------
# Updated prompt constructor for LLM
def construct_prompt(user_input, artist, mood, language, matched_songs):
    header = f"You are a friendly music assistant. The user said: '{user_input}'.\n"
    details = ""
    if language:
        details += f"The user wants {language.upper()} songs. "
    if mood:
        details += f"The mood is '{mood}'. "
    if artist:
        details += f"They are interested in songs by {artist.title()}. "
    if matched_songs:
        details += "Some suggestions from our library include:\n"
        for song in matched_songs[:2]:
            details += f" - '{song['song_name']}' by {song['artist_name']}\n"
    details += ("Respond politely and briefly, mentioning the song names and artists as given. "
                "Ask if they would like to listen to any of these songs.")
    return header + details

# ---------------------------------------------------
# Modify get_stream_url to accept a library_only flag (even if unused for now)
def get_stream_url(song_name, artist_name=None, library_only=False):
    """
    Calls yt-dlp to fetch the streaming URL and thumbnail
    for the given song and artist. Returns fallback values if the result is invalid.
    """
    query = song_name
    if artist_name:
        query += " " + artist_name
    app.logger.info("Fetching stream URL for query: %s", query)
    fallback_url = "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_700KB.mp3"
    fallback_thumb = "https://via.placeholder.com/150"
    
    try:
        search_command = [
            "yt-dlp",
            f"ytsearch1:{query}",
            "--get-url",
            "--get-thumbnail",
            "-f", "bestaudio",
            "--no-playlist"
        ]
        result = subprocess.run(search_command, capture_output=True, text=True, check=True)
        raw_output = result.stdout.strip()
        app.logger.info("Raw yt-dlp output: %s", raw_output)
        output = raw_output.split('\n')
        if len(output) >= 2:
            audio_url = output[0].strip()
            thumbnail_url = output[1].strip()
            # If the returned audio_url is invalid, log and use fallback.
            if not audio_url or audio_url.lower() == "undefined":
                app.logger.warning("Received invalid audio_url ('%s') for query '%s'. Using fallback.", audio_url, query)
                return fallback_url, fallback_thumb
            app.logger.info("Fetched audio_url: %s, thumbnail_url: %s", audio_url, thumbnail_url)
            return audio_url, thumbnail_url
        else:
            app.logger.warning("Output from yt-dlp is insufficient for query: %s", query)
            return fallback_url, fallback_thumb
    except subprocess.CalledProcessError as e:
        app.logger.error("Error fetching stream URL for query '%s': %s", query, e)
        return fallback_url, fallback_thumb

def enrich_song(song):
    # Convert to dict if needed
    if hasattr(song, 'to_dict'):
        song = song.to_dict()
    # Use get_stream_url to fetch URLs if not already present
    if not song.get('audio_url'):
        audio_url, thumbnail_url = get_stream_url(song["song_name"], song["artist_name"])
        # Use a reliable fallback if get_stream_url fails:
        song['audio_url'] = audio_url or "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_700KB.mp3"
        song['thumbnail_url'] = thumbnail_url or "https://via.placeholder.com/150"
    return song

@app.route('/prefetch_song', methods=['POST'])
def prefetch_song():
    data = request.get_json()
    if not data or "song" not in data:
        return jsonify({"error": "Missing song data"}), 400
    
    song = data.get("song")
    # Enrich song metadata with a fresh audio URL if necessary.
    enriched_song = enrich_song(song)
    
    return jsonify({"song": enriched_song})


@app.route('/proxy_audio')
def proxy_audio():
    url = request.args.get('url')
    if not url:
        abort(400, "No URL provided.")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...",
        "Referer": "https://www.youtube.com/",
        "Accept": "audio/webm, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    range_header = request.headers.get('Range')
    if range_header:
        headers['Range'] = range_header
    try:
        remote_response = requests.get(url, headers=headers, stream=True)
    except Exception as e:
        app.logger.error("Error fetching audio URL: %s", e)
        abort(500, f"Error fetching audio URL: {e}")
    
    if remote_response.status_code == 403:
        abort(403, "Access forbidden.")
    
    # Check for proper Content-Type
    content_type = remote_response.headers.get('Content-Type', '')
    if 'html' in content_type.lower():
        # Log error details
        app.logger.error("Unexpected Content-Type received: %s", content_type)
        abort(500, "The fetched URL did not return an audio stream.")
    
    status = remote_response.status_code

    def generate():
        for chunk in remote_response.iter_content(chunk_size=8192):
            if chunk:
                yield chunk

    response_headers = {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': content_type,
    }
    response_headers.setdefault('Accept-Ranges', 'bytes')
    for header in ['Content-Range', 'Accept-Ranges', 'Content-Length', 'Content-Disposition']:
        if header in remote_response.headers:
            response_headers[header] = remote_response.headers[header]
    return Response(stream_with_context(generate()), status=status, headers=response_headers, mimetype=content_type)

def classify_intent(user_input):
    input_lower = user_input.lower().strip()
    greetings = ["hi", "hii","hello", "helo", "hello friend", "hey", "namaste", "hloo", "hlw"]
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
    username = session.get('username')
    return jsonify({'username': username})

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('recommended_songs', None)
    session.pop('current_index', None)
    session.pop('follow_up_artist', None)
    return redirect(url_for('index'))

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("user_input", "").strip()
    is_affirmation = data.get("is_affirmation", False)
    
    if not user_input and not is_affirmation:
        return jsonify({"error": "No user input provided."}), 400

    # Affirmation Branch
    if is_affirmation:
        app.logger.info("Affirmation received.")
        if session.get("was_specific_song"):
            mood_filter = session.get("follow_up_mood")
            language_filter = session.get("follow_up_language")
            app.logger.info("Affirmation criteria: mood=%s, language=%s", mood_filter, language_filter)
            additional_songs = music_df[music_df['mood_label'] == mood_filter]
            if language_filter:
                additional_songs = additional_songs[additional_songs['language'] == language_filter]
            response_msg = {"Here are more songs, you may like them.","Hmmm, you may like these songs also.","Glad to hear, I will suggest more these type of songs. Enjoy"}
        elif session.get("follow_up_artist"):
            artist_filter = session.get("follow_up_artist")
            language_filter = session.get("follow_up_language")
            app.logger.info("Affirmation criteria: artist=%s, language=%s", artist_filter, language_filter)
            additional_songs = music_df[music_df['artist_name'].str.lower() == artist_filter.lower()]
            if language_filter:
                additional_songs = additional_songs[additional_songs['language'] == language_filter]
            response_msg = f"Here are more songs by {artist_filter.title()}."
        else:
            return jsonify({
                "response": "No further recommendations available matching your criteria. Please refine your query.",
                "suggestions": []
            })

        prev_recs = session.get("recommended_songs", [])
        prev_song_names = [song["song_name"] for song in prev_recs] if prev_recs and isinstance(prev_recs[0], dict) else []
        additional_songs = additional_songs[~additional_songs['song_name'].isin(prev_song_names)]
        
        if additional_songs.empty:
            return jsonify({
                "response": "No further recommendations available matching your criteria.",
                "suggestions": []
            })
        
        if len(additional_songs) > 2:
            additional_songs = additional_songs.sample(n=2)
        
        suggestions = additional_songs.copy().replace({np.nan: None}).to_dict(orient='records')
        app.logger.info("Affirmation branch - metadata suggestions: %s", suggestions)
        
        # Store songs and set current_index to -1
        session['recommended_songs'] = suggestions
        session['current_index'] = -1  # Changed from 0 to -1
        session.modified = True
        
        return jsonify({
            "response": response_msg,
            "suggestions": suggestions
        })

    # Normal Query Branch
    intent = classify_intent(user_input)
    
    if intent == "greeting":
        return jsonify({"response": "Hello there! How can I help you with your music today?"})
    
    artist, mood, language, album, query_song, album_requested = extract_entities(user_input)
    if not is_affirmation and language:
        session['follow_up_language'] = language

    if album_requested and album:
        filtered = music_df[music_df['album_movie_name'].str.lower() == album.lower()]
        if not filtered.empty:
            # Remove duplicate song entries (same song name and artist)
            filtered = filtered.drop_duplicates(subset=['song_name', 'artist_name'])
            session["follow_up_album"] = album
            session["was_specific_song"] = False
            follow_up_msg = f"Would you like to listen to more songs from the album {album.title()}?"
        else:
            follow_up_msg = ""
        matched_songs = filtered.copy().replace({np.nan: None}).to_dict(orient='records')

    elif query_song:
        filtered = music_df[music_df['song_name'].str.lower() == query_song.lower()]
        if not filtered.empty:
            # Remove duplicates just in case
            filtered = filtered.drop_duplicates(subset=['song_name', 'artist_name'])
            specific_song = filtered.iloc[0].to_dict()
            session["follow_up_mood"] = specific_song['mood_label']
            session["follow_up_language"] = specific_song['language']
            session["was_specific_song"] = True
            session.pop("follow_up_artist", None)
            matched_songs = [specific_song]
            follow_up_msg = "Would you like to listen to more songs like this?"
        else:
            matched_songs = []
            follow_up_msg = ""

    elif artist:
        filtered = music_df[music_df['artist_name'].str.lower() == artist.lower()]
        if language:
            filtered = filtered[filtered['language'] == language]
        if not filtered.empty:
            # Remove duplicate songs
            filtered = filtered.drop_duplicates(subset=['song_name', 'artist_name'])
            # Optionally limit the number of songs to 2 if more than 2 exist
            if len(filtered) > 2:
                filtered = filtered.sample(n=2)
            session["follow_up_artist"] = artist
            session["was_specific_song"] = False
            session.pop("follow_up_mood", None)
            follow_up_msg = f"Would you like to listen to more songs by {artist.title()}?"
        else:
            follow_up_msg = ""
        matched_songs = filtered.copy().replace({np.nan: None}).to_dict(orient='records')

    else:
        if language:
            filtered = music_df[music_df['language'] == language]
            if mood:
                filtered = filtered[filtered['mood_label'].str.contains(mood, case=False, na=False)]
            if filtered.empty:
                return jsonify({
                    "response": f"Sorry, no songs found for mood '{mood}' in {language.upper()}.",
                    "suggestions": []
                })
            else:
                # Remove duplicates
                filtered = filtered.drop_duplicates(subset=['song_name', 'artist_name'])
                if len(filtered) > 2:
                    filtered = filtered.sample(n=2)
        else:
            if mood:
                filtered = music_df[music_df['mood_label'].str.contains(mood, case=False, na=False)]
                if filtered.empty:
                    return jsonify({
                        "response": f"Sorry, no songs found for mood '{mood}'.",
                        "suggestions": []
                    })
                else:
                    # Remove duplicates
                    filtered = filtered.drop_duplicates(subset=['song_name', 'artist_name'])
                    if len(filtered) > 2:
                        filtered = filtered.sample(n=2)
            else:
                filtered = music_df.sample(n=2)
                
        matched_songs = filtered.copy().replace({np.nan: None}).to_dict(orient='records')
        follow_up_msg = ""
        session["was_specific_song"] = False
        session.pop("follow_up_artist", None)
        session.pop("follow_up_mood", None)
        session.pop("follow_up_album", None)

    # Store songs and set current_index to -1
    session['recommended_songs'] = matched_songs
    session['current_index'] = -1  # Changed from 0 to -1
    session.modified = True
    app.logger.info("Stored recommended_songs (metadata only): %s", matched_songs)

    prompt = construct_prompt(user_input, artist, mood, language, matched_songs)
    if follow_up_msg:
        prompt += " " + follow_up_msg
    
    try:
        response_obj = co.generate(
            model="command",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        response_text = response_obj.generations[0].text.strip()
    except Exception as e:
        return jsonify({"error": "Error generating response from the LLM", "details": str(e)}), 500
    
    return jsonify({
        "response": response_text,
        "suggestions": matched_songs
    })
from flask import jsonify, session

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

    # Fetch a new URL if the current one is missing or invalid
    if not next_song_data.get('audio_url') or next_song_data.get('audio_url') in [None, '', 'undefined']:
        app.logger.info("Audio URL missing or invalid for next song '%s' by '%s'. Fetching a fresh URL.",
                        next_song_data.get("song_name"), next_song_data.get("artist_name"))
        audio_url, thumbnail_url = get_stream_url(next_song_data["song_name"], next_song_data["artist_name"])
        next_song_data["audio_url"] = audio_url
        next_song_data["thumbnail_url"] = thumbnail_url
        recommended_songs[current_index] = next_song_data
        session['recommended_songs'] = recommended_songs
        session.modified = True

    session['current_index'] = current_index
    session.modified = True
    response_text = f"Next song: {next_song_data['song_name']} by {next_song_data['artist_name']}"
    return jsonify({'response': response_text, 'song': next_song_data})

@app.route('/select_song', methods=['GET'])
def select_song():
    recommended_songs = session.get('recommended_songs', [])
    current_index = session.get('current_index', 0)
    if not recommended_songs:
        return jsonify({'response': "No songs in the queue."}), 400

    song_metadata = recommended_songs[current_index]
    # Log the metadata for debugging.
    app.logger.info("Selecting song, metadata: %s", song_metadata)
    
    # Check if audio_url is missing/empty/undefined.
    if not song_metadata.get('audio_url') or song_metadata.get('audio_url') in [None, '', 'undefined']:
        app.logger.info("Audio URL missing for song '%s' by '%s'. Fetching now.", 
                        song_metadata.get("song_name"), song_metadata.get("artist_name"))
        # Ensure metadata fields exist.
        song_name = song_metadata.get("song_name")
        artist_name = song_metadata.get("artist_name")
        if not song_name or not artist_name:
            app.logger.error("Missing song name or artist name in metadata.")
            return jsonify({"error": "Song metadata incomplete."}), 500
        
        audio_url, thumbnail_url = get_stream_url(song_name, artist_name)
        # If get_stream_url returns an invalid URL ("undefined"), use fallback.
        if audio_url in [None, '', 'undefined']:
            app.logger.error("get_stream_url returned an invalid URL for '%s' by '%s'", song_name, artist_name)
            audio_url = "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_700KB.mp3"
            thumbnail_url = "https://via.placeholder.com/150"
        
        song_metadata["audio_url"] = audio_url
        song_metadata["thumbnail_url"] = thumbnail_url or "https://via.placeholder.com/150"
        # Update the session.
        recommended_songs[current_index] = song_metadata
        session['recommended_songs'] = recommended_songs
        session.modified = True

    response_text = f"Selected song: {song_metadata['song_name']} by {song_metadata['artist_name']}"
    return jsonify({'response': response_text, 'song': song_metadata})

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    if not (username and email and password):
        return jsonify({"error": "All fields required"}), 400
    db = get_db()
    existing = db.execute("SELECT * FROM Users WHERE email = ?", (email,)).fetchone()
    close_db()
    if existing:
        return jsonify({"error": "User already exists. Please Login."}), 400
    else:
        session['pending_registration'] = {"username": username, "email": email, "password": password}
    session['email'] = email
    send_otp_via_email(email)
    return jsonify({"message": "Otp sent to your email. Please verify to complete registration."})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')  # Ensure the template filename is correct.
    elif request.method == 'POST':
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

@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    otp = data.get('otp')
    if not otp:
        return jsonify({"error": "OTP is required"}), 400
    pending = session.get('pending_registration')
    if not pending:
        return jsonify({"error": "Session expired. Please register again."}), 400
    if verify_otp(otp):
        register_user(pending['username'], pending['email'], pending['password'], final=True)
        session.pop('pending_registration', None)
        session.pop('email', None)
        return jsonify({"message": "Email verified successfully and registration complete!"})
    else:
        return jsonify({"error": "Invalid or expired OTP."}), 400

if __name__ == '__main__':
    app.run(debug=True)