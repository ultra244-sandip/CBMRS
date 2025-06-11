import os
import numpy as np
import time
import pandas as pd

from google import genai
import subprocess
import re
import unicodedata
import random        
import requests , pytz
from flask import request, Response, abort
import datetime
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from bson import ObjectId  # Ensure this import is available

import os, time, re, unicodedata, subprocess, logging, numpy as np, pandas as pd, requests
from flask import Flask, render_template, jsonify, request, session, redirect, url_for, Response, abort, stream_with_context, flash

from rapidfuzz import process, fuzz
from flask_session import Session
from dotenv import load_dotenv
from bson import ObjectId
from mongo_integration import mongo_bp, chats_collection
from mongo_integration import get_liked_song_ids
# Import authentication functions from Credentials.py
from Credentials import register_user, login_user, get_db, close_db, get_email, get_subscription, update_subscription_status

# Import functions to verify email
from auth import send_otp_via_email, verify_otp, subscription_mail
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.secret_key = "your_secret_key"
app.config['SESSION_TYPE'] = 'cookie'  # Use cookie-based sessions
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['SESSION_FILE_DIR'] = './.flask_session/'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_SAMESITE'] = "Lax"


import logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)


from mongo_integration import mongo_bp
app.register_blueprint(mongo_bp)


''' cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY environment variable not set")
co = cohere.Client(cohere_api_key)'''

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

gemini_client = genai.Client(api_key=gemini_api_key)


def normalize_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

def normalize_art(text):
    text = unicodedata.normalize('NFKC', text)
    # Preserve commas by adding them to the allowed characters.
    text = re.sub(r'[^\w\s,&]', '', text)
    return text.strip().lower()


def normalize_album(album):
    album = re.sub(r'\(.*?\)', '', album)
    album = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', album)
    return normalize_text(album)



# Read and normalize the data.
music_df = pd.read_csv('musicc.csv')
music_df['album_movie_name'] = music_df['album_movie_name'].astype(str).apply(normalize_album)
music_df['song_name'] = music_df['song_name'].astype(str).apply(normalize_text)
music_df['artist_name'] = music_df['artist_name'].astype(str).apply(normalize_art)
music_df['mood_label'] = music_df['mood_label'].astype(str).apply(normalize_text)
music_df['language'] = music_df['language'].astype(str).apply(normalize_text)

# Define the helper function.
def split_artist_field(artist_field):
    # If a comma is present, split by comma.
    if ',' in artist_field:
        return [artist.strip() for artist in artist_field.split(',') if artist.strip()]
    # Otherwise, if an ampersand is present, you can decide whether to split.
    # Be careful: if you know that names like "simon & garfunkel" should remain together,
    # you might need a list of known group names.
    elif '&' in artist_field:
        # For now, let’s assume that if a comma is not present, the ampersand is part of the name.
        # OR if you want to split on ampersand, uncomment the next line.
        # return [artist.strip() for artist in artist_field.split('&') if artist.strip()]
        return [artist_field.strip()]
    else:
        return [artist_field.strip()]

def filter_songs_by_artist(selected_artist, df):
    selected_artist = selected_artist.lower()
    # Filter based on whether the selected_artist is one of the names in the artist field.
    return df[df['artist_name'].apply(lambda a: selected_artist in [x.lower() for x in split_artist_field(a)])]
# Build a flattened list of individual artist names.
processed_artists = []
for art in music_df['artist_name']:
    processed_artists.extend(split_artist_field(art))
    
# Remove duplicates.
all_artists = list(set(processed_artists))


# Optionally, update your other lists as needed.
all_moods = music_df['mood_label'].unique().tolist()
all_languages = music_df['language'].unique().tolist()
all_albums = music_df['album_movie_name'].unique().tolist()
all_song_names = music_df['song_name'].unique().tolist()


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
    
# Assuming all_moods, all_languages, all_artists, all_albums, all_song_names are defined
stop_words = {"suggest me", "recommend me","old", "some", "a", "the", "of", "by", "in", "for", "to", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being"}
import re
import unicodedata
import string
from fuzzywuzzy import process, fuzz
import spacy
from spacy.matcher import PhraseMatcher

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# These globals should be defined (or imported) elsewhere:
# all_moods, all_languages, all_artists, all_song_names, all_albums

def normalize_text(text):
    return text.lower().strip()

def preprocess_query(query):
    # Define a set of keywords to remove
    keywords_to_remove = {"play", "songs", "song", "listen", "music"}
    tokens = query.split()
    # Remove the keywords regardless of case
    filtered_tokens = [tok for tok in tokens if tok.lower() not in keywords_to_remove]
    return " ".join(filtered_tokens).strip()


def correct_artist_spelling(artist_candidate):
    best_match = process.extractOne(artist_candidate, all_artists, scorer=fuzz.token_set_ratio, score_cutoff=60)
    if best_match:
        return best_match[0]
    simplified_candidate = re.sub(r'[^a-zA-Z0-9]', '', artist_candidate)
    best_match = process.extractOne(
        simplified_candidate,
        [re.sub(r'[^a-zA-Z0-9]', '', art) for art in all_artists],
        scorer=fuzz.token_set_ratio,
        score_cutoff=60
    )
    if best_match:
        index = [re.sub(r'[^a-zA-Z0-9]', '', art) for art in all_artists].index(best_match[0])
        return all_artists[index]
    return artist_candidate  # Fallback
import re

def normalize_query(query: str) -> str:
    query = query.lower()
    # Remove only generic filler words but **preserve** mood-related words.
    query = re.sub(r'\b(play (songs|tracks) of|songs of|tracks of)\b', '', query)
    return query.strip()

def split_artists(artist_field: str) -> list:
    if artist_field is None:
        return []
    # If the field contains commas, split by it
    if ',' in artist_field:
        return [a.strip() for a in artist_field.split(',') if a.strip()]
    # Otherwise, return the artist_field in a list
    return [artist_field.strip()]
from rapidfuzz import fuzz

def select_preferred_artist(query: str, candidate_artists: list) -> str:
    if not candidate_artists:
        return None
    normalized_query = normalize_query(query)
    best_candidate = None
    best_score = -1
    for candidate in candidate_artists:
        score = fuzz.token_set_ratio(normalized_query, candidate)
        print(f"Comparing '{candidate}' to normalized query '{normalized_query}' -> score: {score}")
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate

def filter_songs_by_artist(selected_artist, df):
    selected_artist = selected_artist.lower()
    # For each row, split the artist field into individual names and check if our selected_artist is in that list.
    return df[df['artist_name'].apply(lambda a: selected_artist in [x.lower() for x in split_artist_field(a)])]

def fallback_artist_from_query(query, all_artists, min_ngram=2, max_ngram=5, score_cutoff=70):
    """
    Try to extract an artist candidate from the query using a sliding window.
    For each n-gram (from min_ngram to max_ngram tokens), compare it to all_artists
    using fuzzy matching. Returns the candidate with the best score, or None.
    """
    tokens = query.split()
    best_candidate = None
    best_score = -1
    # Try n-grams of various lengths:
    for n in range(min_ngram, min(max_ngram + 1, len(tokens) + 1)):
        for i in range(0, len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i + n])
            result = process.extractOne(ngram, all_artists, scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff)
            if result:
                # Handle both 3-tuple and 2-tuple returns.
                if isinstance(result, (tuple, list)):
                    if len(result) == 3:
                        candidate, score, _ = result
                    else:
                        candidate, score = result
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate
    if best_candidate:
        print(f"Fallback matched '{best_candidate}' with score {best_score} from query segment.")
    return best_candidate


def extract_entities_nlp(user_input):
    original_query = user_input.strip()
    print(f"Original query: '{original_query}'")

    cleaned_query = preprocess_query(original_query)
    normalized_query = normalize_text(cleaned_query)
    print(f"Cleaned & Normalized query: '{normalized_query}'")
    low_query = original_query.lower()

    # New album override block:
    album_keywords = {"album", "soundtrack", "movie album", "movie"}
    if any(keyword in low_query for keyword in album_keywords):
        album_requested = True
        print("Album keyword detected; skipping artist extraction.")
        found_artist = None  # ensure no artist is set
    else:
        album_requested = False

    # Initialize other variables.
    found_mood, found_language = None, None
    # Do NOT reinitialize found_artist; we already set it (or cleared it) above.
    found_album = None
    found_song = None
    is_info_query = False
    explicit_artist_extraction = False
    is_artist_request = False

    # --------------------  
    # **Step 1: Detect Information Retrieval Queries Early**
    info_triggers = [
        "tell me", "do you know", "tell me about",
        "who is", "where is", "what is", "when was",
        "information about", "how does", "explain"
    ]

    if any(trigger in low_query for trigger in info_triggers):
        is_info_query = True
        print("Information retrieval query detected; skipping song and album extraction.")

    # --- Step 1: Extract Language First ---
    doc = nlp(cleaned_query)
    matcher_language = PhraseMatcher(nlp.vocab, attr="LOWER")
    language_patterns = [nlp.make_doc(lang) for lang in all_languages]
    matcher_language.add("LANGUAGE", language_patterns)
    matches_language = matcher_language(doc)
    if matches_language:
        match_id, start, end = matches_language[0]
        found_language = doc[start:end].text.lower()
        print(f"Found language (via spaCy): '{found_language}'")
    else:
        print("No language detected.")

    # --- Step 2: Extract Mood Before Artist ---
    matcher_mood = PhraseMatcher(nlp.vocab, attr="LOWER")
    mood_patterns = [nlp.make_doc(mood) for mood in all_moods]
    matcher_mood.add("MOOD", mood_patterns)
    matches_mood = matcher_mood(doc)
    if matches_mood:
        match_id, start, end = matches_mood[0]
        found_mood = doc[start:end].text.lower()
        print(f"Found mood (via spaCy): '{found_mood}'")
    else:
        print("No mood detected.")

    # --- Step 3: Extract Artist (only if no album keyword detected) ---
    if not album_requested:
        # Step 3a: NER-based Extraction
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                candidate = ent.text or ""
                candidate_artists = split_artists(candidate)
                print(f"NER candidate artists from '{candidate}': {candidate_artists}")
                preferred_artist = select_preferred_artist(original_query, candidate_artists)
                if preferred_artist:
                    candidate = preferred_artist
                    print(f"Preferred artist selected via NER: '{candidate}'")
                best_artist = process.extractOne(candidate, all_artists, scorer=fuzz.token_set_ratio, score_cutoff=70)
                if best_artist:
                    found_artist = best_artist[0]
                    print(f"Extracted artist via NER after refinement: '{found_artist}' with score {best_artist[1]}")
                    explicit_artist_extraction = True
                    break

        # Step 3b: Regex-based Extraction
        if not found_artist:
            artist_pattern = re.compile(r'(?:songs?\s+(?:by|of)\s+)(.+)', re.IGNORECASE)
            match = artist_pattern.search(original_query)
            if match:
                artist_candidate = match.group(1).strip()
                if artist_candidate.lower() in all_moods:
                    print(f"Skipping '{artist_candidate}' as it's a mood, not an artist.")
                    found_artist = None
                else:
                    candidate_artists = split_artists(artist_candidate)
                    preferred_artist = select_preferred_artist(original_query, candidate_artists)
                    if preferred_artist:
                        best_artist = process.extractOne(preferred_artist, all_artists, scorer=fuzz.WRatio, score_cutoff=70)
                        if best_artist:
                            candidate_from_db = best_artist[0]
                            if len(preferred_artist.split()) > len(candidate_from_db.split()):
                                found_artist = preferred_artist
                            else:
                                found_artist = candidate_from_db
                            explicit_artist_extraction = True
                            print(f"Extracted artist from explicit pattern: '{found_artist}' with score {best_artist[1]}")

            # Step 3c: Fallback Extraction via Sliding Window
            if not found_artist:
                filtered_query = original_query
                if found_language:
                    filtered_query = filtered_query.replace(found_language, "")
                if found_mood:
                    for mt in found_mood.split():
                        filtered_query = filtered_query.replace(mt, "")
                filtered_query = filtered_query.strip()
                filler_words = {"play", "songs", "song", "listen", "music", "some", "any"}
                filtered_tokens = [tok for tok in filtered_query.split() if tok.lower() not in filler_words]
                filtered_query = " ".join(filtered_tokens).strip()
                if len(filtered_query.split()) >= 2:
                    fallback_candidate = fallback_artist_from_query(filtered_query, all_artists)
                    if fallback_candidate:
                        found_artist = fallback_candidate
                        explicit_artist_extraction = True
                        print(f"Fallback extraction found artist: '{found_artist}'.")
                else:
                    print("Filtered query too generic; skipping fallback extraction.")
    else:
        # When album_requested is True, we skip artist extraction entirely.
        print("Skipping artist extraction because album_requested is True.")

    # --- Step 4: Extract Album (if applicable) ---
    if re.search(r'\b(movie|album)\b', low_query, re.IGNORECASE):
        temp_query = re.sub(r'\b(play|recommend|suggest|find|please)\b', '', original_query).strip()
        album_candidate = " ".join([tok for tok in temp_query.split() if tok.lower() not in {"movie", "album", "songs", "song", "of", "by"}]).strip()
        print(f"Album candidate: '{album_candidate}'")
        if album_candidate:
            best_album = process.extractOne(album_candidate, all_albums, scorer=fuzz.ratio, score_cutoff=60)
            if best_album:
                found_album = best_album[0]
                print(f"Found album: '{found_album}' with score {best_album[1]}")
            else:
                print("No album match found.")
        # --------------------  
        # **Step 7: Specific Song Extraction**
        if explicit_artist_extraction or album_requested:
            print("Skipping song extraction due to explicit artist or album request.")
            found_song = None
            is_artist_request = True
        else:
            # Build candidate tokens from the normalized query.
            query_tokens = normalized_query.split()
            filter_tokens = {"of", "by", "me", "please", "the","some", "a", "an", "songs", "song"}
            candidate_tokens = [
                tok.translate(str.maketrans('', '', string.punctuation))
                for tok in query_tokens if tok.lower() not in filter_tokens
            ]
            
            # Concatenate tokens to form the candidate song string.
            candidate_song = " ".join(candidate_tokens).strip()
            
            # -- Remove the mood phrase from candidate song string if it exists.
            if found_mood:
                # Remove the exact mood phrase if present.
                candidate_song = candidate_song.replace(found_mood, "").strip()
            
            print(f"Candidate song string: '{candidate_song}'")
            
            # If the candidate song string is empty or only a filler word, treat as artist-only query.
            filler_words = {"some", "any", "songs", "track", "music"}
            if candidate_song in filler_words or candidate_song == "":
                print("No valid candidate song extracted; treating as an artist-only or recommendation query.")
                is_artist_request = True
                found_song = None
            else:
                candidate_token_count = len(candidate_song.split())
                score_cutoff = 90 if candidate_token_count == 1 else 80
                if candidate_song in all_song_names:
                    found_song = candidate_song
                    print(f"Found song by exact match: '{found_song}'")
                else:
                    best_match = process.extractOne(candidate_song, all_song_names,
                                                    scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff)
                    if best_match:
                        found_song = best_match[0]
                        print(f"Found song: '{found_song}' with score {best_match[1]}")
                    else:
                        print("No matching song found through fuzzy search.")
                        found_song = None
                is_artist_request = False

    # --------------------  
    # **Step 8: Final Debug Summary**
    print("\n--- Extracted Entities ---")
    print(f"Artist: {found_artist}")
    print(f"Mood: {found_mood}")
    print(f"Language: {found_language}")
    print(f"Album: {found_album}")
    print(f"Song: {found_song}")
    print(f"Album requested: {album_requested}")
    print(f"Info query: {is_info_query}")
    print(f"Artist-only query: {is_artist_request}")
    print("-------------------------\n")

    return found_artist, found_mood, found_language, found_album, found_song, album_requested, is_info_query

def construct_prompt(user_input, intent, artist=None, mood=None, language=None, matched_songs=None):
    if intent == "information_request":
        header = f"You are a knowledgeable assistant and a friendly music assistant. your name is sur-sangeet. The user asked: '{user_input}'.\n"
        details = ("Please provide concise and factual information relevant to this query. "
                   "Do not suggest songs unless explicitly asked.")
        return header + details
    else:
        header = f"You are a friendly music assistant. your name is sur-sangeet. The user said: '{user_input}'.\n"
        details = ""
        if language:
            if isinstance(language, list):
                lang_str = " / ".join(language).upper()
            else:
                lang_str = language.upper()
            details += f"The user wants {lang_str} songs. "
        if mood:
            details += f"The mood is '{mood}'. "
        if artist:
            details += f"They are interested in songs by {artist.title()}. "
        if matched_songs:
            details += "Some suggestions from our library include:\n"
            for song in matched_songs[:5]:
                details += f" - '{song['song_name']}' by {song['artist_name']}\n"
        details += ("Respond politely and briefly, mentioning the song names and artists as given. "
                    "Ask if they would like to listen to any of these songs.")
        return header + details

# ---------------------------------------------------
# Modify get_stream_url to accept a library_only flag (even if unused for now)
def get_stream_url(song_name, artist_name=None, library_only=False):
    query = song_name
    if artist_name:
        query += " " + artist_name

    logging.info("Fetching stream URL for query: %s", query)
    fallback_url = "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_700KB.mp3"
    fallback_thumb = "https://via.placeholder.com/150"
    
    # Command for yt-dlp
    search_command = [
        "yt-dlp",
        f"ytsearch1:{query}",
        "--get-url",
        "--get-thumbnail",
        "--add-header", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "--add-header", "Referer: https://www.youtube.com/",
        "-f", "bestaudio[ext=m4a]/bestaudio[ext=webm]/best",
        "--no-playlist"
    ]
    
    num_retries = 3
    for attempt in range(1, num_retries + 1):
        try:
            result = subprocess.run(search_command, capture_output=True, text=True, check=True)
            raw_output = result.stdout.strip()
            logging.info("Raw yt-dlp output: %s", raw_output)
            output_lines = raw_output.split('\n')
            if len(output_lines) >= 2:
                audio_url = output_lines[0].strip()
                thumbnail_url = output_lines[1].strip()
                if not audio_url or audio_url.lower() == "undefined":
                    logging.warning("Invalid audio_url ('%s') for query '%s'. Using fallback.", audio_url, query)
                    return fallback_url, fallback_thumb
                logging.info("Fetched audio_url: %s, thumbnail_url: %s", audio_url, thumbnail_url)
                return audio_url, thumbnail_url
            else:
                logging.warning("Insufficient output from yt-dlp for query: %s", query)
                # Fall back after retries if insufficient output
                break
        except subprocess.CalledProcessError as e:
            logging.error("yt-dlp error for query '%s' on attempt %d: %s, stderr: %s", query, attempt, e, e.stderr)
            if attempt < num_retries:
                time.sleep(1)  # Pause briefly before retrying
            else:
                # If last attempt fails, return fallback URL
                break

    logging.error("Unexpected failure when fetching URL for query '%s'. Using fallback.", query)
    return fallback_url, fallback_thumb

def enrich_song(song):
    # Convert to dict if needed
    if hasattr(song, 'to_dict'):
        song = song.to_dict()
    
    # If the song doesn't have an audio_url, attempt to fetch it.
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
    # Use a single, comprehensive greetings list
    greetings = ["hi", "hii", "hello", "helo", "hello friend", "hey", "namaste", "hloo", "hlw","how are you"]
    info_triggers = ["tell me", "do you know", "tell me about", "tell me something about", "who", "where", "what", "when", "information about","how"]
    
    # Check for information request
    if any(trigger in input_lower for trigger in info_triggers):
        return "information_request"
    
    # Check for greeting (relaxed condition: any word can be a greeting)
    if any(greeting in input_lower.split() for greeting in greetings):
        return "greeting"
    
    # Check for song request
    song_keywords = ["play", "listen", "song", "music", "suggest", "recommend"]
    if any(keyword in input_lower for keyword in song_keywords):
        return "song_request"
    
    # Default to conversation if no specific intent is matched
    return "conversation"

def is_affirmative(text):
    affirmative_words = ["yes", "yeah", "sure", "yep", "ok", "okay", "of course"]
    return text.strip().lower() in affirmative_words
def is_negative(text):
    negative_words = ["no", "nope", "nah", "not really", "negative","na","not"]
    text = text.strip().lower()
    # For an exact match, you could check:
    if text in negative_words:
        return True
    # Or check if the text includes one of these words:
    return any(word in text.split() for word in negative_words)


@app.route('/')
def index():
    username = session.get('username')
    chat_history = []
    if username:
        # Fetch chat history for the logged-in user, sorted by most recent first
        chat_history_cursor = chats_collection.find({"user_id": username}).sort("created_at", -1) #
        for chat_doc in chat_history_cursor:
            chat_doc['_id'] = str(chat_doc['_id']) # Convert ObjectId to string

            # Handle created_at
            if 'created_at' in chat_doc and isinstance(chat_doc['created_at'], datetime):
                dt_obj = chat_doc['created_at']
                if dt_obj.tzinfo is None:  # If naive, assume it's UTC from MongoDB
                    dt_obj = pytz.utc.localize(dt_obj)
                # Convert to IST for display
                chat_doc['created_at'] = dt_obj.astimezone(ZoneInfo("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S') #

            # Optionally, handle updated_at if you display it in the list
            if 'updated_at' in chat_doc and isinstance(chat_doc['updated_at'], datetime):
                updated_dt_obj = chat_doc['updated_at']
                if updated_dt_obj.tzinfo is None: # If naive, assume it's UTC
                    updated_dt_obj = pytz.utc.localize(updated_dt_obj)
                # Convert to IST for display
                chat_doc['updated_at_display'] = updated_dt_obj.astimezone(ZoneInfo("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S')

            chat_history.append(chat_doc)
    return render_template('index.html', username=username, chat_history=chat_history)

@app.route('/chatting')
@app.route('/chatting/<chat_id>')
def chatting(chat_id=None):
    username = session.get('username')
    chat_messages = []
    all_chats = []
    chat_state = {}

    if username:
        all_chats_cursor = chats_collection.find({"user_id": username}).sort("created_at", -1)
        for chat_item in all_chats_cursor:
            chat_item['_id'] = str(chat_item['_id'])
            if 'created_at' in chat_item and isinstance(chat_item['created_at'], datetime):
                dt_obj = chat_item['created_at']
                if dt_obj.tzinfo is None:  # Naive from Mongo = UTC
                    dt_obj = pytz.utc.localize(dt_obj)
                chat_item['created_at'] = dt_obj.astimezone(ZoneInfo("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S')
            all_chats.append(chat_item)

    if chat_id:
        try:
            chat = chats_collection.find_one({"_id": ObjectId(chat_id), "user_id": username})
        except Exception as e:
            return "Invalid chat ID", 400

        if chat:
            if 'created_at' in chat and isinstance(chat['created_at'], datetime):
                dt_obj = chat['created_at']
                if dt_obj.tzinfo is None:  # Naive from Mongo = UTC
                    dt_obj = pytz.utc.localize(dt_obj)
                chat['created_at'] = dt_obj.astimezone(ZoneInfo("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S')

            if 'updated_at' in chat and isinstance(chat['updated_at'], datetime):
                updated_dt_obj = chat['updated_at']
                if updated_dt_obj.tzinfo is None:  # Naive from Mongo = UTC
                    updated_dt_obj = pytz.utc.localize(updated_dt_obj)
                chat['updated_at_display'] = updated_dt_obj.astimezone(ZoneInfo("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S')

            chat_messages = chat.get('messages', [])
            chat_state = {
                "follow_up_artist": chat.get("follow_up_artist"),
                "follow_up_album": chat.get("follow_up_album"),
                "follow_up_mood": chat.get("follow_up_mood"),
            }
        else:
            app.logger.warning("No chat found with chat_id: %s", chat_id)
            return "Chat not found", 404
    else:
        # New chat session: Clear previous session data
        app.logger.info("No chat id provided, starting a new chat session.")
        session.pop('follow_up_mood', None)
        session.pop('follow_up_artist', None)
        session.pop('follow_up_album', None)
        session.pop('recommended_songs', None)
        session.pop('current_index', None)
        session.pop('played_songs', None)
        # Log session after popping to confirm keys are removed
        app.logger.info("Session after popping keys: %s", {k: v for k, v in session.items() if k.startswith('follow_up')})

    update_session_state(chat_state)
    return render_template('chatPage.html', username=username, chat_id=chat_id, chat_messages=chat_messages, chat_history=all_chats)

def update_session_state(chat_state):
    session.update(chat_state)
    app.logger.info("Session after update: %s", {k: v for k, v in session.items() if k.startswith('follow_up')})

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
@app.route('/save_chat', methods=['POST'])
def save_chat():
    app.logger.info("Session in /save_chat: %s", dict(session))
    data = request.get_json()
    chat_id = data.get('chat_id')
    if not chat_id or chat_id == "None" or chat_id.strip() == "":
        chat_id = None

    # Add this line to log the session state
    app.logger.info("Session before saving: follow_up_mood=%s", session.get("follow_up_mood"))

    state = {
        "recommended_songs": session.get('recommended_songs', []),
        "current_index": session.get('current_index', -1),
        "played_songs": session.get('played_songs', []),
        "follow_up_artist": session.get('follow_up_artist'),
        "follow_up_mood": session.get('follow_up_mood')
    }
    app.logger.info("Saving chat with follow_up_mood=%s", state["follow_up_mood"])

    if chat_id:
        # Update existing chat – this avoids duplicate entries.
        try:
            result = chats_collection.update_one(
                {"_id": ObjectId(chat_id)},
                {"$set": {
                    "messages": data['messages'],
                    "recommended_songs": state["recommended_songs"],
                    "current_index": state["current_index"],
                    "played_songs": state["played_songs"],
                    "follow_up_artist": state["follow_up_artist"],
                    "follow_up_mood": state["follow_up_mood"],
                    "updated_at": datetime.now(ZoneInfo("Asia/Kolkata"))
                }}
            )
            if result.matched_count == 0:
                app.logger.warning("Chat not found for chat_id: %s", chat_id)
                return jsonify({"error": "Chat not found"}), 404
            app.logger.info("Chat updated successfully")
            return jsonify({"status": "success", "message": "Chat updated"}), 200
        except Exception as e:
            app.logger.error("Error updating chat: %s", e)
            return jsonify({"error": str(e)}), 500
    else:
        # Create a new chat since chat_id is not valid.
        data['created_at'] = datetime.now(ZoneInfo("Asia/Kolkata"))
        data["recommended_songs"] = state["recommended_songs"]
        data["current_index"] = state["current_index"]
        data["played_songs"] = state["played_songs"]
        data["follow_up_artist"] = state["follow_up_artist"]
        data["follow_up_mood"] = state["follow_up_mood"]
        
        app.logger.info("Creating new chat with created_at: %s", data['created_at'])
        try:
            result = chats_collection.insert_one(data)
            new_id = str(result.inserted_id)
            app.logger.info("New chat created with id: %s", new_id)
            return jsonify({"status": "success", "id": new_id}), 201
        except Exception as e:
            app.logger.error("Error creating new chat: %s", e)
            return jsonify({"error": str(e)}), 500
 

@app.route('/delete_chat/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    username = session['username']
    try:
        result = chats_collection.delete_one({"_id": ObjectId(chat_id), "user_id": username})
        if result.deleted_count == 1:
            return jsonify({"status": "success"}), 200
        else:
            return jsonify({"error": "Chat not found or not authorized"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/login')
def login_page():
    return render_template('loging.html')  # Ensure the template file is named correctly

@app.route('/get_user')
def get_user():
    username = session.get('username')
    # If you can also store the user_id in session, include that here.
    return jsonify({'user_id': username, 'username': username})

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('recommended_songs', None)
    session.pop('current_index', None)
    session.pop('follow_up_artist', None)
    return redirect(url_for('index'))

def convert_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_sets(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets(i) for i in obj]
    else:
        return obj


def update_session_state(new_data: dict) -> None:
    """
    Merge new session data with existing session state.
    
    This function updates the session object with new values from new_data,
    while preserving any keys that are not affected. Keys with a None value can
    be used to explicitly remove or nullify a setting if desired.
    
    Parameters:
       new_data (dict): A dictionary containing session keys and the new values.
                        Example: {
                          'follow_up_artist': 'kumar sanu',
                          'follow_up_mood': 'romantic',
                          'follow_up_album': None
                        }
    """
    logging.info("Session before merging: %s", dict(session))

    for key, value in new_data.items():
        if value is not None:
            session[key] = value
        else:
            # Option to remove a key or set it to None
            session.pop(key, None)

    logging.info("Session after merging: %s", dict(session))

    
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    # If auto_greeting flag is sent, we ignore user_input.
    auto_greeting = data.get("auto_greeting", False)
    user_input = data.get("user_input", "").strip()
    chat_id = data.get("chat_id", "").strip()
    is_affirmation = data.get("is_affirmation", False)

    # Check negative response – not needed for auto-greeting.
    if not auto_greeting and is_negative(user_input):
        response_text = "Got it, let me know if you need anything else."
        return jsonify({"response": response_text})

    # If auto_greeting is True OR we detect no chat_id (i.e., a new chat), then create a new chat.
    if auto_greeting or not chat_id or chat_id == "None":
        greeting_messages = [
            "Hello there! How can I assist you with your music journey today?",
            "Hi! Ready to explore some new tracks? Let me know what you're in the mood for.",
            "Greetings! I'm here to help you find your favorite song. What are you in the mood for?",
            "Hey! How can I help you with music today?"
        ]
        greeting = random.choice(greeting_messages)
        new_chat = {
            "created_at": datetime.now(ZoneInfo("Asia/Kolkata")),
            "messages": [{
                "sender": "bot",
                "text": greeting,
                "timestamp": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
            }],
            "user_id": session.get("username"),
            "current_index": -1,
            "played_songs": [],
            "follow_up_artist": None,
            "follow_up_album": None,
            "follow_up_mood": None
        }
        try:
            result = chats_collection.insert_one(new_chat)
            new_id = str(result.inserted_id)
            # Return the greeting and the new chat_id 
            return jsonify({"response": greeting, "chat_id": new_id})
        except Exception as e:
            app.logger.error("Error creating new chat: %s", e)
            return jsonify({"error": str(e)}), 500
        
    intent = classify_intent(user_input)
    if intent == "greeting":
        response_text = ("Hello! How can I assist you with music today? "
                         "You can ask for songs by a specific artist, mood, or movie album.")
        return jsonify({"response": response_text})
    
    # If no user input is provided, return an error.
    if not user_input and not is_affirmation:
        return jsonify({"error": "No user input provided."}), 400
      
    # Define specific phrases that unambiguously indicate a personal playlist query.
    specific_playlist_phrases = [
        "my playlist","playlist","my playlist songs","playlist songs" "my liked songs", "my favourite songs", "my favorite songs", "liked songs","favourite songs"
    ]
    
    # Check if the query includes any of these specific phrases.
    playlist_query = any(phrase in user_input.lower() for phrase in specific_playlist_phrases)
    
    # Only for playlist queries, enforce that the user must be logged in.
    user_id = session.get("username")
    if playlist_query:
        if not user_id:
            # Only logged-in users can access personal playlists.
            return jsonify({"error": "You must be logged in to use your playlist"}), 401
        
        # Optionally, check for language filtering.
        preferred_language = None
        for lang in all_languages:
            if lang in user_input.lower():
                preferred_language = lang
                break
              
        liked_song_ids = get_liked_song_ids(user_id)
        
        if liked_song_ids:
            # Assume your music_df contains a field "track_id" that uniquely identifies songs.
            liked_songs_df = music_df[music_df["track_id"].isin(liked_song_ids)]
            
            if preferred_language:
                liked_songs_df = liked_songs_df[liked_songs_df["language"].str.lower() == preferred_language.lower()]
            
            suggestions = liked_songs_df.copy().replace({np.nan: None}).to_dict(orient='records')
            response_msg = "Here are your liked songs." if suggestions else "You have no liked songs matching that language."
        else:
            response_msg = "You have not liked any songs yet."
    
        return jsonify({
            "response": response_msg,
            "suggestions": suggestions if liked_song_ids else []
        })
    
    def filter_disliked_songs(df, user_id):
        if user_id:
            from mongo_integration import get_disliked_song_ids
            disliked_songs = get_disliked_song_ids(user_id)
            return df[~df['track_id'].isin(disliked_songs)]
        return df

    if not is_affirmation:
        extracted_artist, extracted_mood, extracted_language, extracted_album, extracted_song, album_requested, is_info_query = extract_entities_nlp(user_input)
        update_data = {}

        if extracted_album:
            update_data["follow_up_album"] = extracted_album
            # Explicitly set related keys to None if album is specified
            update_data["follow_up_artist"] = None
            update_data["follow_up_mood"] = None
            update_data["follow_up_language"] = None
        else:
            if extracted_artist:
                update_data["follow_up_artist"] = extracted_artist
            if extracted_mood:
                update_data["follow_up_mood"] = extracted_mood
            if extracted_language:
                update_data["follow_up_language"] = extracted_language
            else:
                update_data["follow_up_language"] = None  # or leave unchanged if that's your intent

        update_session_state(update_data)

        app.logger.info("Session after update: follow_up_mood=%s", session.get("follow_up_mood"))

    # Affirmation Branch Processing
    if is_affirmation:
        preferred_language = session.get("preferred_language", "").strip()
        
        # If both follow-up artist and mood exist, filter by both.
        if session.get("follow_up_artist") and session.get("follow_up_mood"):
            artist_filter = session.get("follow_up_artist")
            mood_filter = session.get("follow_up_mood")
            additional_songs = music_df[
                (music_df['artist_name'].str.lower() == artist_filter.lower()) &
                (music_df['mood_label'].str.contains(mood_filter, case=False, na=False))
            ]
            response_msg = f"Here are more {preferred_language.title()} songs by {artist_filter.title()} that match your mood."

        # If only follow-up artist exists (and no mood)
        elif session.get("follow_up_artist"):
            artist_filter = session.get("follow_up_artist")
            additional_songs = music_df[music_df['artist_name'].str.lower() == artist_filter.lower()]
            response_msg = f"Here are more {preferred_language.title()} songs by {artist_filter.title()}."
        
        # If no artist, but an album filter exists:
        elif session.get("follow_up_album"):
            album_filter = session.get("follow_up_album")
            additional_songs = music_df[music_df['album_movie_name'].str.lower() == album_filter.lower()]
            response_msg = f"Here are more songs from the album {album_filter.title()}."
        
        # If only a mood filter exists:
        elif session.get("follow_up_mood"):
            mood_filter = session.get("follow_up_mood")
            additional_songs = music_df[music_df['mood_label'].str.contains(mood_filter, case=False, na=False)]
            response_msg = f"Here are more {preferred_language.title()} songs that match your mood."
        
        else:
            return jsonify({"response": "No further recommendations available.", "suggestions": []})
        
        # Apply the preferred language filter if necessary:
        if preferred_language:
            additional_songs = additional_songs[additional_songs['language'].str.lower() == preferred_language.lower()]
        
        # Remove previously recommended songs:
        prev_recs = session.get("recommended_songs", [])
        prev_song_names = [song["song_name"].strip().lower() for song in prev_recs]
        additional_songs = additional_songs[~additional_songs['song_name'].str.strip().str.lower().isin(prev_song_names)]
        
        # Deduplicate and limit to 5 suggestions:
        additional_songs = additional_songs.drop_duplicates(subset=['song_name'])
        if len(additional_songs) > 5:
            additional_songs = additional_songs.sample(n=5)
        
        suggestions = additional_songs.copy().replace({np.nan: None}).to_dict(orient='records')
        session["recommended_songs"] = suggestions
        session["current_index"] = -1
        session.modified = True

        return jsonify({"response": response_msg, "suggestions": suggestions})


    # Normal Query Branch
    artist, mood, language, album, query_song, album_requested, is_info_query = extract_entities_nlp(user_input)
    filtered = music_df.copy()
    follow_up_msg = ""
    matched_songs = []

    if query_song:
        if artist:
            filtered = filtered[(filtered['song_name'].str.lower() == query_song.lower()) & 
                                (filtered['artist_name'].str.lower() == artist.lower())]
        else:
            filtered = filtered[filtered['song_name'].str.lower() == query_song.lower()]
        if not filtered.empty:
            specific_song = filtered.iloc[0].to_dict()
            session["follow_up_mood"] = specific_song['mood_label']
            session.pop("follow_up_artist", None)
            session.pop("follow_up_album", None)
            session["was_specific_song"] = True
            matched_songs = [specific_song]
            follow_up_msg = "Would you like to listen to more songs like this?"
        else:
            matched_songs = []

    elif album:
        filtered = filtered[filtered['album_movie_name'].str.lower() == album.lower()]
        if artist:
            filtered = filtered[filtered['artist_name'].str.lower() == artist.lower()]
        if mood:
            filtered = filtered[filtered['mood_label'].str.contains(mood, case=False, na=False)]
        if language:
            filtered = filtered[filtered['language'].str.lower() == language.lower()]
        if not filtered.empty:
            session["follow_up_album"] = album
            session.pop("follow_up_artist", None)
            session.pop("follow_up_mood", None)
            session["was_specific_song"] = False
            follow_up_msg = f"Would you like to listen to more songs from the album {album.title()}?"
        matched_songs = filtered.copy().replace({np.nan: None}).to_dict(orient='records')

    elif not query_song and not album:
        if artist:
            # Using substring matching:
            filtered = filtered[filtered['artist_name'].str.lower().str.contains(artist.lower(), regex=False)]
            session["follow_up_artist"] = artist
            session.pop("follow_up_album", None)
            if mood:
                session["follow_up_mood"] = mood
            session["was_specific_song"] = False
            follow_up_msg = f"Would you like to listen to more songs by {artist.title()}?"

            # Update follow_up_mood only if present in the query.
            if mood:
                session["follow_up_mood"] = mood
            session["was_specific_song"] = False
            follow_up_msg = f"Would you like to listen to more songs by {artist.title()}?"

        if mood:
            filtered = filtered[filtered['mood_label'].str.contains(mood, case=False, na=False)]
        if language:
            filtered = filtered[filtered['language'].str.lower() == language.lower()]

            matched_songs = filtered.copy().replace({np.nan: None}).to_dict(orient='records')
        else:
            matched_songs = []

    user_id = session.get("username")
    filtered = filter_disliked_songs(filtered, user_id)
    matched_songs = filtered.copy().replace({np.nan: None}).to_dict(orient='records') if not matched_songs else matched_songs

    if language:
        session["preferred_language"] = language
    elif matched_songs:
        session["preferred_language"] = matched_songs[0]['language']

    # After filtering and obtaining matched_songs from your music_df

    def deduplicate_songs(songs):
        """
        Remove duplicate songs based on a normalized song name.
        """
        unique = []
        seen = set()
        for song in songs:
            # Use a normalized song name (or track id if available) to determine uniqueness
            song_id = song['song_name'].strip().lower()
            if song_id not in seen:
                unique.append(song)
                seen.add(song_id)
        return unique

    def filter_played_songs(songs, played_set):
        """
        Filter out songs whose normalized names are in played_set.
        """
        return [song for song in songs if song['song_name'].strip().lower() not in played_set]

    if not matched_songs:
        response_text = "Sorry, I couldn’t find any songs matching your request."
    else:
        # Deduplicate the matched songs
        matched_songs = deduplicate_songs(matched_songs)
        
        # Retrieve already played songs from the session (stored as lower-case strings)
        played_songs = session.get('played_songs', [])
        played_set = set(song.lower() for song in played_songs)
        
        # Filter out played songs from the deduplicated list
        matched_songs = filter_played_songs(matched_songs, played_set)
        
        # Limit to 2 recommendations if there are more than needed
        if len(matched_songs) > 5:
            matched_songs = random.sample(matched_songs, 5)
        
        session['recommended_songs'] = matched_songs
        session['current_index'] = -1
        session.modified = True

        if is_info_query:
            # Information branch: build an information prompt and return early
            prompt = construct_prompt(user_input, "information_request", artist, mood, language, None)
            try:
                response = gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                )
                response_text = response.text.strip()
            except Exception as e:
                app.logger.error("Error in Gemini API: %s", e)
                response_text = "Sorry, there was an error processing your request."

            return jsonify({"response": response_text})
        else:
            prompt = construct_prompt(user_input, "song_request", artist, mood, language, matched_songs)
            if follow_up_msg:
                prompt += " " + follow_up_msg
            try:
                response = gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                )
                response_text = response.text.strip()
            except Exception as e:
                app.logger.error("Error in Gemini API: %s", e)
                response_text = "Sorry, there was an error processing your request."


    return jsonify({"response": response_text, "suggestions": matched_songs})

def deduplicate_songs(songs):
    """
    Remove duplicate songs based on a normalized song name.
    This function ensures that the same song (even if it appears with different moods)
    appears only once in the returned list.
    """
    unique = {}
    for song in songs:
        norm_name = normalize_text(song['song_name'])
        if norm_name not in unique:
            unique[norm_name] = song
    return list(unique.values())


@app.route('/next_song', methods=['GET'])
def next_song():
    # ---------------------------
    # Playlist Mode: if a personal playlist (liked songs) is in session, use that branch.
    if 'liked_songs' in session and session.get('liked_songs'):
        liked_songs = session.get('liked_songs')
        current_index = session.get('current_playlist_index', 0)
        
        if current_index >= len(liked_songs):
            # Reset the pointer; alternatively, you might choose to shuffle
            current_index = 0
        
        next_song_data = liked_songs[current_index]
        session['current_playlist_index'] = current_index + 1
        session.modified = True
        
        response_text = f"Now Playing (Playlist Mode): {next_song_data['song_name']} by {next_song_data['artist_name']}"
        return jsonify({'response': response_text, 'song': next_song_data})
    
    # ---------------------------
    # Regular Recommendation Mode:
    recommended_songs = session.get('recommended_songs', [])
    # Deduplicate recommended songs first
    recommended_songs = deduplicate_songs(recommended_songs)
    
    if not recommended_songs:
        return jsonify({'response': "No songs in the queue. Please request some songs first!"}), 400

    # Get the list of already played songs (normalized)
    played_songs = session.get('played_songs', [])
    played_set = set(normalize_text(song) for song in played_songs)
    
    # Filter out any songs already played
    remaining_songs = [song for song in recommended_songs if normalize_text(song['song_name']) not in played_set]
    
    if not remaining_songs:
        return jsonify({'response': "You've reached the end of the song list. Want more recommendations?"}), 200

    # Select the next song (e.g., the first one in the filtered list)
    next_song_data = remaining_songs[0]
    norm_name = normalize_text(next_song_data['song_name'])
    
    # Add the normalized song name to the played songs
    played_songs.append(norm_name)
    
    # Also, remove all songs with this normalized name from the recommendation list
    recommended_songs = [song for song in recommended_songs if normalize_text(song['song_name']) != norm_name]
    
    session['played_songs'] = played_songs
    session['recommended_songs'] = recommended_songs
    session.modified = True

    # Refresh the audio URL if necessary.
    if not next_song_data.get('audio_url') or next_song_data.get('audio_url') in [None, '', 'undefined']:
        app.logger.info("Audio URL missing for next song '%s' by '%s'. Fetching a fresh URL.",
                        next_song_data.get("song_name"), next_song_data.get("artist_name"))
        audio_url, thumbnail_url = get_stream_url(next_song_data["song_name"], next_song_data["artist_name"])
        next_song_data["audio_url"] = audio_url
        next_song_data["thumbnail_url"] = thumbnail_url
        session['recommended_songs'] = recommended_songs
        session.modified = True

    response_text = f"Next song: {next_song_data['song_name']} by {next_song_data['artist_name']}"
    return jsonify({'response': response_text, 'song': next_song_data})

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
        return render_template('loging.html')  # Ensure the template filename is correct.
    
    elif request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
    
        if not (username and password):
            return jsonify({"error": "Credentials required"}), 400
        message = login_user(username, password)
    
        if message == "Login successful":
            session["username"] = username
            session["email"] = get_email(username)  # Stores the email which will be needed later
            session["subscription"] = get_subscription(username)
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

@app.route('/upgrade', methods=['GET'])
def upgrade():
    username = session["username"]
    email = session["email"]
    if not email:
        flash("Please log in to upgrade!", "error")
        return redirect('/login')
    
    try:
        subscription_mail(email)
        sub_stat = update_subscription_status(username)
        session["subscription"] = sub_stat
        flash("🎉 You've been upgraded to Premium! Check your mail.", "success")
    except Exception as e:
        flash(f"Error sending mail: {str(e)}", "error")
    
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)