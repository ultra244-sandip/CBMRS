import re
import unicodedata
import string
from rapidfuzz import process, fuzz
import spacy
from spacy.matcher import PhraseMatcher

# — trigger lists —
INFO_TRIGGERS = [
    "tell me", "do you know", "tell me about",
    "tell me something about", "who ", "where ",
    "what ", "when ", "information about"
]
RECOMMEND_TRIGGERS = ["suggest", "recommend", "some", "give me"]
PLAY_TRIGGERS = ["play", "show me", "play the song", "play the movie"]

# Load spaCy

try:
    nlp = spacy.load("en_core_web_sm")
    print("[DEBUG] spaCy model loaded.")
except Exception as e:
    print(f"[ERROR] spaCy load failed: {e}")
    nlp = None



class EntityExtractor:
    def __init__(self, all_languages, all_moods, all_artists, all_albums, all_songs):
        global nlp
        self.languages = [l.lower() for l in all_languages]
        self.moods = [m.lower() for m in all_moods]
        self.artists = [a.lower() for a in all_artists]
        self.albums = [a.lower() for a in all_albums]
        self.songs = [s.lower() for s in all_songs]
        self.filler_words = {"play", "song", "songs", "recommend", "suggest", "please", "of", "by"}
        
        # Setup spaCy matchers (assuming nlp is globally loaded)
        if nlp:
            self.lang_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
            self.lang_matcher.add("LANGUAGE", [nlp.make_doc(text) for text in self.languages])
            self.mood_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
            self.mood_matcher.add("MOOD", [nlp.make_doc(text) for text in self.moods])
        else:
            self.lang_matcher = None
            self.mood_matcher = None
    def robust_fuzzy_match(self, token: str, choices: list, primary_cutoff: int = 80, secondary_cutoff: int = 60):
        match = process.extractOne(token, choices, scorer=fuzz.token_set_ratio, score_cutoff=primary_cutoff)
        if match is None:
            match = process.extractOne(token, choices, scorer=fuzz.token_set_ratio, score_cutoff=secondary_cutoff)
        return match[0] if match else None

    def extract(self, user_input: str) -> dict:
        """
        Extracts entities (artist, mood, language, album, song) from the user input.
        Uses spaCy for mood/language recognition, regex to capture segments after "by"/"of",
        robust fuzzy matching for spelling mistakes, and attempts song extraction only if no
        artist/album is detected.
        """
        print(f"[DEBUG] Original query: '{user_input}'")
        original_query = user_input.strip()
        normalized_query = original_query.lower()
        low_query = normalized_query

        # Initialize extracted entities.
        found_language = found_mood = found_artist = found_album = found_song = None
        album_requested = False
        is_info_query = False
        is_artist_request = False

        # Process with spaCy.
        doc = nlp(user_input)

        # Use PhraseMatcher for mood.
        if self.mood_matcher:
            mood_matches = self.mood_matcher(doc)
            if mood_matches:
                found_mood = doc[mood_matches[0][1]:mood_matches[0][2]].text.lower()
                print(f"[DEBUG] Found mood: '{found_mood}'")
        # Use PhraseMatcher for language.
        if self.lang_matcher:
            lang_matches = self.lang_matcher(doc)
            if lang_matches:
                found_language = doc[lang_matches[0][1]:lang_matches[0][2]].text.lower()
                print(f"[DEBUG] Found language: '{found_language}'")

        # Check if info query.
        info_triggers = [
            "tell me", "do you know", "tell me about", "who", "where", "what", "when", "information about"
        ]
        if any(trigger in low_query for trigger in info_triggers):
            is_info_query = True
            print("[DEBUG] Info query detected; skipping song/album extraction.")
            return {
                "artist": None,
                "mood": found_mood,
                "language": found_language,
                "album": None,
                "song": None,
                "album_requested": False,
                "is_info_query": True,
                "is_artist_request": False
            }

        # ----- Album extraction section -----
        if re.search(r'\b(movie|album)\b', low_query, re.IGNORECASE):
            album_requested = True
            album_match = re.search(r'from\s+(.+?)\s+(?:movie|album)', low_query, re.IGNORECASE)
            if album_match:
                album_candidate = album_match.group(1).strip()
                print(f"[DEBUG] Album candidate (pattern): '{album_candidate}'")
            else:
                command_words = ['play', 'recommend', 'suggest', 'find', 'please']
                temp_query = original_query
                for word in command_words:
                    temp_query = temp_query.replace(word, '')
                temp_query = temp_query.strip()
                album_stopwords = {"movie", "album", "songs", "song", "of", "by"}
                tokens_temp = temp_query.split()
                album_tokens = [tok for tok in tokens_temp if tok.lower() not in album_stopwords]
                album_candidate = " ".join(album_tokens).strip()
                print(f"[DEBUG] Album candidate (fallback): '{album_candidate}'")
            if album_candidate:
                best_album = process.extractOne(album_candidate, self.albums, scorer=fuzz.ratio, score_cutoff=60)
                if best_album:
                    found_album = best_album[0]
                    print(f"[DEBUG] Found album: '{found_album}' with score {best_album[1]}")
                else:
                    print("[DEBUG] No album match found.")

        # ----- Artist extraction section -----
        # If an album query is detected, skip further artist extraction.
        if not album_requested:
            # First try using spaCy's NER for PERSON entities.
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    candidate = ent.text
                    best_artist = process.extractOne(candidate, self.artists, scorer=fuzz.token_set_ratio, score_cutoff=60)
                    if best_artist:
                        found_artist = best_artist[0]
                        print(f"[DEBUG] Extracted artist via NER: '{found_artist}' with score {best_artist[1]}")
                        break
            # If no artist is found using NER, try a regex-based approach.
            if not found_artist:
                # Updated regex pattern: capture text after "by" or "of" until the end or a boundary keyword.
                artist_pattern = re.compile(r"\b(?:by|of)\s+(.+?)(?=$|\s+(?:in|with|song|songs))", re.IGNORECASE)
                match = artist_pattern.search(original_query)
                if match:
                    artist_candidate = match.group(1).strip()
                    best_artist = process.extractOne(artist_candidate, self.artists, scorer=fuzz.token_set_ratio, score_cutoff=60)
                    if best_artist:
                        found_artist = best_artist[0]
                        print(f"[DEBUG] Extracted artist from pattern: '{found_artist}' with score {best_artist[1]}")
                # Alternative pattern if needed.
                if not found_artist:
                    artist_pattern_alt = re.compile(r"\bplay\s+([a-zA-Z\s]+?)\s+songs?\b", re.IGNORECASE)
                    alt_match = artist_pattern_alt.search(original_query)
                    if alt_match:
                        artist_candidate = alt_match.group(1).strip()
                        filler = {"some", "any"}
                        candidate_tokens = artist_candidate.split()
                        filtered_tokens = [token for token in candidate_tokens 
                                             if token.lower() not in self.moods 
                                             and token.lower() not in self.languages 
                                             and token.lower() not in filler]
                        filtered_candidate = " ".join(filtered_tokens).strip()
                        if filtered_candidate:
                            best_artist = process.extractOne(filtered_candidate, self.artists, scorer=fuzz.token_set_ratio, score_cutoff=70)
                            if best_artist:
                                found_artist = best_artist[0]
                                print(f"[DEBUG] Extracted artist from alternative pattern: '{found_artist}' with score {best_artist[1]}")

            # --- NEW BRANCH: Handle multiple artists in the extracted string ---
            if found_artist and ("," in found_artist or "&" in found_artist or " and " in found_artist.lower()):
                # Split using commas, ampersands, or 'and'
                individual_artists = re.split(r",|&|\band\b", found_artist)
                individual_artists = [artist.strip() for artist in individual_artists if artist.strip()]
                print(f"[DEBUG] Split extracted artist into: {individual_artists}")
                # If we have a candidate extracted from the by/of pattern, use it for fuzzy matching.
                if 'extracted_candidate' in locals() and extracted_candidate:
                    best_individual = max(individual_artists, key=lambda a: fuzz.ratio(extracted_candidate, a.lower()))
                    found_artist = best_individual
                    print(f"[DEBUG] Selected primary artist from split using candidate '{extracted_candidate}': '{found_artist}'")
                else:
                    # Otherwise, simply choose the first name.
                    found_artist = individual_artists[0]
                    print(f"[DEBUG] Selected primary artist from split: '{found_artist}'")


        # Remove the raw extracted candidate from normalized query (if artist was captured) to prevent interfering with song extraction.
        if not album_requested:
            extracted_candidate = None
            m = re.search(r"\b(?:by|of)\s+(.+?)(?=$|\s+(?:in|with|song|songs))", original_query, re.IGNORECASE)
            if m:
                extracted_candidate = m.group(1).strip().lower()
                print(f"[DEBUG] Extracted candidate (by/of): '{extracted_candidate}'")
                normalized_query = normalized_query.replace(extracted_candidate, "")

        # ----- Candidate song extraction -----
        if album_requested:
            candidate_song = ""
            is_artist_request = False
        else:
            query_tokens = low_query.split()
            if found_artist:
                artist_tokens = found_artist.split()
                tokens_filtered = []
                for token in query_tokens:
                    remove = False
                    for art_tok in artist_tokens:
                        if fuzz.ratio(token, art_tok.lower()) > 80:
                            remove = True
                            break
                    if not remove:
                        tokens_filtered.append(token)
                query_without_artist = " ".join(tokens_filtered)
            else:
                query_without_artist = low_query

            filter_tokens = {"of", "by", "song", "songs", "me", "play", "recommend",
                            "suggest", "find", "please", "the", "a", "an"}
            candidate_tokens = [tok.translate(str.maketrans('', '', string.punctuation)).lower()
                                for tok in query_without_artist.split()
                                if tok.translate(str.maketrans('', '', string.punctuation)).lower() not in filter_tokens]
            if found_language:
                candidate_tokens = [tok for tok in candidate_tokens if tok != found_language]
            if found_mood:
                candidate_tokens = [tok for tok in candidate_tokens if tok != found_mood]
            candidate_song = " ".join(candidate_tokens).strip()

        print(f"[DEBUG] Candidate song string: '{candidate_song}'")

        filler_words = {"some", "any"}
        if candidate_song.lower() in filler_words or candidate_song == "":
            print("[DEBUG] No valid candidate song extracted; treating as artist/mood-only query.")
            if found_artist is None and found_mood is not None:
                is_artist_request = False
            else:
                is_artist_request = True
            found_song = None
        else:
            candidate_token_count = len(candidate_song.split())
            score_cutoff = 90 if candidate_token_count == 1 else 80
            if candidate_song in self.songs:
                found_song = candidate_song
                print(f"[DEBUG] Found song by exact match: '{found_song}'")
            else:
                best_match = process.extractOne(candidate_song, self.songs, scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff)
                if best_match:
                    found_song = best_match[0]
                    print(f"[DEBUG] Found song: '{found_song}' with score {best_match[1]}")
                else:
                    print("[DEBUG] No matching song found via fuzzy search.")
                    found_song = None
            is_artist_request = False

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

        return {
            "artist": found_artist,
            "mood": found_mood,
            "language": found_language,
            "album": found_album,
            "song": found_song,
            "album_requested": album_requested,
            "is_info_query": is_info_query,
            "is_artist_request": is_artist_request
        }

