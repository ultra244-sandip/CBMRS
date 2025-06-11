from pymongo import MongoClient, ASCENDING
from pymongo.errors import ServerSelectionTimeoutError
import os
from flask import Blueprint, request, jsonify
import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a blueprint to register MongoDB-related endpoints
mongo_bp = Blueprint('mongo_bp', __name__)

# Connect to MongoDB
MONGO_URI = os.getenv("MONGO_URI", "your-default-mongo-uri")
try:
    client = MongoClient(MONGO_URI)
    db = client["chat_app_db"]
except ServerSelectionTimeoutError as e:
    print(f"Failed to connect to MongoDB: {e}")
    exit(1)

# Define collections for chat sessions and feedback
chats_collection = db["chats"]
feedback_collection = db["feedback"]

# Create indexes for faster queries
chats_collection.create_index([("user_id", ASCENDING)])
feedback_collection.create_index([("user_id", ASCENDING), ("song_id", ASCENDING)], unique=True)

@mongo_bp.route('/get_chats/<user_id>', methods=['GET'])
def get_chats(user_id):
    chats = list(chats_collection.find({"user_id": user_id}))
    for chat in chats:
        chat['_id'] = str(chat['_id'])
    return jsonify(chats), 200

@mongo_bp.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    if not data or 'user_id' not in data or 'song_id' not in data or 'feedback' not in data:
        return jsonify({"error": "Missing required fields (user_id, song_id, feedback)"}), 400
    data['submitted_at'] = datetime.datetime.utcnow()
    result = feedback_collection.update_one(
        {"user_id": data["user_id"], "song_id": data["song_id"]},
        {"$set": data},
        upsert=True
    )
    return jsonify({"status": "success"}), 201

@mongo_bp.route('/get_feedback/<user_id>', methods=['GET'])
def get_feedback(user_id):
    feedback = list(feedback_collection.find({"user_id": user_id}))
    for fb in feedback:
        fb["_id"] = str(fb["_id"])
    return jsonify(feedback), 200

def get_disliked_song_ids(user_id):
    """Return a list of song_id's that the user has disliked."""
    disliked = feedback_collection.find({"user_id": user_id, "feedback": "dislike"})
    return [doc["song_id"] for doc in disliked]

def get_liked_song_ids(user_id):
    """Return a list of song_id's that the user has liked."""
    liked = feedback_collection.find({"user_id": user_id, "feedback": "like"})
    return [doc["song_id"] for doc in liked]
