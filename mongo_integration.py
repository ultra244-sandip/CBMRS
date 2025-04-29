# mongo_integration.py
from pymongo import MongoClient, ASCENDING
import os
from flask import Blueprint, request, jsonify
import datetime

# Create a blueprint to register MongoDB-related endpoints
mongo_bp = Blueprint('mongo_bp', __name__)

# Connect to MongoDB Atlas. Make sure your MONGO_URI environment variable is set.
MONGO_URI = os.getenv("MONGO_URI", "your-default-mongo-uri")
client = MongoClient(MONGO_URI)
db = client["chat_app_db"]  # Choose a database name

# Define collections for chat sessions and feedback.
chats_collection = db["chats"]
feedback_collection = db["feedback"]

# Create indexes for faster queries.
chats_collection.create_index([("user_id", ASCENDING)])
feedback_collection.create_index([("user_id", ASCENDING), ("song_id", ASCENDING)], unique=True)

@mongo_bp.route('/save_chat', methods=['POST'])
def save_chat():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    if 'created_at' not in data:
        data['created_at'] = datetime.datetime.utcnow()
    result = chats_collection.insert_one(data)
    return jsonify({"status": "success", "id": str(result.inserted_id)}), 201

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
