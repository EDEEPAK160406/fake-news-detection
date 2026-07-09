from typing import Optional

from pymongo import MongoClient

from app.core.config import settings

_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.mongo_uri)
    return _client


def get_database():
    return get_client()[settings.mongo_db_name]


def prediction_collection():
    return get_database()["predictions"]


def feedback_collection():
    return get_database()["feedback"]


def image_authenticity_collection():
    return get_database()["image_authenticity"]


def video_authenticity_collection():
    return get_database()["video_authenticity"]
