import pymongo
from config import app_config


# Generic functions
def get_db_client():
    """Returns MongoDB client object, connect to MongoDB Atlas instances if required"""
    if not app_config.MONGO_CONN_STR:
        return None
    try:
        if app_config.mongo_client == None:
            client = pymongo.MongoClient(app_config.MONGO_CONN_STR, serverSelectionTimeoutMS=5000)
            app_config.mongo_client = client
    except Exception as e:
        print(f"MongoDB connection error: {e}")
    return app_config.mongo_client


def fetch_document(client, db, collection):
    """Get a single document from the provided db and collection"""
    document = None
    try:
        document = client[db][collection].find_one()
    except Exception as e:
        print(e)
    return document


def update_document(client, db, collection, key, value):
    """Update the passed key in the document for provided db and collection"""
    try:
        document = fetch_document(client, db, collection)
        client[db][collection].update_one(
            {"_id": document["_id"]},
            {"$set": {key: value}},
        )
    except Exception as e:
        print(e)


# Use case specific functions
def fetch_curr_access_count():
    try:
        client = get_db_client()
        if client:
            document = fetch_document(
                client=client, db=app_config.db, collection=app_config.collection
            )
            if document:
                curr_count = document[app_config.key]
                app_config.openai_curr_access_count = curr_count
                return
    except Exception as e:
        print(f"Failed to fetch count from MongoDB: {e}")
    
    # Fallback to local count if MongoDB fails
    if app_config.openai_curr_access_count is None:
        app_config.openai_curr_access_count = 0


def increment_curr_access_count():
    try:
        client = get_db_client()
        updated_count = app_config.openai_curr_access_count + 1
        if client:
            update_document(
                client=client,
                db=app_config.db,
                collection=app_config.collection,
                key=app_config.key,
                value=updated_count,
            )
    except Exception as e:
        print(f"Failed to increment count in MongoDB: {e}")
    
    app_config.openai_curr_access_count += 1
